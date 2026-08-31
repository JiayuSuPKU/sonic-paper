"""Download and quantify T47D TRPS1-degradation PRO-seq near ANKRD30A.

Source: GEO accession GSE236172, "T47D dTAG-TRPS1 C28 Time-course
PRO-seq". The workflow opens the processed timepoint-combined strand-specific
bigWigs remotely, writes small ANKRD30A-region bigWigs locally, and quantifies
combined plus/minus signal around the four PB-defined ANKRD30A TSSs.

Idempotent pipeline:

1. Load PB84, PB2, PB56, and PB107 transcript/TSS coordinates from the
   workflow-generated ANKRD30A transcript visualization cache.
2. Write the sample manifest to
   ``results/splisosm_atera_brca/ankrd30a_trps1_proseq/``.
3. Download only the required ANKRD30A genomic interval from each remote GEO
   bigWig into ``data/atera_ffpe_human_brca/ankrd30a_trps1_proseq/``.
4. Quantify summed signal in +/-1 kb TSS windows and the fixed ANKRD30A gene
   body interval.
5. Cache the signal-window table and per-timepoint signal-change table.

The processed GEO bigWigs are already combined across replicates, so the
resulting signal changes are descriptive effect sizes, not replicate-level
statistical tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "scripts.splisosm_atera_brca.ankrd30a"

import numpy as np
import pandas as pd
import pyBigWig

from ..config import (
    ANKRD30A_TRANSCRIPT_ANNOTATIONS_CSV,
    ANKRD30A_TRPS1_PROSEQ_DATA_DIR,
    ANKRD30A_TRPS1_PROSEQ_RESULT_DIR,
    ANKRD30A_TRPS1_PROSEQ_SAMPLE_MANIFEST_CSV,
    ANKRD30A_TRPS1_PROSEQ_SIGNAL_CSV,
    ANKRD30A_TRPS1_PROSEQ_SIGNAL_WINDOWS_CSV,
)
from ..utils import log, project_path, save_csv
from .common import (
    DEFAULT_SLICE_REGION,
    PB_EVENT_CLASS_BY_TRANSCRIPT,
    PB_TRANSCRIPT_SHORT_LABELS,
    TARGET_GENE,
    TRPS1_PROSEQ_GEO_ACCESSION,
    TRPS1_PROSEQ_SIGNAL_EPSILON,
    TRPS1_PROSEQ_SOURCE_BASE_URL,
    TRPS1_PROSEQ_TIMEPOINTS,
    TRPS1_PROSEQ_TSS_HALF_WINDOW_BP,
    default_slice_region_1based,
)


def _ensure_dirs() -> None:
    """Create the Atera-local PRO-seq data and result roots."""
    ANKRD30A_TRPS1_PROSEQ_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ANKRD30A_TRPS1_PROSEQ_RESULT_DIR.mkdir(parents=True, exist_ok=True)


def build_sample_manifest() -> pd.DataFrame:
    """Return expected local ANKRD30A-region GSE236172 bigWig paths."""
    rows: list[dict[str, object]] = []
    # One processed GEO track pair exists per timepoint: plus and minus strand,
    # already combined across biological replicates by the source study.
    for order, (timepoint, time_hours, geo_timepoint) in enumerate(TRPS1_PROSEQ_TIMEPOINTS):
        pos_name = (
            f"{TRPS1_PROSEQ_GEO_ACCESSION}_T47D_{geo_timepoint}_plus_combined_normalized.bigWig"
        )
        neg_name = (
            f"{TRPS1_PROSEQ_GEO_ACCESSION}_T47D_{geo_timepoint}_minus_combined_normalized.bigWig"
        )
        rows.append(
            {
                "sample_id": TRPS1_PROSEQ_GEO_ACCESSION,
                "sample_label": f"{timepoint}_combined",
                "timepoint": timepoint,
                "time_hours": time_hours,
                "timepoint_order": order,
                "replicate": "combined",
                "pos_bigwig_file": pos_name,
                "neg_bigwig_file": neg_name,
                "pos_bigwig_url": f"{TRPS1_PROSEQ_SOURCE_BASE_URL}{pos_name}",
                "neg_bigwig_url": f"{TRPS1_PROSEQ_SOURCE_BASE_URL}{neg_name}",
                "pos_bigwig_path": str(ANKRD30A_TRPS1_PROSEQ_DATA_DIR / pos_name),
                "neg_bigwig_path": str(ANKRD30A_TRPS1_PROSEQ_DATA_DIR / neg_name),
            }
        )
    return pd.DataFrame(rows)


def _selected_transcripts() -> pd.DataFrame:
    """Load the four PB transcript rows that define the ANKRD30A TSS event."""
    if not ANKRD30A_TRANSCRIPT_ANNOTATIONS_CSV.exists():
        raise FileNotFoundError(
            "Missing ANKRD30A transcript cache: "
            f"{project_path(ANKRD30A_TRANSCRIPT_ANNOTATIONS_CSV)}"
        )
    annotations = pd.read_csv(ANKRD30A_TRANSCRIPT_ANNOTATIONS_CSV)
    # The visualization cache is feature-level; keep only transcript rows so
    # each PB ID contributes one TSS and one transcript span.
    transcripts = annotations[annotations["feature"].astype(str).eq("transcript")]
    selected = transcripts[
        transcripts["transcript_id"].astype(str).isin(PB_EVENT_CLASS_BY_TRANSCRIPT)
    ].copy()
    if selected.empty:
        raise ValueError("No selected PB.3828 transcript rows found in Atera cache.")
    for column in ("start_1based", "end_1based", "tss_1based"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    return selected


def build_ankrd30a_signal_windows() -> pd.DataFrame:
    """Build the PB TSS and gene-body windows used for PRO-seq quantification."""
    tx = _selected_transcripts()
    rows: list[dict[str, object]] = []
    plot_order = {transcript_id: i for i, transcript_id in enumerate(PB_EVENT_CLASS_BY_TRANSCRIPT)}
    strands = tx["strand"].dropna().astype(str).unique()
    gene_strand = strands[0] if len(strands) else "+"
    for _, row in tx.iterrows():
        transcript_id = str(row["transcript_id"])
        event_class = PB_EVENT_CLASS_BY_TRANSCRIPT[transcript_id]
        tss = int(row["tss_1based"])
        # pyBigWig uses half-open 0-based intervals; cached annotations retain
        # 1-based coordinates for human-readable reporting and plotting.
        start_0 = max(tss - TRPS1_PROSEQ_TSS_HALF_WINDOW_BP - 1, 0)
        end_0 = tss + TRPS1_PROSEQ_TSS_HALF_WINDOW_BP
        rows.append(
            {
                "feature_id": transcript_id,
                "feature_type": "pb_tss",
                "plot_order": plot_order[transcript_id],
                "display_label": (
                    f"{PB_TRANSCRIPT_SHORT_LABELS[transcript_id]} TSS ({event_class})"
                ),
                "event_class": event_class,
                "chrom": row["chrom"],
                "strand": row.get("strand", gene_strand),
                "anchor_1based": tss,
                "start_0based": start_0,
                "end_0based": end_0,
                "start_1based": start_0 + 1,
                "end_1based": end_0,
                "window_length_bp": end_0 - start_0,
                "window_label": (f"+/-{TRPS1_PROSEQ_TSS_HALF_WINDOW_BP:,} bp around PB TSS"),
            }
        )

    gene_chrom, gene_start, gene_end = default_slice_region_1based()
    # Gene-body signal is a broad control: it uses the same fixed ANKRD30A
    # interval as TCGA BAM slicing and TCGA gene-level count normalization.
    rows.append(
        {
            "feature_id": f"{TARGET_GENE}_gene_body",
            "feature_type": "gene_body",
            "plot_order": len(PB_EVENT_CLASS_BY_TRANSCRIPT),
            "display_label": f"{TARGET_GENE} gene body",
            "event_class": "control",
            "chrom": gene_chrom,
            "strand": gene_strand,
            "anchor_1based": np.nan,
            "start_0based": gene_start - 1,
            "end_0based": gene_end,
            "start_1based": gene_start,
            "end_1based": gene_end,
            "window_length_bp": gene_end - gene_start + 1,
            "window_label": DEFAULT_SLICE_REGION,
        }
    )
    return pd.DataFrame(rows).sort_values(["plot_order"])


def _ankrd30a_download_region(windows: pd.DataFrame) -> dict[str, object]:
    """Return the minimal genomic span needed by all signal windows."""
    return {
        "chrom": str(windows["chrom"].iloc[0]),
        "start_0based": int(windows["start_0based"].min()),
        "end_0based": int(windows["end_0based"].max()),
    }


def _bigwig_chrom_name(bw: Any, chrom: str) -> str:
    """Return the chromosome token accepted by one bigWig."""
    chroms = bw.chroms()
    if chrom in chroms:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in chroms:
        return chrom[3:]
    prefixed = f"chr{chrom}"
    if prefixed in chroms:
        return prefixed
    raise ValueError(f"Chromosome {chrom} was not found in source bigWig.")


def _nonzero_runs(
    values: np.ndarray, start_0based: int
) -> tuple[list[int], list[int], list[float]]:
    """Collapse dense per-base values into nonzero runs for compact bigWigs."""
    starts: list[int] = []
    ends: list[int] = []
    out_values: list[float] = []
    run_start: int | None = None
    run_value: float | None = None

    for offset, raw_value in enumerate(values):
        if pd.isna(raw_value) or float(raw_value) == 0.0:
            if run_start is not None:
                starts.append(run_start)
                ends.append(start_0based + offset)
                out_values.append(float(run_value))
                run_start = None
                run_value = None
            continue

        value = float(raw_value)
        absolute_position = start_0based + offset
        if run_start is None:
            run_start = absolute_position
            run_value = value
        elif value != run_value:
            starts.append(run_start)
            ends.append(absolute_position)
            out_values.append(float(run_value))
            run_start = absolute_position
            run_value = value

    if run_start is not None:
        starts.append(run_start)
        ends.append(start_0based + len(values))
        out_values.append(float(run_value))
    return starts, ends, out_values


def _write_region_bigwig(
    source_url: str,
    path: Path,
    region: dict[str, object],
    *,
    force: bool = False,
) -> str:
    """Write a local bigWig containing only the ANKRD30A genomic interval."""
    if path.exists() and not force:
        return "cached"

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.bigWig")
    tmp.unlink(missing_ok=True)

    chrom = str(region["chrom"])
    start = int(region["start_0based"])
    end = int(region["end_0based"])
    # Read only the ANKRD30A interval from the remote GEO bigWig; this avoids
    # keeping full-genome signal tracks in the project data directory.
    with pyBigWig.open(source_url) as source:
        source_chrom = _bigwig_chrom_name(source, chrom)
        chrom_size = int(source.chroms()[source_chrom])
        values = source.values(source_chrom, start, end, numpy=True)
    # Store the partial bigWig sparsely by collapsing adjacent identical
    # nonzero values into runs.
    starts, ends, out_values = _nonzero_runs(values, start)

    with pyBigWig.open(str(tmp), "w") as out:
        out.addHeader([(source_chrom, chrom_size)])
        if starts:
            out.addEntries(
                [source_chrom] * len(starts),
                starts,
                ends=ends,
                values=out_values,
            )
    tmp.replace(path)
    return "downloaded_region"


def download_region_bigwigs(
    manifest: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Download the ANKRD30A region from every configured remote GEO bigWig."""
    region = _ankrd30a_download_region(windows)
    rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        # Build an explicit download-status table with one row per strand track
        # so dry runs and workflow summaries can show exactly what is cached.
        for strand, url_col, path_col in (
            ("pos", "pos_bigwig_url", "pos_bigwig_path"),
            ("neg", "neg_bigwig_url", "neg_bigwig_path"),
        ):
            path = Path(str(row[path_col]))
            rows.append(
                {
                    "kind": "bigwig",
                    "sample_id": row["sample_id"],
                    "timepoint": row["timepoint"],
                    "replicate": row["replicate"],
                    "strand": strand,
                    "url": row[url_col],
                    "path": str(path),
                    "region": (
                        f"{region['chrom']}:{int(region['start_0based']) + 1:,}-"
                        f"{int(region['end_0based']):,}"
                    ),
                    "exists": path.exists(),
                    "status": "dry_run" if dry_run else None,
                }
            )
    if dry_run:
        return pd.DataFrame(rows)

    for item in rows:
        path = Path(str(item["path"]))
        item["status"] = _write_region_bigwig(
            str(item["url"]),
            path,
            region,
            force=force,
        )
        item["exists"] = path.exists()
    return pd.DataFrame(rows)


def _local_bigwigs_complete(manifest: pd.DataFrame) -> bool:
    """Return whether every local partial bigWig exists."""
    paths = [
        Path(str(row[col]))
        for _, row in manifest.iterrows()
        for col in ("pos_bigwig_path", "neg_bigwig_path")
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


def _signal_cache_is_current(signal: pd.DataFrame) -> bool:
    """Return whether the cached table uses summed, not length-normalized, signal."""
    required = {
        "combined_abs_sum",
        "baseline_combined_abs_sum",
        "delta_combined_abs_sum_vs_0h",
        "log2fc_combined_abs_sum_vs_0h",
    }
    legacy_normalized_columns = {
        "combined_mean_abs_signal",
        "baseline_combined_mean_abs_signal",
        "delta_combined_mean_abs_signal_vs_0h",
        "log2fc_combined_mean_abs_signal_vs_0h",
    }
    return required.issubset(signal.columns) and not (
        legacy_normalized_columns & set(signal.columns)
    )


def _combined_signal(
    pos_bw: Any, neg_bw: Any, chrom: str, start: int, end: int
) -> dict[str, float]:
    """Sum absolute plus plus absolute minus bigWig signal in one interval."""
    bw_chrom = _bigwig_chrom_name(pos_bw, chrom)
    plus = np.asarray(pos_bw.values(bw_chrom, start, end, numpy=True), dtype=float)
    minus = np.asarray(neg_bw.values(bw_chrom, start, end, numpy=True), dtype=float)
    plus = np.nan_to_num(plus, nan=0.0)
    minus = np.nan_to_num(minus, nan=0.0)
    # The processed strand tracks can be signed; absolute plus + absolute minus
    # captures local PRO-seq activity without assigning TSS directionality.
    combined_sum = float(np.abs(plus).sum() + np.abs(minus).sum())
    return {"combined_abs_sum": combined_sum}


def quantify_signal_changes(
    manifest: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify combined-strand PRO-seq signal changes versus the 0h track."""
    rows: list[dict[str, object]] = []
    for _, sample in manifest.iterrows():
        pos_path = Path(str(sample["pos_bigwig_path"]))
        neg_path = Path(str(sample["neg_bigwig_path"]))
        if not pos_path.exists() or not neg_path.exists():
            raise FileNotFoundError(
                "Missing processed bigWig files for "
                f"{sample['sample_label']}: {pos_path}, {neg_path}"
            )
        with pyBigWig.open(str(pos_path)) as pos_bw, pyBigWig.open(str(neg_path)) as neg_bw:
            for _, window in windows.iterrows():
                signal = _combined_signal(
                    pos_bw,
                    neg_bw,
                    str(window["chrom"]),
                    int(window["start_0based"]),
                    int(window["end_0based"]),
                )
                rows.append({**sample.to_dict(), **window.to_dict(), **signal})

    out = pd.DataFrame(rows)
    # Every timepoint is compared with the 0h combined track for the same
    # feature, giving descriptive deltas/log2 fold changes.
    baseline = out[out["timepoint"].eq("0h")][["feature_id", "combined_abs_sum"]].rename(
        columns={"combined_abs_sum": "baseline_combined_abs_sum"}
    )
    out = out.merge(baseline, on="feature_id", how="left")
    out["delta_combined_abs_sum_vs_0h"] = out["combined_abs_sum"] - out["baseline_combined_abs_sum"]
    out["log2fc_combined_abs_sum_vs_0h"] = np.log2(
        (out["combined_abs_sum"] + TRPS1_PROSEQ_SIGNAL_EPSILON)
        / (out["baseline_combined_abs_sum"] + TRPS1_PROSEQ_SIGNAL_EPSILON)
    )
    return out.sort_values(["plot_order", "timepoint_order"])


def run_trps1_proseq(
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Download partial bigWigs and cache ANKRD30A PRO-seq signal summaries."""
    _ensure_dirs()
    # The manifest defines source/local bigWig paths; the windows define the
    # exact TSS and gene-body intervals quantified downstream.
    manifest = build_sample_manifest()
    windows = build_ankrd30a_signal_windows()

    if dry_run:
        return {
            "status": "dry_run",
            "download_status": download_region_bigwigs(
                manifest,
                windows,
                force=force,
                dry_run=True,
            ).to_dict("records"),
            "outputs": [
                str(ANKRD30A_TRPS1_PROSEQ_SAMPLE_MANIFEST_CSV),
                str(ANKRD30A_TRPS1_PROSEQ_SIGNAL_WINDOWS_CSV),
                str(ANKRD30A_TRPS1_PROSEQ_SIGNAL_CSV),
            ],
        }

    save_csv(manifest, ANKRD30A_TRPS1_PROSEQ_SAMPLE_MANIFEST_CSV)
    save_csv(windows, ANKRD30A_TRPS1_PROSEQ_SIGNAL_WINDOWS_CSV)
    # Downloading is idempotent unless --force is used; cached partial bigWigs
    # are reused for local signal extraction.
    download_status = download_region_bigwigs(
        manifest,
        windows,
        force=force,
        dry_run=False,
    )

    if (
        ANKRD30A_TRPS1_PROSEQ_SIGNAL_CSV.exists()
        and not force
        and _local_bigwigs_complete(manifest)
    ):
        signal = pd.read_csv(ANKRD30A_TRPS1_PROSEQ_SIGNAL_CSV)
        if _signal_cache_is_current(signal):
            status = "cached"
        else:
            signal = quantify_signal_changes(manifest, windows)
            save_csv(signal, ANKRD30A_TRPS1_PROSEQ_SIGNAL_CSV)
            status = "computed"
    else:
        # Recompute signal when requested, when the signal CSV is missing, or
        # when any local partial bigWig is incomplete.
        signal = quantify_signal_changes(manifest, windows)
        save_csv(signal, ANKRD30A_TRPS1_PROSEQ_SIGNAL_CSV)
        status = "computed"

    return {
        "status": status,
        "n_signal_rows": int(signal.shape[0]),
        "n_signal_windows": int(windows.shape[0]),
        "n_bigwigs": int(download_status.shape[0]),
        "n_bigwigs_cached": int(download_status["status"].astype(str).eq("cached").sum()),
        "n_bigwigs_downloaded": int(
            download_status["status"].astype(str).eq("downloaded_region").sum()
        ),
        "outputs": [
            str(ANKRD30A_TRPS1_PROSEQ_SAMPLE_MANIFEST_CSV),
            str(ANKRD30A_TRPS1_PROSEQ_SIGNAL_WINDOWS_CSV),
            str(ANKRD30A_TRPS1_PROSEQ_SIGNAL_CSV),
        ],
        "data_dir": str(ANKRD30A_TRPS1_PROSEQ_DATA_DIR),
    }


def parse_args() -> argparse.Namespace:
    """Parse the standalone TRPS1 PRO-seq cache workflow CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Download GSE236172 ANKRD30A-region bigWigs and quantify PB TSS PRO-seq signal changes."
        )
    )
    parser.add_argument("--force", action="store_true", help="Recompute all caches.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads and outputs without writing caches.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the standalone PRO-seq cache workflow."""
    args = parse_args()
    status = run_trps1_proseq(force=args.force, dry_run=args.dry_run)
    for key, value in status.items():
        log(f"{key}: {value}")


if __name__ == "__main__":
    main()
