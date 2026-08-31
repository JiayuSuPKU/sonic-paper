"""Build ANKRD30A long-read annotations and notebook-facing transcript caches.

The ANKRD30A figure needs a small, inspectable view of reference GENCODE
transcripts, PacBio QC-pass transcripts, Atera probe genomic segments, and
probe-compatibility classes. The main all-SVP probe-mapping workflow owns the
genome/probe overlap caches; this module extracts the ANKRD30A rows, adds
PacBio SQANTI coding metadata plus ER+ full-length count support, and writes
the three plain CSVs consumed by ``visualization.ipynb``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    ANKRD30A_PROBE_MAPPINGS_CSV,
    ANKRD30A_REGULATOR_PRIORITY_CSV,
    ANKRD30A_TRANSCRIPT_ANNOTATIONS_CSV,
    ANKRD30A_VISUALIZATION_CACHE_DIR,
    GENCODE_GTF,
    PACBIO_FL_COUNTS_TSV,
    PACBIO_GFF,
    SQANTI_TXT,
    SVP_PROBE_RESULT_DIR,
)
from ..probe_mapping import open_text, parse_attrs, parse_segments, read_gencode_subset
from ..utils import project_path, save_csv
from .common import (
    ANKRD30A_VISUALIZATION_CACHE_FILES,
    PB_TSS_HYPOTHESIS_TRANSCRIPTS,
    PROBE_CLASS_ORDER,
    TARGET_GENE,
    _codewords_from_cache,
    _probe_compatibility_class,
    _read_required_tsv,
    _require_columns,
)


@lru_cache(maxsize=1)
def all_pacbio_ankrd30a_transcript_ids() -> tuple[str, ...]:
    """Return every PB-QC-passed ANKRD30A transcript in the PacBio GFF."""
    transcript_ids: list[str] = []
    with open_text(PACBIO_GFF) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "transcript":
                continue
            attrs = parse_attrs(fields[8])
            # StringTie quantification uses the full PB-QC ANKRD30A transcript
            # model, even though the final figure highlights four event isoforms.
            if attrs.get("gene_name") != TARGET_GENE:
                continue
            transcript_id = str(attrs.get("transcript_id", ""))
            if transcript_id:
                transcript_ids.append(transcript_id)
    if not transcript_ids:
        raise RuntimeError(
            f"No PB-QC-passed {TARGET_GENE} transcripts found in {project_path(PACBIO_GFF)}"
        )
    return tuple(dict.fromkeys(transcript_ids))


def _read_pacbio_gene_annotations(path: Path = PACBIO_GFF, gene: str = TARGET_GENE) -> pd.DataFrame:
    """Read PacBio QC-pass GFF rows for one gene with normalized coordinates."""
    rows: list[dict[str, object]] = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            attrs = parse_attrs(fields[8])
            if attrs.get("gene_name") != gene:
                continue
            # Preserve both 1-based annotation coordinates and 0-based intervals
            # used by overlap/counting code.
            chrom, source, feature, start, end, score, strand, frame, _ = fields
            rows.append(
                {
                    "source_name": "PacBio QC pass",
                    "chrom": chrom,
                    "source": source,
                    "feature": feature,
                    "start_1based": int(start),
                    "end_1based": int(end),
                    "start_0based": int(start) - 1,
                    "end_0based": int(end),
                    "score": score,
                    "strand": strand,
                    "frame": frame,
                    **attrs,
                }
            )
    if not rows:
        raise RuntimeError(f"No {gene} rows found in PacBio GFF: {project_path(path)}")
    return pd.DataFrame(rows)


def _read_sqanti_annotations(path: Path, transcript_ids: Iterable[str]) -> pd.DataFrame:
    """Read SQANTI rows for the PacBio transcript IDs used by ANKRD30A."""
    transcript_ids = set(map(str, transcript_ids))
    if not transcript_ids:
        raise ValueError("SQANTI annotation loading requires PacBio transcript IDs")
    rows: list[dict[str, object]] = []
    with open_text(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        idx_iso = header.index("isoform")
        # The SQANTI table is genome-wide; stream only the ANKRD30A PB IDs to
        # keep this cache step cheap and deterministic.
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(header) or fields[idx_iso] not in transcript_ids:
                continue
            rows.append(dict(zip(header, fields, strict=False)))
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(
            f"SQANTI annotation has no rows for {len(transcript_ids):,} {TARGET_GENE} PacBio transcripts"
        )
    missing = sorted(transcript_ids - set(out["isoform"].astype(str)))
    if missing:
        raise ValueError("SQANTI annotation is missing PacBio transcripts: " + ", ".join(missing))
    for col in [
        "FL_counts",
        "FL_samples",
        "ORF_length",
        "CDS_length",
        "length",
        "exons",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _read_pacbio_fl_counts(path: Path = PACBIO_FL_COUNTS_TSV) -> pd.DataFrame:
    """Read the required PacBio per-sample full-length count table."""
    return _read_required_tsv(path, "PacBio per-sample FL count table")


def _summarize_pacbio_fl_counts(
    counts: pd.DataFrame, transcript_ids: Iterable[str]
) -> pd.DataFrame:
    """Return per-transcript FL totals split by ER+ and ER-/TNBC samples."""
    if counts.empty:
        raise ValueError("PacBio per-sample FL count table is empty")
    id_col = next(
        (
            col
            for col in [
                "superPBID",
                "isoform",
                "pbid",
                "transcript_id",
                "sqanti2_isoform",
            ]
            if col in counts.columns
        ),
        str(counts.columns[0]) if len(counts.columns) else None,
    )
    tx_ids = set(map(str, transcript_ids))
    subset = counts[counts[id_col].astype(str).isin(tx_ids)].copy()
    if subset.empty:
        raise ValueError("PacBio per-sample FL count table has no ANKRD30A transcript rows")
    numeric_cols = [
        col
        for col in subset.columns
        if col != id_col and pd.to_numeric(subset[col], errors="coerce").notna().any()
    ]
    if not numeric_cols:
        raise ValueError("PacBio per-sample FL count table has no numeric sample columns")
    numeric = subset[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    # The sample names encode receptor status; ER+ counts drive PB transcript
    # prioritization, while ER-/other counts remain supporting metadata.
    er_positive = [col for col in numeric_cols if re.search(r"ERpos", col, flags=re.IGNORECASE)]
    er_negative = [
        col for col in numeric_cols if re.search(r"ERneg|TNBC", col, flags=re.IGNORECASE)
    ]
    other = [col for col in numeric_cols if col not in set(er_positive) | set(er_negative)]
    if not er_positive:
        raise ValueError("PacBio per-sample FL count table has no ER-positive sample columns")

    def totals(sample_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if not sample_cols:
            zeros = np.zeros(numeric.shape[0], dtype=float)
            return zeros, zeros
        values = numeric[sample_cols]
        return values.sum(axis=1).to_numpy(dtype=float), values.gt(0).sum(axis=1).to_numpy(
            dtype=float
        )

    all_counts, all_samples = totals(numeric_cols)
    er_pos_counts, er_pos_samples = totals(er_positive)
    er_neg_counts, er_neg_samples = totals(er_negative)
    other_counts, other_samples = totals(other)
    return pd.DataFrame(
        {
            "transcript_id": subset[id_col].astype(str).to_numpy(),
            "FL_counts_from_sample_table": all_counts,
            "FL_samples_from_sample_table": all_samples,
            "er_positive_fl_counts": er_pos_counts,
            "er_positive_fl_samples": er_pos_samples,
            "er_negative_fl_counts": er_neg_counts,
            "er_negative_fl_samples": er_neg_samples,
            "other_fl_counts": other_counts,
            "other_fl_samples": other_samples,
            "n_er_positive_long_read_samples": len(er_positive),
            "n_er_negative_long_read_samples": len(er_negative),
            "n_other_long_read_samples": len(other),
            "er_positive_long_read_sample_columns": "|".join(er_positive),
        }
    )


def _tss_1based(row: pd.Series) -> int:
    """Return the 1-based TSS coordinate for a transcript row."""
    if str(row["strand"]) == "-":
        return int(row["end_1based"])
    return int(row["start_1based"])


def _coerce_numeric_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric in place when present."""
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _attach_sqanti_metadata(transcripts: pd.DataFrame, sqanti: pd.DataFrame) -> pd.DataFrame:
    """Add SQANTI coding/ORF metadata to PacBio transcript rows."""
    cols = [
        "isoform",
        "structural_category",
        "associated_transcript",
        "subcategory",
        "coding",
        "ORF_length",
        "CDS_length",
        "FL_counts",
        "FL_samples",
        "diff_to_TSS",
        "diff_to_TTS",
    ]
    sqanti_small = sqanti[[col for col in cols if col in sqanti.columns]].copy()
    return transcripts.merge(
        sqanti_small,
        left_on="transcript_id",
        right_on="isoform",
        how="left",
        suffixes=("", "_sqanti"),
    )


def _attach_fl_count_metadata(transcripts: pd.DataFrame, fl_summary: pd.DataFrame) -> pd.DataFrame:
    """Prefer ER-stratified sample-level FL counts over aggregate SQANTI counts."""
    transcripts = transcripts.merge(fl_summary, on="transcript_id", how="left")
    missing_fl = (
        transcripts[transcripts["FL_counts_from_sample_table"].isna()]["transcript_id"]
        .astype(str)
        .tolist()
    )
    if missing_fl:
        raise ValueError(
            "PacBio per-sample FL count summary is missing transcripts: " + ", ".join(missing_fl)
        )
    for target_col, source_col in [
        ("FL_counts", "FL_counts_from_sample_table"),
        ("FL_samples", "FL_samples_from_sample_table"),
    ]:
        sample_values = pd.to_numeric(transcripts[source_col], errors="coerce")
        existing_values = (
            pd.to_numeric(transcripts[target_col], errors="coerce")
            if target_col in transcripts
            else pd.Series(np.nan, index=transcripts.index)
        )
        transcripts[target_col] = sample_values.combine_first(existing_values)
    transcripts["FL_count_source"] = np.where(
        transcripts["FL_counts_from_sample_table"].notna(),
        "BRCA_superPBID_counts.tsv",
        "SQANTI",
    )
    return transcripts


def _add_pb_tss_metadata(transcripts: pd.DataFrame) -> pd.DataFrame:
    """Add transcript TSS coordinates and validate the focused PB event rows."""
    transcripts["tss_1based"] = transcripts.apply(_tss_1based, axis=1)
    focus_tss = transcripts[
        transcripts["transcript_id"].astype(str).isin(PB_TSS_HYPOTHESIS_TRANSCRIPTS)
    ]["tss_1based"].astype(int)
    if focus_tss.empty:
        raise ValueError(
            "PacBio transcript table lacks focused ANKRD30A TSS transcripts: "
            + ", ".join(PB_TSS_HYPOTHESIS_TRANSCRIPTS)
        )
    return _coerce_numeric_columns(
        transcripts,
        [
            "FL_counts",
            "FL_samples",
            "er_positive_fl_counts",
            "er_positive_fl_samples",
            "er_negative_fl_counts",
            "er_negative_fl_samples",
            "other_fl_counts",
            "other_fl_samples",
            "ORF_length",
            "CDS_length",
        ],
    )


@lru_cache(maxsize=1)
def _compute_ankrd30a_long_read_annotations() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build ANKRD30A PacBio structures and FL-count summaries from source files."""
    # Start from PB-QC structural rows, then enrich transcript rows with SQANTI
    # ORF/coding calls and ER-stratified sample-level FL support.
    features = _read_pacbio_gene_annotations()
    transcripts = features[features["feature"].astype(str).eq("transcript")].copy()
    transcript_ids = transcripts["transcript_id"].dropna().astype(str).unique().tolist()
    sqanti = _read_sqanti_annotations(SQANTI_TXT, transcript_ids)
    fl_summary = _summarize_pacbio_fl_counts(_read_pacbio_fl_counts(), transcript_ids)
    transcripts = _attach_sqanti_metadata(transcripts, sqanti)
    transcripts = _attach_fl_count_metadata(transcripts, fl_summary)
    transcripts = _add_pb_tss_metadata(transcripts)

    return transcripts, features, sqanti


def _copy_frames(frames: tuple[pd.DataFrame, ...]) -> tuple[pd.DataFrame, ...]:
    """Return copies so callers cannot mutate cached annotation frames."""
    return tuple(frame.copy() for frame in frames)


def prepare_ankrd30a_long_read_annotations(
    *, force: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ANKRD30A PacBio structures and transcript metadata.

    The transcript table carries SQANTI coding fields and ER-stratified PacBio
    full-length count summaries. The same in-memory tables feed ReMap support,
    TCGA visualization focus transcripts, and the compact transcript/probe
    cache written for the notebook.
    """
    if force:
        _compute_ankrd30a_long_read_annotations.cache_clear()
    return _copy_frames(_compute_ankrd30a_long_read_annotations())  # type: ignore[return-value]


# Visualization cache assembly


def load_ankrd30a_visualization_cache() -> dict[str, pd.DataFrame]:
    """Load the plain-CSV ANKRD30A tables consumed by ``visualization.ipynb``."""
    missing = [
        path
        for path in ANKRD30A_VISUALIZATION_CACHE_FILES
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        missing_text = ", ".join(project_path(path) for path in missing)
        raise FileNotFoundError(
            f"Missing ANKRD30A visualization cache file(s): {missing_text}. "
            "Run scripts/atera_ffpe_human_brca/workflow.py before rendering figures."
        )
    cache = {
        "transcript_annotations": pd.read_csv(ANKRD30A_TRANSCRIPT_ANNOTATIONS_CSV),
        "probe_mappings": pd.read_csv(ANKRD30A_PROBE_MAPPINGS_CSV),
        "regulator_priority": pd.read_csv(ANKRD30A_REGULATOR_PRIORITY_CSV),
    }
    _require_columns(
        cache["transcript_annotations"],
        [
            "annotation_source",
            "feature",
            "transcript_id",
            "start_0based",
            "end_0based",
            "probe_compatibility_class",
        ],
        "ANKRD30A transcript annotation cache",
    )
    _require_columns(
        cache["probe_mappings"],
        [
            "mapping_row_id",
            "segment_index",
            "codeword",
            "chrom",
            "start_0based",
            "end_0based",
        ],
        "ANKRD30A probe mapping cache",
    )
    _require_columns(
        cache["regulator_priority"],
        ["regulator", "regulator_class", "pvalue", "pvalue_adj", "ankrd30a_du_rank"],
        "ANKRD30A regulator-priority cache",
    )
    return cache


def _visualization_cache_ready() -> bool:
    """Return true when the three plain-CSV notebook cache files are readable."""
    if any(
        not path.exists() or path.stat().st_size == 0 for path in ANKRD30A_VISUALIZATION_CACHE_FILES
    ):
        return False
    load_ankrd30a_visualization_cache()
    return True


def _ankrd30a_compatibility_from_svp_cache() -> pd.DataFrame:
    """Load ANKRD30A-only transcript/probe compatibility from all-SVP caches."""
    frames: list[pd.DataFrame] = []
    for filename, source in [
        ("svp_gencode_transcript_probe_compatibility.csv", "GENCODE v49"),
        ("svp_pacbio_transcript_probe_compatibility.csv", "PacBio QC pass"),
    ]:
        path = SVP_PROBE_RESULT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {project_path(path)}. Run workflow.py to build the all-SVP probe cache."
            )
        table = pd.read_csv(path)
        _require_columns(
            table,
            ["gene", "transcript_id", "compatibility_group", "compatible_codewords"],
            f"all-SVP {source} compatibility cache",
        )
        table = table[table["gene"].astype(str).eq(TARGET_GENE)].copy()
        if table.empty:
            raise ValueError(f"all-SVP {source} compatibility cache has no {TARGET_GENE} rows")
        table["annotation_source"] = source
        # The all-SVP cache stores raw compatible codewords; collapse them into
        # the small biological classes used by the ANKRD30A transcript panel.
        table["probe_compatibility_class"] = table["compatibility_group"].map(
            lambda value: _probe_compatibility_class(_codewords_from_cache(value))
        )
        table["probe_class_order"] = (
            table["probe_compatibility_class"].map(PROBE_CLASS_ORDER).fillna(99).astype(int)
        )
        frames.append(table)
    return pd.concat(frames, ignore_index=True, sort=False)


def _annotation_tss_table(features: pd.DataFrame) -> pd.DataFrame:
    """Return per-transcript TSS values from transcript feature rows."""
    _require_columns(
        features,
        [
            "feature",
            "annotation_source",
            "transcript_id",
            "strand",
            "start_1based",
            "end_1based",
        ],
        "transcript annotation features",
    )
    tx = features[features["feature"].astype(str).eq("transcript")].copy()
    if tx.empty:
        raise ValueError("Transcript annotation features contain no transcript rows")
    tx["tss_1based_computed"] = np.where(
        tx["strand"].astype(str).eq("-"),
        pd.to_numeric(tx["end_1based"], errors="coerce"),
        pd.to_numeric(tx["start_1based"], errors="coerce"),
    )
    return tx[["annotation_source", "transcript_id", "tss_1based_computed"]]


def _combine_ankrd30a_transcript_annotations(
    gencode_features: pd.DataFrame,
    pacbio_features: pd.DataFrame,
    pacbio_transcripts: pd.DataFrame,
    compatibility: pd.DataFrame,
) -> pd.DataFrame:
    """Build one plain feature table with transcript/probe annotations attached."""
    if gencode_features.empty:
        raise ValueError("GENCODE ANKRD30A feature table is empty")
    if pacbio_features.empty:
        raise ValueError("PacBio ANKRD30A feature table is empty")
    if compatibility.empty:
        raise ValueError("ANKRD30A transcript/probe compatibility table is empty")
    gencode = gencode_features.copy()
    pacbio = pacbio_features.copy()
    gencode["annotation_source"] = "GENCODE v49"
    pacbio["annotation_source"] = "PacBio QC pass"

    # Attach PacBio transcript-level SQANTI and ER+ FL-count metadata to every
    # PacBio feature row so one CSV is enough for transcript plotting.
    pacbio_meta_cols = [
        "transcript_id",
        "isoform",
        "structural_category_sqanti",
        "associated_transcript_sqanti",
        "subcategory",
        "coding",
        "ORF_length",
        "CDS_length",
        "FL_counts",
        "FL_samples",
        "diff_to_TSS",
        "diff_to_TTS",
        "FL_counts_from_sample_table",
        "FL_samples_from_sample_table",
        "er_positive_fl_counts",
        "er_positive_fl_samples",
        "er_negative_fl_counts",
        "er_negative_fl_samples",
        "other_fl_counts",
        "other_fl_samples",
        "n_er_positive_long_read_samples",
        "n_er_negative_long_read_samples",
        "n_other_long_read_samples",
        "er_positive_long_read_sample_columns",
        "FL_count_source",
        "tss_1based",
    ]
    pacbio_meta_cols = [
        col
        for col in pacbio_meta_cols
        if col in pacbio_transcripts.columns
        and (col == "transcript_id" or col not in pacbio.columns)
    ]
    if "transcript_id" in pacbio_meta_cols:
        pacbio = pacbio.merge(
            pacbio_transcripts[pacbio_meta_cols].drop_duplicates("transcript_id"),
            on="transcript_id",
            how="left",
        )

    annotations = pd.concat([gencode, pacbio], ignore_index=True, sort=False)
    # TSS is repeated on every feature row so plotting can draw transcripts from
    # one flat CSV without re-parsing GTF/GFF files.
    tss = _annotation_tss_table(annotations)
    annotations = annotations.merge(tss, on=["annotation_source", "transcript_id"], how="left")
    if "tss_1based" not in annotations:
        annotations["tss_1based"] = np.nan
    annotations["tss_1based"] = pd.to_numeric(
        annotations["tss_1based"], errors="coerce"
    ).combine_first(pd.to_numeric(annotations["tss_1based_computed"], errors="coerce"))
    annotations = annotations.drop(columns="tss_1based_computed")

    compat_cols = [
        "annotation_source",
        "transcript_id",
        "compatible_codewords",
        "compatible_probe_oligos",
        "compatibility_group",
        "n_compatible_codewords",
        "n_transcripts_in_compatibility_group",
        "probe_compatibility_class",
        "probe_class_order",
        "n_exons",
        "exon_chain",
        "sqanti_structural_category",
        "sqanti_associated_transcript",
        "sqanti_subcategory",
        "sqanti_coding",
        "sqanti_ORF_length",
        "sqanti_CDS_length",
        "sqanti_FL_counts",
        "sqanti_FL_samples",
    ]
    required_compat_cols = [
        "annotation_source",
        "transcript_id",
        "compatible_codewords",
        "compatible_probe_oligos",
        "probe_compatibility_class",
        "probe_class_order",
    ]
    _require_columns(
        compatibility,
        required_compat_cols,
        "ANKRD30A transcript/probe compatibility table",
    )
    compat_cols = [col for col in compat_cols if col in compatibility.columns]
    compat = compatibility[compat_cols].drop_duplicates(["annotation_source", "transcript_id"])
    annotations = annotations.merge(compat, on=["annotation_source", "transcript_id"], how="left")
    probe_class = (
        annotations["probe_compatibility_class"]
        if "probe_compatibility_class" in annotations
        else pd.Series(index=annotations.index, dtype=object)
    )
    annotations["probe_compatibility_class"] = probe_class.fillna("no-overlap")
    probe_order = (
        annotations["probe_class_order"]
        if "probe_class_order" in annotations
        else pd.Series(99, index=annotations.index)
    )
    annotations["probe_class_order"] = pd.to_numeric(probe_order, errors="coerce").fillna(99)
    return annotations.sort_values(
        ["annotation_source", "transcript_id", "start_1based", "end_1based", "feature"],
        na_position="last",
    ).reset_index(drop=True)


def _ankrd30a_probe_segments_from_svp_cache() -> pd.DataFrame:
    """Return ANKRD30A probe mappings expanded to one row per mapped segment."""
    mapping_path = SVP_PROBE_RESULT_DIR / "svp_probe_mappings.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Missing {project_path(mapping_path)}. Run all-SVP probe mapping before ANKRD30A panel caching."
        )
    mappings = pd.read_csv(mapping_path)
    _require_columns(
        mappings,
        ["target_name", "mapping_status", "codeword", "segments_0based"],
        "all-SVP probe mapping cache",
    )
    mappings = mappings[mappings["target_name"].astype(str).eq(TARGET_GENE)].copy()
    if mappings.empty:
        raise ValueError(f"all-SVP probe mapping cache has no {TARGET_GENE} rows")
    mappings = mappings[mappings["mapping_status"].astype(str).str.startswith("mapped")].copy()
    if mappings.empty:
        raise ValueError(f"all-SVP probe mapping cache has no mapped {TARGET_GENE} rows")
    rows: list[dict[str, object]] = []
    for row_idx, row in mappings.reset_index(drop=True).iterrows():
        base = row.to_dict()
        mapping_row_id = f"{TARGET_GENE}_probe_mapping_{row_idx + 1}"
        segments = parse_segments(row)
        # Split probes are preserved as multiple rows with a shared mapping ID;
        # the notebook connects those segments when drawing probe structures.
        for segment_index, (chrom, start0, end0) in enumerate(segments, start=1):
            record = dict(base)
            record["mapping_row_id"] = mapping_row_id
            record["segment_index"] = segment_index
            record["segment_count"] = len(segments)
            record["chrom"] = chrom
            record["start_0based"] = int(start0)
            record["end_0based"] = int(end0)
            record["start_1based"] = int(start0) + 1
            record["end_1based"] = int(end0)
            record["segment_region_0based"] = f"{chrom}:{int(start0)}-{int(end0)}"
            record["segment_region_1based"] = f"{chrom}:{int(start0) + 1}-{int(end0)}"
            rows.append(record)
    return pd.DataFrame(rows)


def _write_ankrd30a_visualization_cache(
    pacbio_transcripts: pd.DataFrame,
    pacbio_features: pd.DataFrame,
    regulator_priority: pd.DataFrame,
    *,
    force: bool = False,
) -> dict[str, object]:
    """Write the three plain ANKRD30A CSV tables consumed by the figure notebook.

    The transcript annotation CSV is intentionally denormalized because these
    are small figure-specific tables. Probe mappings and DU regulator priority
    remain separate files for easier inspection.
    """
    if _visualization_cache_ready() and not force:
        return {
            "status": "cached",
            "outputs": [str(path) for path in ANKRD30A_VISUALIZATION_CACHE_FILES],
        }

    # GENCODE parsing is the expensive step this cache avoids in the notebook.
    gencode_features = read_gencode_subset(GENCODE_GTF, [TARGET_GENE])
    compat = _ankrd30a_compatibility_from_svp_cache()
    annotations = _combine_ankrd30a_transcript_annotations(
        gencode_features,
        pacbio_features,
        pacbio_transcripts,
        compat,
    )
    probe_segments = _ankrd30a_probe_segments_from_svp_cache()
    ANKRD30A_VISUALIZATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(annotations, ANKRD30A_TRANSCRIPT_ANNOTATIONS_CSV)
    save_csv(probe_segments, ANKRD30A_PROBE_MAPPINGS_CSV)
    save_csv(regulator_priority, ANKRD30A_REGULATOR_PRIORITY_CSV)
    return {
        "status": "computed",
        "outputs": [str(path) for path in ANKRD30A_VISUALIZATION_CACHE_FILES],
        "n_annotation_rows": int(annotations.shape[0]),
        "n_probe_mapping_rows": int(probe_segments.shape[0]),
        "n_compatibility_rows": int(compat.shape[0]),
        "n_regulator_priority_rows": int(regulator_priority.shape[0]),
    }
