"""TCGA ANKRD30A transcript/probe quantification and ER+ regulator validation.

The TCGA validation stage expects the all-SVP probe mapping cache (from workflow.py),
local TCGA-BRCA BAM slices around the ANKRD30A locus (from ./download_tcga_brca_ankrd30a.py),
and the associated TCGA per-sample clinical/expression data
(from ./download_tcga_brca_clinical_expression.py) to be present.

It (1) quantifies ANKRD30A transcripts with StringTie/prepDE, (2) counts
Atera probe genomic regions with featureCounts, (3) computes associations
with TF expression and overall survival, and (4) merges those results with
ER status, and selected TF expression rows into the tables used by
``visualization.ipynb`` for ER+ correlation and survival-oriented panels.
"""

from __future__ import annotations

import csv
import re
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import (
    PACBIO_GFF,
    SVP_PROBE_RESULT_DIR,
    TCGA_BAM_MANIFEST_TSV,
    TCGA_CBIO_CLINICAL_SAMPLE_TXT,
    TCGA_CBIO_RSEM_TXT,
    TCGA_METRIC_TABLE_TSV,
)
from ..probe_mapping import open_text, parse_attrs, parse_segments
from ..utils import bh_fdr, log, normalize_gene_symbol, project_path
from .common import (
    CORRELATION_MIN_SAMPLES,
    CORRELATION_TARGETS,
    PB_TRANSCRIPTS_OF_INTEREST,
    PROBE_REGION_CODEWORDS,
    PROBE_REGION_COUNTS_NAME,
    PROBE_REGION_FEATURES,
    PROBE_REGION_USAGE_NAME,
    QUANT_SAMPLE_COLUMNS,
    SAMPLE_COLUMNS,
    TARGET_GENE,
    TCGA_QUANTIFICATION_OUTPUTS,
    TRANSCRIPT_COUNT_NAME,
    TRANSCRIPT_USAGE_NAME,
    _probe_region_feature_name,
    _quant_path,
    _read_required_tsv,
    _require_columns,
    default_slice_region_1based,
)
from .long_read import all_pacbio_ankrd30a_transcript_ids

PROBE_REGION_AGGREGATION_COLUMN = "probe_region_aggregation"
PROBE_REGION_AGGREGATION_VALUE = "sum_regions_by_codeword"
STRINGTIE_CMD = "stringtie"
STRINGTIE_PREPDE_CMD = "prepDE.py"
STRINGTIE_PREPDE_READ_LENGTH = 75
FEATURECOUNTS_CMD = "featureCounts"
FEATURECOUNTS_THREADS = 4
FEATURECOUNTS_BATCH_SIZE = 100


def _sample_key(row: dict[str, object]) -> str:
    """Build the stable sample key used by TCGA quantification matrices."""
    missing = [col for col in ["case_submitter_id", "file_id"] if not str(row.get(col, "")).strip()]
    if not str(row.get("sample_submitter_id", "") or row.get("aliquot_submitter_id", "")).strip():
        missing.append("sample_submitter_id_or_aliquot_submitter_id")
    if missing:
        raise ValueError(f"TCGA manifest row is missing sample-key columns: {missing}")
    raw = "__".join(
        [
            str(row["case_submitter_id"]),
            str(row["sample_submitter_id"] or row["aliquot_submitter_id"]),
            str(row["file_id"]),
        ]
    )
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or "NA"


def tcga_quantification_ready() -> bool:
    """Return whether all TCGA validation matrices are present and complete."""

    def has_required_features(filename: str, required_features: Iterable[str]) -> bool:
        path = _quant_path(filename)
        if not path.exists() or path.stat().st_size == 0:
            return False
        table = pd.read_csv(path, sep="\t", nrows=1)
        return set(map(str, required_features)).issubset(set(table.columns.astype(str)))

    # The figure focuses on four PB event transcripts, but StringTie must use
    # every PB-QC ANKRD30A transcript as its guide reference. Requiring all
    # source transcripts here prevents stale four-transcript caches from being
    # silently reused.
    all_pb_transcripts = all_pacbio_ankrd30a_transcript_ids()
    return (
        has_required_features(TRANSCRIPT_COUNT_NAME, all_pb_transcripts)
        and has_required_features(TRANSCRIPT_USAGE_NAME, all_pb_transcripts)
        and has_required_features(
            PROBE_REGION_COUNTS_NAME,
            [*PROBE_REGION_FEATURES, PROBE_REGION_AGGREGATION_COLUMN],
        )
        and has_required_features(
            PROBE_REGION_USAGE_NAME,
            [*PROBE_REGION_FEATURES, PROBE_REGION_AGGREGATION_COLUMN],
        )
    )


def _resolve_tcga_bam_rows() -> list[dict[str, object]]:
    """Read the TCGA manifest and keep rows with a usable local ANKRD30A BAM slice."""
    if not TCGA_BAM_MANIFEST_TSV.exists():
        raise FileNotFoundError(f"Missing TCGA BAM manifest: {project_path(TCGA_BAM_MANIFEST_TSV)}")
    manifest = pd.read_csv(TCGA_BAM_MANIFEST_TSV, sep="\t", dtype=str).fillna("")
    _require_columns(
        manifest,
        [*QUANT_SAMPLE_COLUMNS, "slice_region", "slice_status"],
        "TCGA BAM manifest",
    )
    # Quantification runs only on BAM slices already downloaded and marked ok by
    # the Atera-local TCGA download script.
    manifest = manifest[manifest["slice_status"].astype(str).eq("ok")].copy()
    if manifest.empty:
        raise FileNotFoundError(
            "TCGA BAM manifest has no slice_status=ok rows. "
            "Run ankrd30a/download_tcga_brca_ankrd30a.py before quantification."
        )
    rows: list[dict[str, object]] = []
    missing_bams: list[str] = []
    for row in manifest.to_dict("records"):
        bam = Path(str(row.get("local_slice_bam", "")))
        if not bam.exists() or bam.stat().st_size == 0:
            missing_bams.append(str(bam))
            continue
        row["local_slice_bam"] = str(bam)
        rows.append(row)
    if missing_bams:
        preview = ", ".join(missing_bams[:5])
        raise FileNotFoundError(
            f"TCGA BAM manifest has {len(missing_bams):,} ok rows without local BAM files: {preview}"
        )
    return rows


def _ankrd30a_probe_region_table() -> pd.DataFrame:
    """Return direct genomic regions for ANKRD30A Atera codeword-level probe counting."""
    mapping_path = SVP_PROBE_RESULT_DIR / "svp_probe_mappings.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Missing {project_path(mapping_path)}. Run scripts/splisosm_atera_brca/workflow.py "
            "to build the all-SVP probe mapping cache before TCGA probe quantification."
        )
    mappings = pd.read_csv(mapping_path)
    _require_columns(
        mappings,
        ["target_name", "mapping_status", "codeword", "probe_oligo_label"],
        "all-SVP probe mapping cache",
    )
    mappings = mappings[mappings["target_name"].astype(str).eq(TARGET_GENE)].copy()

    if mappings.empty:
        raise ValueError(f"all-SVP probe mapping cache has no {TARGET_GENE} probe rows")

    rows: list[dict[str, object]] = []
    mapped = mappings[mappings["mapping_status"].astype(str).str.startswith("mapped")].copy()
    mapped = mapped[pd.to_numeric(mapped["codeword"], errors="coerce").isin(PROBE_REGION_CODEWORDS)]
    seen: set[tuple[object, ...]] = set()
    for _, probe in mapped.iterrows():
        codeword = int(probe["codeword"])
        label = str(probe.get("probe_oligo_label", codeword))
        for chrom, start0, end0 in parse_segments(probe):
            key = (codeword, label, chrom, int(start0), int(end0))
            if key in seen:
                continue
            # Count every split-probe segment directly; TCGA probe-region
            # quantification deliberately avoids probe-to-transcript assignment.
            seen.add(key)
            rows.append(
                {
                    "probe_feature": _probe_region_feature_name(codeword),
                    "codeword": codeword,
                    "probe_oligo_label": label,
                    "chrom": chrom,
                    "start_0based": int(start0),
                    "end_0based": int(end0),
                    "region": f"{chrom}:{int(start0) + 1}-{int(end0)}",
                }
            )
    if not rows:
        raise ValueError(f"No mapped {TARGET_GENE} probe regions found for TCGA counting")
    out = pd.DataFrame(rows).sort_values(["codeword", "probe_oligo_label", "chrom", "start_0based"])
    missing_codewords = sorted(
        set(PROBE_REGION_CODEWORDS) - set(pd.to_numeric(out["codeword"], errors="coerce"))
    )
    if missing_codewords:
        raise ValueError(f"Missing required ANKRD30A probe codewords: {missing_codewords}")
    return out


def _write_featurecounts_saf(path: Path, probe_regions: pd.DataFrame) -> None:
    """Write SAF annotation for featureCounts probe and gene-region counting."""
    gene_chrom, gene_start, gene_end = default_slice_region_1based()
    # The broad gene-region feature provides the per-sample denominator for
    # transcript/probe usage. It intentionally matches the TCGA BAM slice and
    # TRPS1 PRO-seq gene-body control interval.
    saf_rows = [
        {
            "GeneID": "gene_region_total",
            "Chr": gene_chrom,
            "Start": gene_start,
            "End": gene_end,
            "Strand": "+",
        }
    ]
    for row in probe_regions.to_dict("records"):
        saf_rows.append(
            {
                "GeneID": str(row["probe_feature"]),
                "Chr": str(row["chrom"]),
                "Start": int(row["start_0based"]) + 1,
                "End": int(row["end_0based"]),
                "Strand": "+",
            }
        )
    pd.DataFrame(saf_rows).to_csv(path, sep="\t", index=False)


def _run_featurecounts_probe_counts(
    rows: list[dict[str, object]],
    probe_regions: pd.DataFrame,
    work_dir: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Count direct probe-region and broad gene-region fragments with featureCounts."""
    featurecounts = FEATURECOUNTS_CMD

    saf = work_dir / "ankrd30a_probe_regions.saf"
    _write_featurecounts_saf(saf, probe_regions)
    probe_features = [_probe_region_feature_name(codeword) for codeword in PROBE_REGION_CODEWORDS]
    probe_counts_by_bam: dict[str, dict[str, float]] = {
        str(Path(str(row["local_slice_bam"]))): dict.fromkeys(probe_features, 0.0) for row in rows
    }
    gene_total_by_bam: dict[str, float] = {
        str(Path(str(row["local_slice_bam"]))): 0.0 for row in rows
    }

    for batch_start in range(0, len(rows), FEATURECOUNTS_BATCH_SIZE):
        batch = rows[batch_start : batch_start + FEATURECOUNTS_BATCH_SIZE]
        bams = [str(Path(str(row["local_slice_bam"]))) for row in batch]
        out = (
            work_dir
            / f"featurecounts_probe_batch_{batch_start // FEATURECOUNTS_BATCH_SIZE + 1}.txt"
        )
        cmd = [
            featurecounts,
            "-F",
            "SAF",
            "-a",
            str(saf),
            "-o",
            str(out),
            "-T",
            str(FEATURECOUNTS_THREADS),
            "-s",
            "0",
            "-O",
            "--primary",
            "-p",
            "--countReadPairs",
            *bams,
        ]
        # Batch featureCounts to avoid command-line length issues with hundreds
        # of TCGA BAM slices.
        log(
            f"featureCounts ANKRD30A probe regions {batch_start + 1:,}-{batch_start + len(batch):,}/{len(rows):,}"
        )
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"featureCounts failed with exit {completed.returncode}\n{detail}")

        table = pd.read_csv(out, sep="\t", comment="#")
        count_cols = [
            col
            for col in table.columns
            if col not in {"Geneid", "Chr", "Start", "End", "Strand", "Length"}
        ]
        for _, feature_row in table.iterrows():
            feature = str(feature_row["Geneid"])
            for col in count_cols:
                value = float(pd.to_numeric(feature_row[col], errors="coerce"))
                bam_key = str(Path(col))
                if feature == "gene_region_total":
                    gene_total_by_bam[bam_key] = value
                elif feature in probe_features:
                    # A codeword can have multiple mapped probe regions. Sum
                    # duplicate featureCounts rows into one codeword-level
                    # output column instead of letting the last region overwrite
                    # earlier regions.
                    probe_counts_by_bam[bam_key][feature] += value
    return probe_counts_by_bam, gene_total_by_bam


def _write_ankrd30a_pacbio_guide(path: Path) -> tuple[str, ...]:
    """Write all PB-QC-passed ANKRD30A PacBio rows as the StringTie guide."""
    expected = all_pacbio_ankrd30a_transcript_ids()
    with open_text(PACBIO_GFF) as handle, path.open("w") as out:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            attrs = parse_attrs(fields[8])
            if attrs.get("gene_name") != TARGET_GENE:
                continue
            # Keep the complete ANKRD30A PB-QC model in the guide so selected
            # four-transcript ratios are estimated in the context of other PB isoforms.
            out.write(line)
    return expected


def _run_stringtie_transcript_counts(
    rows: list[dict[str, object]],
    work_dir: Path,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Estimate PacBio transcript raw counts from sliced BAMs with StringTie/prepDE."""
    stringtie = STRINGTIE_CMD
    prepde = STRINGTIE_PREPDE_CMD

    guide = work_dir / "ankrd30a_pacbio_qc_guide.gff"
    required_transcripts = _write_ankrd30a_pacbio_guide(guide)
    sample_map: dict[str, str] = {}
    sample_list = work_dir / "stringtie_samples.tsv"
    with sample_list.open("w") as sample_handle:
        for idx, row in enumerate(rows, start=1):
            bam = Path(str(row["local_slice_bam"]))
            bam_key = str(bam)
            sample_key = _sample_key(row)
            sample_map[sample_key] = bam_key
            sample_dir = work_dir / "stringtie" / sample_key
            sample_dir.mkdir(parents=True, exist_ok=True)
            out_gtf = sample_dir / "transcripts.gtf"
            log(f"StringTie ANKRD30A transcripts {idx:,}/{len(rows):,}: {bam.name}")
            # -e constrains abundance estimation to the PB guide transcripts;
            # prepDE below converts those estimates to raw count matrices.
            cmd = [
                stringtie,
                bam_key,
                "-e",
                "-G",
                str(guide),
                "-o",
                str(out_gtf),
                "-p",
                "1",
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True)
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(
                    f"StringTie failed for {bam.name} with exit {completed.returncode}\n{detail}"
                )
            sample_handle.write(f"{sample_key}\t{out_gtf}\n")

    gene_counts = work_dir / "stringtie_gene_counts.csv"
    transcript_counts = work_dir / "stringtie_transcript_counts.csv"
    cmd = [
        prepde,
        "-i",
        str(sample_list),
        "-g",
        str(gene_counts),
        "-t",
        str(transcript_counts),
        "-l",
        str(STRINGTIE_PREPDE_READ_LENGTH),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"prepDE.py failed with exit {completed.returncode}\n{detail}")

    count_table = pd.read_csv(transcript_counts)
    if count_table.empty or "transcript_id" not in count_table:
        raise ValueError("prepDE.py transcript count table is empty or lacks transcript_id")
    transcript_features = count_table["transcript_id"].astype(str).tolist()
    # All guide transcripts must be present in the raw output so downstream
    # matrices cannot silently fall back to a four-transcript reference.
    missing = sorted(set(required_transcripts) - set(transcript_features))
    if missing:
        raise RuntimeError(
            "prepDE.py did not return all PB-QC ANKRD30A transcript counts: " + ", ".join(missing)
        )
    counts_by_bam = {
        bam_key: dict.fromkeys(transcript_features, 0.0) for bam_key in sample_map.values()
    }
    for sample_key, bam_key in sample_map.items():
        if sample_key not in count_table:
            raise ValueError(f"prepDE.py transcript count table is missing sample {sample_key}")
        values = (
            pd.to_numeric(count_table[sample_key], errors="coerce").fillna(0).to_numpy(dtype=float)
        )
        counts_by_bam[bam_key] = dict(zip(transcript_features, values, strict=False))
    return counts_by_bam, transcript_features


def _write_tcga_quantification_matrices(rows: list[dict[str, object]]) -> None:
    """Write StringTie transcript and featureCounts probe count/usage matrices."""
    probe_regions = _ankrd30a_probe_region_table()

    with tempfile.TemporaryDirectory(prefix="ankrd30a_tcga_quant_", dir="/private/tmp") as tmp:
        work_dir = Path(tmp)
        # The two quantification backends share the same BAM slices but produce
        # different features: direct genomic probe regions and PB transcripts.
        probe_counts_by_bam, gene_total_by_bam = _run_featurecounts_probe_counts(
            rows, probe_regions, work_dir
        )
        transcript_counts_by_bam, transcript_features = _run_stringtie_transcript_counts(
            rows, work_dir
        )

    probe_features = [_probe_region_feature_name(codeword) for codeword in PROBE_REGION_CODEWORDS]

    # raw transcript counts
    _write_sample_matrix(
        rows=rows,
        values_by_bam=transcript_counts_by_bam,
        features=transcript_features,
        path=_quant_path(TRANSCRIPT_COUNT_NAME),
        extra_totals_by_bam={"gene_region_total": gene_total_by_bam},
    )
    # transcript usage normalized by the broad gene-region total
    _write_sample_matrix(
        rows=rows,
        values_by_bam=transcript_counts_by_bam,
        features=transcript_features,
        path=_quant_path(TRANSCRIPT_USAGE_NAME),
        normalize=True,
        normalizer_by_bam=gene_total_by_bam,
        extra_totals_by_bam={"gene_region_total": gene_total_by_bam},
    )
    # raw probe-region counts
    _write_sample_matrix(
        rows=rows,
        values_by_bam=probe_counts_by_bam,
        features=probe_features,
        path=_quant_path(PROBE_REGION_COUNTS_NAME),
        extra_totals_by_bam={
            "gene_region_total": gene_total_by_bam,
            PROBE_REGION_AGGREGATION_COLUMN: PROBE_REGION_AGGREGATION_VALUE,
        },
    )
    # probe-region usage normalized by the broad gene-region total
    _write_sample_matrix(
        rows=rows,
        values_by_bam=probe_counts_by_bam,
        features=probe_features,
        path=_quant_path(PROBE_REGION_USAGE_NAME),
        normalize=True,
        normalizer_by_bam=gene_total_by_bam,
        extra_totals_by_bam={
            "gene_region_total": gene_total_by_bam,
            PROBE_REGION_AGGREGATION_COLUMN: PROBE_REGION_AGGREGATION_VALUE,
        },
    )


def _write_sample_matrix(
    *,
    rows: list[dict[str, object]],
    values_by_bam: dict[str, dict[str, float]],
    features: list[str],
    path: Path,
    normalize: bool = False,
    normalizer_by_bam: dict[str, float] | None = None,
    extra_totals_by_bam: dict[str, dict[str, float] | float] | None = None,
) -> None:
    """Write sample-by-feature matrices with optional within-row normalization."""
    output_rows: list[dict[str, object]] = []
    for row in rows:
        bam = str(Path(str(row["local_slice_bam"])))
        values = values_by_bam[bam]
        total = sum(float(values[feature]) for feature in features)
        # Every matrix starts with identical sample metadata and then appends
        # feature counts/usages for the current output type.
        out_row: dict[str, object] = {key: row[key] for key in QUANT_SAMPLE_COLUMNS}
        out_row["sample_matrix_key"] = _sample_key(row)
        out_row["feature_total"] = total
        if extra_totals_by_bam:
            for name, totals in extra_totals_by_bam.items():
                if isinstance(totals, dict):
                    out_row[name] = totals[bam]
                else:
                    out_row[name] = totals
        normalizer = normalizer_by_bam[bam] if normalizer_by_bam else total
        for feature in features:
            value = float(values[feature])
            out_row[feature] = (value / normalizer) if normalize and normalizer > 0 else value
        output_rows.append(out_row)
    extra_cols = list(extra_totals_by_bam or {})
    columns = (
        list(QUANT_SAMPLE_COLUMNS) + ["sample_matrix_key", "feature_total"] + extra_cols + features
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows, columns=columns).to_csv(path, sep="\t", index=False)
    log(f"wrote {project_path(path)} ({len(output_rows):,} rows)")


def quantify_tcga_ankrd30a(
    *,
    force: bool = False,
) -> dict[str, object]:
    """Quantify TCGA ANKRD30A with StringTie transcripts and featureCounts probes."""
    if tcga_quantification_ready() and not force:
        return {
            "status": "cached",
            "n_bams": None,
            "outputs": [str(_quant_path(name)) for name in TCGA_QUANTIFICATION_OUTPUTS],
        }

    rows = _resolve_tcga_bam_rows()
    # Recompute the four matrices together so count and usage outputs stay in sync.
    _write_tcga_quantification_matrices(rows)
    computed_outputs = [
        str(_quant_path(name)) for name in TCGA_QUANTIFICATION_OUTPUTS if _quant_path(name).exists()
    ]

    if not tcga_quantification_ready():
        raise RuntimeError("TCGA ANKRD30A quantification did not produce all required matrices")
    status = "computed"
    return {"status": status, "n_bams": len(rows), "outputs": computed_outputs}


def _norm_er_status(value: object) -> str:
    """Normalize cBioPortal ER status labels to Positive/Negative/Unknown."""
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if text in {"", "[Not Available]", "Indeterminate", "Equivocal", "nan"}:
        return "Unknown"
    if text.lower().startswith("pos"):
        return "Positive"
    if text.lower().startswith("neg"):
        return "Negative"
    return "Unknown"


def _load_tcga_clinical_status() -> pd.DataFrame:
    """Derive ER status from the raw cBioPortal sample clinical table."""
    clinical = _read_required_tsv(
        TCGA_CBIO_CLINICAL_SAMPLE_TXT,
        "TCGA cBioPortal clinical sample table",
        comment="#",
        dtype=str,
    )
    _require_columns(
        clinical,
        ["SAMPLE_ID", "ER_STATUS_BY_IHC"],
        "TCGA cBioPortal clinical sample table",
    )
    # The figures only stratify by ER status; overall survival comes from the
    # separate GDC clinical table loaded by the visualization notebook.
    clinical["er_status"] = clinical["ER_STATUS_BY_IHC"].map(_norm_er_status)
    clinical["sample15"] = clinical["SAMPLE_ID"].astype(str).str.slice(0, 15)
    out = clinical[["sample15", "er_status"]].drop_duplicates("sample15")
    if not out["er_status"].isin(["Positive", "Negative"]).any():
        raise ValueError("TCGA cBioPortal clinical sample table has no usable ER status values")
    return out


def _merge_clinical_status(metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach local ER status needed by TCGA figure panels."""
    clinical = _load_tcga_clinical_status()
    _require_columns(metrics, ["sample15"], "TCGA ANKRD30A metric table")
    clinical = clinical[["sample15", "er_status"]].drop_duplicates("sample15")
    metrics = metrics.merge(clinical, on="sample15", how="left", suffixes=("", "_clinical"))
    if "er_status_clinical" in metrics:
        if "er_status" in metrics:
            metrics["er_status"] = metrics["er_status"].fillna(metrics["er_status_clinical"])
        else:
            metrics["er_status"] = metrics["er_status_clinical"]
        metrics = metrics.drop(columns="er_status_clinical")
    if not metrics["er_status"].isin(["Positive", "Negative"]).any():
        raise ValueError("TCGA ANKRD30A metric table could not be annotated with ER status")
    return metrics


def _merge_quantification_matrix(
    metrics: pd.DataFrame,
    *,
    filename: str,
    suffix: str,
    features: Iterable[str],
    matrix_label: str,
    total_prefix: str,
    feature_column_name: Callable[[str], str],
    include_gene_region_total: bool = True,
) -> pd.DataFrame:
    """Merge one TCGA count/usage matrix into the compact sample metric table."""
    path = _quant_path(filename)
    table = _read_required_tsv(path, f"TCGA {matrix_label} matrix")
    _require_columns(table, ["sample_matrix_key"], f"TCGA {matrix_label} matrix")

    required_features = [str(feature) for feature in features]
    _require_columns(table, required_features, f"TCGA {matrix_label} matrix")
    required_totals = ["feature_total"]
    if include_gene_region_total:
        required_totals.append("gene_region_total")
    _require_columns(table, required_totals, f"TCGA {matrix_label} matrix")
    cols = ["sample_matrix_key"] + required_totals + required_features
    table = table[cols].copy()
    # Prefix every merged feature so transcript and probe-region metrics cannot collide.
    rename: dict[str, str] = {}
    if "feature_total" in table:
        rename["feature_total"] = f"ankrd30a_{total_prefix}_{suffix}_total"
    if "gene_region_total" in table:
        rename["gene_region_total"] = f"ankrd30a_gene_region_{suffix}_total"
    for feature in required_features:
        rename[feature] = f"ankrd30a_{feature_column_name(feature)}_{suffix}"
    merged = metrics.merge(table.rename(columns=rename), on="sample_matrix_key", how="left")
    required_merged = [rename[col] for col in [*required_totals, *required_features]]
    missing_values = merged[required_merged].isna().any(axis=1)
    if missing_values.any():
        n_missing = int(missing_values.sum())
        raise ValueError(
            f"TCGA {matrix_label} matrix is missing values for {n_missing:,} samples after merge"
        )
    return merged


def _add_event_fraction_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add event-level transcript and probe fractions used by TCGA validation."""
    out = metrics.copy()

    def fraction(numerator_cols: list[str], other_cols: list[str]) -> pd.Series:
        _require_columns(out, [*numerator_cols, *other_cols], "TCGA ANKRD30A metric table")
        numerator = out[numerator_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        other = out[other_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        return numerator / (numerator + other).replace(0, np.nan)

    # Usage fractions are equivalent to count fractions here because each feature is
    # normalized by the same ANKRD30A gene-region denominator within a sample.
    out["ankrd30a_pb84_2_usage_fraction"] = fraction(
        ["ankrd30a_PB_3828_84_usage", "ankrd30a_PB_3828_2_usage"],
        ["ankrd30a_PB_3828_56_usage", "ankrd30a_PB_3828_107_usage"],
    )
    out["ankrd30a_probe_45213_usage_fraction"] = fraction(
        ["ankrd30a_probe_45213_usage"],
        ["ankrd30a_probe_7939_usage"],
    )
    return out


def prepare_tcga_metric_table(*, force: bool = False) -> pd.DataFrame:  # noqa: C901
    """Build or load the merged TCGA sample metric table.

    Transcript and direct probe-region matrices are stored separately on disk.
    This function joins the focused columns needed by the validation panels into
    one compact sample table.
    """
    quant_ready = tcga_quantification_ready()
    if TCGA_METRIC_TABLE_TSV.exists() and not force and quant_ready:
        cached = pd.read_csv(TCGA_METRIC_TABLE_TSV, sep="\t")
        _require_columns(
            cached,
            [
                "sample15",
                "sample_type",
                "er_status",
                "ankrd30a_gene_region_counts_total",
                *[metric for metric, _label in CORRELATION_TARGETS],
            ],
            "cached TCGA ANKRD30A metric table",
        )
        return cached

    base = _read_required_tsv(
        _quant_path(TRANSCRIPT_COUNT_NAME), "TCGA PacBio transcript count matrix"
    )
    _require_columns(base, SAMPLE_COLUMNS, "TCGA PacBio transcript count matrix")
    meta_cols = list(SAMPLE_COLUMNS)
    metrics = base[meta_cols].copy()
    metrics["sample15"] = metrics["sample_submitter_id"].astype(str).str.slice(0, 15)

    # Keep upstream matrices separate on disk, then join only the columns needed
    # by TCGA validation and figure panels into one compact sample table.
    for (
        filename,
        suffix,
        features,
        matrix_label,
        total_prefix,
        name_fn,
        include_gene_region_total,
    ) in [
        (
            TRANSCRIPT_COUNT_NAME,
            "counts",
            PB_TRANSCRIPTS_OF_INTEREST,
            "PacBio transcript counts",
            "pacbio_transcript",
            lambda tx: str(tx).replace(".", "_"),
            False,
        ),
        (
            TRANSCRIPT_USAGE_NAME,
            "usage",
            PB_TRANSCRIPTS_OF_INTEREST,
            "PacBio transcript usage",
            "pacbio_transcript",
            lambda tx: str(tx).replace(".", "_"),
            False,
        ),
        (
            PROBE_REGION_COUNTS_NAME,
            "counts",
            PROBE_REGION_FEATURES,
            "probe-region counts",
            "probe_region",
            str,
            True,
        ),
        (
            PROBE_REGION_USAGE_NAME,
            "usage",
            PROBE_REGION_FEATURES,
            "probe-region usage",
            "probe_region",
            str,
            True,
        ),
    ]:
        metrics = _merge_quantification_matrix(
            metrics,
            filename=filename,
            suffix=suffix,
            features=features,
            matrix_label=matrix_label,
            total_prefix=total_prefix,
            feature_column_name=name_fn,
            include_gene_region_total=include_gene_region_total,
        )
    metrics = _merge_clinical_status(metrics)

    for col in metrics.columns:
        if col.startswith("ankrd30a_"):
            metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
    # Add the two event-level usage fractions plotted and correlated in the
    # DU/TCGA supplementary figure.
    metrics = _add_event_fraction_metrics(metrics)
    _require_columns(
        metrics,
        [
            "sample15",
            "sample_type",
            "er_status",
            "ankrd30a_gene_region_counts_total",
            "ankrd30a_pb84_2_usage_fraction",
            "ankrd30a_probe_45213_usage_fraction",
        ],
        "assembled TCGA ANKRD30A metric table",
    )
    sort_cols = [
        col
        for col in [
            "case_submitter_id",
            "sample15",
            "ankrd30a_pacbio_transcript_counts_total",
            "ankrd30a_probe_region_counts_total",
        ]
        if col in metrics
    ]
    if sort_cols:
        ascending = [True] * len(sort_cols)
        if sort_cols[-1] in {
            "ankrd30a_pacbio_transcript_counts_total",
            "ankrd30a_probe_region_counts_total",
        }:
            # When duplicate sample IDs exist, keep the row with more counting evidence.
            ascending[-1] = False
        metrics = metrics.sort_values(sort_cols, ascending=ascending)
    metrics = metrics.drop_duplicates("sample15")

    TCGA_METRIC_TABLE_TSV.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TCGA_METRIC_TABLE_TSV, sep="\t", index=False)
    log(f"wrote {project_path(TCGA_METRIC_TABLE_TSV)}")
    return metrics


def _primary_tumor_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Keep one primary-tumor row per TCGA sample and normalize ER status labels."""
    if metrics.empty:
        raise ValueError("TCGA ANKRD30A metric table is empty")
    _require_columns(
        metrics,
        ["sample_type", "sample15", "er_status"],
        "TCGA ANKRD30A metric table",
    )
    metrics = metrics[metrics["sample_type"].astype(str).eq("Primary Tumor")].copy()
    if metrics.empty:
        raise ValueError("TCGA ANKRD30A metric table has no Primary Tumor rows")
    metrics["sample15"] = metrics["sample15"].astype(str)
    metrics["er_status_clean"] = metrics["er_status"].fillna("Unknown").astype(str).str.lower()
    sort_cols = [
        col
        for col in [
            "case_submitter_id",
            "sample15",
            "ankrd30a_pacbio_transcript_counts_total",
            "ankrd30a_probe_region_counts_total",
        ]
        if col in metrics
    ]
    if sort_cols:
        ascending = [True] * len(sort_cols)
        if sort_cols[-1] in {
            "ankrd30a_pacbio_transcript_counts_total",
            "ankrd30a_probe_region_counts_total",
        }:
            # For duplicate primary-tumor rows, prefer the one with more evidence.
            ascending[-1] = False
        metrics = metrics.sort_values(sort_cols, ascending=ascending)
    metrics = metrics.drop_duplicates("sample15")
    return metrics


def load_tcga_metrics(*, force: bool = False) -> pd.DataFrame:
    """Load primary-tumor TCGA metrics with ER status normalized for tests."""
    metrics = prepare_tcga_metric_table(force=force)
    return _primary_tumor_metrics(metrics)


def prepare_tcga_regulator_expression(  # noqa: C901
    regulators: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load selected regulator expression rows from the raw cBioPortal RSEM matrix."""
    columns = [
        "sample15",
        "case_submitter_id",
        "regulator",
        "regulator_class",
        "regulator_expression",
    ]
    if regulators is None:
        raise ValueError("TCGA regulator expression requires a non-empty regulator list")
    requested: dict[str, str] = {}
    for regulator in regulators:
        key = normalize_gene_symbol(regulator)
        if key and key not in requested:
            requested[key] = str(regulator)
    if not requested:
        raise ValueError("TCGA regulator expression requires a non-empty regulator list")
    if not TCGA_CBIO_RSEM_TXT.exists():
        raise FileNotFoundError(
            f"Missing TCGA cBioPortal RSEM expression matrix: {project_path(TCGA_CBIO_RSEM_TXT)}"
        )

    frames: list[pd.DataFrame] = []
    with TCGA_CBIO_RSEM_TXT.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 3:
            raise ValueError(
                f"Invalid TCGA cBioPortal RSEM expression matrix header: {project_path(TCGA_CBIO_RSEM_TXT)}"
            )
        sample_cols = header[2:]
        # Stream the wide matrix row-wise; only selected prioritized TF rows are
        # needed for the final figure.
        for parts in reader:
            if len(parts) < 3:
                continue
            key = normalize_gene_symbol(parts[0])
            if key not in requested:
                continue
            # Load only requested regulator rows from the wide cBioPortal matrix,
            # then convert them into a long sample-by-regulator table.
            raw_values = parts[2 : 2 + len(sample_cols)]
            if len(raw_values) < len(sample_cols):
                raw_values.extend([""] * (len(sample_cols) - len(raw_values)))
            values = pd.to_numeric(pd.Series(raw_values, index=sample_cols), errors="coerce")
            frame = pd.DataFrame(
                {
                    "sample15": values.index.astype(str).str.slice(0, 15),
                    "case_submitter_id": values.index.astype(str).str.slice(0, 12),
                    "regulator": requested[key],
                    "regulator_class": "TF",
                    "regulator_expression": values.to_numpy(dtype=float),
                }
            )
            frames.append(frame)
    if not frames:
        raise ValueError(
            "TCGA cBioPortal RSEM expression matrix contains none of the requested regulators: "
            + ", ".join(requested.values())
        )
    expr = pd.concat(frames, ignore_index=True)
    missing_regulators = sorted(set(requested.values()) - set(expr["regulator"].astype(str)))
    if missing_regulators:
        raise ValueError(
            "TCGA cBioPortal RSEM expression matrix is missing requested regulators: "
            + ", ".join(missing_regulators)
        )
    expr = expr.groupby(
        ["sample15", "case_submitter_id", "regulator", "regulator_class"],
        as_index=False,
    )["regulator_expression"].mean()
    return expr[columns].copy()


def compute_regulator_correlations(  # noqa: C901
    metrics: pd.DataFrame,
    regulators: list[str],
    expr: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Test ER+ TCGA associations between regulator expression and PB probe metrics."""
    # metrics is the sample-level ANKRD30A validation table: clinical ER status,
    # PB transcript count/usage ratios, and Atera probe-region count/usage ratios.
    if expr is None:
        # expr is a long sample-by-regulator cBioPortal RSEM expression table;
        # values stay raw here and are log1p-transformed for Spearman below.
        expr = prepare_tcga_regulator_expression(regulators)
    if metrics.empty:
        raise ValueError("TCGA ANKRD30A metric table is empty")
    if not regulators:
        raise ValueError("No prioritized regulators were selected for TCGA correlations")
    if expr.empty:
        raise ValueError("TCGA regulator-expression table is empty")
    _require_columns(
        expr,
        ["sample15", "regulator", "regulator_expression"],
        "TCGA regulator-expression table",
    )
    _require_columns(
        metrics,
        ["sample15", "case_submitter_id", "er_status_clean"],
        "TCGA ANKRD30A metric table",
    )
    expr = expr[expr["regulator"].astype(str).isin(regulators)].copy()
    if expr.empty:
        raise ValueError("TCGA regulator-expression table has no prioritized regulators")
    er = metrics[metrics["er_status_clean"].eq("positive")].copy()
    if er.empty:
        raise ValueError("TCGA ANKRD30A metric table has no ER-positive primary tumors")
    target_cols = [metric for metric, _ in CORRELATION_TARGETS]
    _require_columns(er, target_cols, "TCGA ER-positive ANKRD30A metric table")
    er = er[["sample15", "case_submitter_id"] + target_cols]
    rows: list[dict[str, object]] = []
    label_by_metric = dict(CORRELATION_TARGETS)
    for regulator, g in expr.groupby("regulator"):
        merged = er.merge(g, on="sample15", how="inner")
        for metric in target_cols:
            sub = (
                merged[["regulator_expression", metric]].replace([np.inf, -np.inf], np.nan).dropna()
            )
            if (
                sub.shape[0] < CORRELATION_MIN_SAMPLES
                or sub["regulator_expression"].nunique() < 2
                or sub[metric].nunique() < 2
            ):
                rho, pvalue = np.nan, np.nan
            else:
                target_values = sub[metric]
                if str(metric).endswith("_counts"):
                    target_values = np.log1p(target_values.clip(lower=0))
                # Spearman is used only within ER+ primary tumors to match the
                # validation question in the DU/TCGA figure; regulator
                # expression is log1p(TF/RBP RSEM expression).
                rho, pvalue = spearmanr(
                    np.log1p(sub["regulator_expression"].clip(lower=0)), target_values
                )
            rows.append(
                {
                    "regulator": regulator,
                    "tf": regulator,
                    "target_metric": metric,
                    "target_label": label_by_metric[metric],
                    "n_er_positive_primary_tumors": int(sub.shape[0]),
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "pvalue": float(pvalue) if np.isfinite(pvalue) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out["pvalue_adj"] = bh_fdr(out["pvalue"].to_numpy())
    order = {regulator: i for i, regulator in enumerate(regulators)}
    out["_regulator_order"] = out["regulator"].map(order).fillna(999)
    return out.sort_values(["pvalue_adj", "_regulator_order"]).drop(columns="_regulator_order")
