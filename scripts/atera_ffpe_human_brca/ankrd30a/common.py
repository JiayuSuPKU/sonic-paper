"""Common constants, dependency settings, and helpers for ANKRD30A analyses."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..config import (
    ANKRD30A_PROBE_MAPPINGS_CSV,
    ANKRD30A_REGULATOR_PRIORITY_CSV,
    ANKRD30A_TRANSCRIPT_ANNOTATIONS_CSV,
    HOME,
    TCGA_ANALYSIS_DIR,
)
from ..utils import project_path

TARGET_GENE = "ANKRD30A"
DEFAULT_SLICE_REGION = "chr10:37115598-37242567"
PB_TSS_CHROM = "chr10"
PB_TSS_WINDOW_BP = 10_000
PB_TSS_CORE_BP = 1000


def default_slice_region_1based() -> tuple[str, int, int]:
    """Return the fixed ANKRD30A validation interval as 1-based inclusive coordinates."""
    chrom, coords = DEFAULT_SLICE_REGION.split(":", 1)
    start, end = coords.replace(",", "").split("-", 1)
    return chrom, int(start), int(end)


# The focused Atera event compares a luminal/DCIS-compatible PB class against an
# invasive-compatible PB class; downstream ratios and colors use this ordering.
PB_EVENT_TRANSCRIPTS_BY_CLASS = {
    "45213+50762": ("PB.3828.84", "PB.3828.2"),
    "50762+7939": ("PB.3828.56", "PB.3828.107"),
}
PB_TRANSCRIPTS_OF_INTEREST = tuple(
    tx for transcripts in PB_EVENT_TRANSCRIPTS_BY_CLASS.values() for tx in transcripts
)
PB_EVENT_CLASS_BY_TRANSCRIPT = {
    tx: event_class
    for event_class, probe_class in (
        ("luminal", "45213+50762"),
        ("invasive", "50762+7939"),
    )
    for tx in PB_EVENT_TRANSCRIPTS_BY_CLASS[probe_class]
}
PB_TRANSCRIPT_SHORT_LABELS = {
    tx: tx.replace("PB.3828.", "PB") for tx in PB_TRANSCRIPTS_OF_INTEREST
}
PB_TSS_HYPOTHESIS_TRANSCRIPTS = PB_TRANSCRIPTS_OF_INTEREST

# T47D dTAG-TRPS1 PRO-seq constants. The processed GEO bigWigs are
# timepoint-combined strand-specific normalized tracks; these settings define
# the narrow ANKRD30A-region cache consumed by the figure notebook.
TRPS1_PROSEQ_GEO_ACCESSION = "GSE236172"
TRPS1_PROSEQ_SOURCE_BASE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE236nnn/GSE236172/suppl/"
)
TRPS1_PROSEQ_TIMEPOINTS = (
    ("0h", 0.0, "0hour"),
    ("0.5h", 0.5, "0.5hour"),
    ("1h", 1.0, "1hour"),
    ("2h", 2.0, "2hour"),
    ("4h", 4.0, "4hour"),
    ("24h", 24.0, "24hour"),
)
TRPS1_PROSEQ_TSS_HALF_WINDOW_BP = 1_000
TRPS1_PROSEQ_SIGNAL_EPSILON = 1e-3
PYBIGWIG_MODULE = "pyBigWig"
PYBIGWIG_INSTALL_HINT = (
    "Install pyBigWig in the quadsv_dev environment before running "
    "ankrd30a/run_trps1_proseq.py."
)

CORRELATION_TARGETS = (
    ("ankrd30a_pb84_2_usage_fraction", "PB84+2 usage fraction"),
    ("ankrd30a_probe_45213_usage_fraction", "probe 45213 usage fraction"),
)
CORRELATION_MIN_SAMPLES = 8

# Quantification results of selected ANKRD30A transcripts and probe regions in
# TCGA-BRCA samples are cached under this subdirectory of the main result root.
LOCAL_TCGA_QUANT_DIR = TCGA_ANALYSIS_DIR / "quantification"
TRANSCRIPT_COUNT_NAME = "ANKRD30A_pacbio_qc_transcript_counts.tsv"
TRANSCRIPT_USAGE_NAME = "ANKRD30A_pacbio_qc_transcript_usage.tsv"
PROBE_REGION_COUNTS_NAME = "ANKRD30A_pacbio_qc_probe_region_counts.tsv"
PROBE_REGION_USAGE_NAME = "ANKRD30A_pacbio_qc_probe_region_usage.tsv"
TCGA_QUANTIFICATION_OUTPUTS = (
    TRANSCRIPT_COUNT_NAME,
    TRANSCRIPT_USAGE_NAME,
    PROBE_REGION_COUNTS_NAME,
    PROBE_REGION_USAGE_NAME,
)

# External tools used only by the ANKRD30A TCGA quantification stage.
BIOINFO_BIN = HOME / "miniforge3" / "envs" / "bioinfo" / "bin"
STRINGTIE_BIN = BIOINFO_BIN / "stringtie"
STRINGTIE_PREPDE_BIN = BIOINFO_BIN / "prepDE.py"
STRINGTIE_PREPDE_READ_LENGTH = 75
FEATURECOUNTS_BIN = BIOINFO_BIN / "featureCounts"
FEATURECOUNTS_THREADS = 4
FEATURECOUNTS_BATCH_SIZE = 100
BEDTOOLS_BIN = Path("/opt/homebrew/bin/bedtools")

SAMPLE_COLUMNS = (
    "file_id",
    "case_submitter_id",
    "case_id",
    "sample_submitter_id",
    "sample_id",
    "sample_type",
    "aliquot_submitter_id",
    "bam_file_name",
    "workflow_type",
    "sample_matrix_key",
)
QUANT_SAMPLE_COLUMNS = (
    "file_id",
    "case_submitter_id",
    "case_id",
    "sample_submitter_id",
    "sample_id",
    "sample_type",
    "aliquot_submitter_id",
    "bam_file_name",
    "workflow_type",
    "local_slice_bam",
)
PROBE_REGION_CODEWORDS = (45213, 50762, 7939, 6037)
PROBE_REGION_FEATURES = tuple(
    f"probe_{codeword}" for codeword in PROBE_REGION_CODEWORDS
)
PROBE_CLASS_ORDER = {
    "45213-only": 0,
    "45213+50762": 1,
    "50762-only": 2,
    "50762+7939": 3,
    "7939-only": 4,
    "broad long-compatible": 5,
    "other multi-probe": 6,
    "no-overlap": 7,
}
ANKRD30A_VISUALIZATION_CACHE_FILES = (
    ANKRD30A_TRANSCRIPT_ANNOTATIONS_CSV,
    ANKRD30A_PROBE_MAPPINGS_CSV,
    ANKRD30A_REGULATOR_PRIORITY_CSV,
)


def _codewords_from_cache(value: object) -> set[int]:
    """Parse cached compatibility groups such as ``50762+7939``."""
    if pd.isna(value):
        return set()
    text = str(value)
    if not text or text == "no_probe_overlap":
        return set()
    return {int(token) for token in re.split(r"[^0-9]+", text) if token}


def _probe_compatibility_class(codewords: set[int]) -> str:
    """Collapse codeword hits into ANKRD30A figure compatibility classes."""
    if not codewords:
        return "no-overlap"
    if codewords == {45213}:
        return "45213-only"
    if codewords == {50762}:
        return "50762-only"
    if codewords == {7939}:
        return "7939-only"
    if 50762 in codewords and 7939 in codewords and 45213 not in codewords:
        return "50762+7939"
    if 45213 in codewords and 50762 in codewords and 7939 not in codewords:
        return "45213+50762"
    if 6037 in codewords:
        return "broad long-compatible"
    return "other multi-probe"


def _quant_path(filename: str) -> Path:
    """Return the Atera-local path for one TCGA quantification matrix."""
    return LOCAL_TCGA_QUANT_DIR / filename


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    """Raise a concise error when a required table schema is incomplete."""
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _read_required_tsv(
    path: Path, label: str, **read_csv_kwargs: object
) -> pd.DataFrame:
    """Read one required TSV cache used by the final figure workflow."""
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing required {label}: {project_path(path)}")
    table = pd.read_csv(path, sep="\t", **read_csv_kwargs)
    if table.empty:
        raise ValueError(f"Required {label} is empty: {project_path(path)}")
    return table


def _probe_region_feature_name(codeword: int | str) -> str:
    """Return the matrix feature name for one Atera ANKRD30A codeword."""
    return f"probe_{int(codeword)}"


def _resolve_tool(path: Path, label: str) -> str:
    """Return an executable path if configured and present."""
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"{label} executable not found at {project_path(path)}")
