"""Shared paths and numerical analysis configuration."""

from __future__ import annotations

from scripts.paper_utils import PROJECT_ROOT, ensure_directories

DATA_DIR = PROJECT_ROOT / "data" / "qtest_benchmark"
RESULT_DIR = PROJECT_ROOT / "results" / "qtest_benchmark"
FIGURE_DIR = PROJECT_ROOT / "figures" / "qtest_benchmark"

# Runtime and implementation benchmark files
BENCHMARK_SCALING_FILE = RESULT_DIR / "benchmark_scaling.csv"
BENCHMARK_PROJECTION_FILE = RESULT_DIR / "benchmark_projection.csv"
VALIDATION_RESULT_DIR = RESULT_DIR / "validation"

# Small intestine illustration input and output files
RAW_SPATIALDATA = DATA_DIR / "visium_hd_mouse_intestine.zarr"
SMALL_INTESTINE_SCORES_FILE = RESULT_DIR / "small_intestine_scores.csv"
SMALL_INTESTINE_PATTERN_RANKS_FILE = RESULT_DIR / "small_intestine_pattern_ranks.csv"

# Lineage tracing benchmark input and output files
LINEAGE_DATA_DIR = DATA_DIR / "lineage_tracing"
LINEAGE_ANALYSIS_DATA_FILE = LINEAGE_DATA_DIR / "analysis_tumors.h5ad"
LINEAGE_TREE_DIR = LINEAGE_DATA_DIR / "trees"

LINEAGE_RESULT_DIR = RESULT_DIR / "lineage_tracing"
LINEAGE_CLUSTER_SCORES_FILE = LINEAGE_RESULT_DIR / "cluster_scores.csv.gz"
LINEAGE_GENE_SCORES_FILE = LINEAGE_RESULT_DIR / "gene_scores.csv.gz"
LINEAGE_BIVARIATE_SUMMARY_FILE = LINEAGE_RESULT_DIR / "bivariate_summary.csv"
LINEAGE_GENE_DIAGNOSTICS_FILE = LINEAGE_RESULT_DIR / "gene_diagnostics.npz"
LINEAGE_BIVARIATE_DIAGNOSTICS_FILE = LINEAGE_RESULT_DIR / "bivariate_diagnostics.npz"
LINEAGE_TARGET_METADATA_FILE = LINEAGE_RESULT_DIR / "target_metadata.csv.gz"

# PDF figure files
FRAMEWORK_FIGURE_FILE = FIGURE_DIR / "main_framework.pdf"
RUNTIME_MAIN_FIGURE_FILE = FIGURE_DIR / "main_runtime.pdf"
VALIDATION_SUPPLEMENTARY_FIGURE_FILE = FIGURE_DIR / "supp_validation.pdf"
LINEAGE_MAIN_FIGURE_FILE = FIGURE_DIR / "main_lineage.pdf"
LINEAGE_SUPPLEMENTARY_FIGURE_FILE = FIGURE_DIR / "supp_lineage.pdf"

# Small intestine example genes and config parameters
ARTIFACT_GENES = ("Reg3b", "Fabp6", "Fabp2", "Krt19")
IMMUNE_GENES = ("Coro1a", "Ptprc", "Cd37", "Ms4a1")

CAR_K_NEIGHBORS = 4
CAR_RHOS = (0.10, 0.30, 0.50, 0.70, 0.90, 0.95)
SLOW_DECAY_RHO = 0.70
FAST_DECAY_RHO = 0.95
PIXEL_SIZE_UM = 16.0

# Runtime benchmark parameters
M_FEATURES = 10_000
RUNTIME_MEMORY_BUDGET_MB = 8_000
BENCHMARK_TARGETS = (
    ("Visium", 5_000, "regular"),
    ("Slide-seqV2", 50_000, "irregular"),
    ("Visium HD 16µm", 100_000, "regular"),
    ("Xenium", 200_000, "irregular"),
    ("Visium HD 2µm", 6_000_000, "regular"),
)


def ensure_output_dirs() -> None:
    """Create ignored result and figure directories."""
    ensure_directories(RESULT_DIR, FIGURE_DIR)
