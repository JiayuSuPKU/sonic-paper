"""Configuration for embryo spatial-pattern spectral embeddings.

This module owns both shared analysis constants and the two species-specific
configuration objects:

- ``MOUSE`` — Stereo-seq MOSTA mouse embryo sections.
- ``HUMAN`` — Stereo-seq HESTA human embryo sections.

The radial-spectrum embedding uses geometric, low-frequency-dense bins at
``N_RADIAL_BINS=80`` (merged to a shared-supported grid), L1-log shape scaling,
equal frequency weighting, per-bin z-score, and ``N_PC=5``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.paper_utils import PROJECT_ROOT, ensure_directories

# --- repo-relative roots -----------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "data" / "embryo_xspecies"
RESULT_DIR = PROJECT_ROOT / "results" / "embryo_xspecies"
FIGURE_DIR = PROJECT_ROOT / "figures" / "embryo_xspecies"
COMPARISON_RESULT_DIR = RESULT_DIR / "comparison"
MOUSE_ONLY_RESULT_DIR = RESULT_DIR / "mouse_only"
HUMAN_ONLY_RESULT_DIR = RESULT_DIR / "human_only"

# --- shared spatial-pattern analysis ----------------------------------------
# Use library-size-normalized log1p expression over every configured slide.
# A downstream gene must be detected in strictly more than 1% of observed bins
# in every slide of its species.
ANALYSIS_EXPRESSION_MODE = "log1p_norm"
ANALYSIS_NORMALIZATION_TARGET_SUM = 1e4
MIN_GENE_DETECTED_BIN_FRACTION = 0.01

# --- SONIC spatial-variability detector configuration ----------------------
# Each species config exposes ``SV_RESULT_DIR`` and ``SV_MANIFEST_JSON`` under
# its mouse-only or human-only result directory. Every gene returned by the
# default panel selector is tested; raw Q, Z, and p-values are retained without
# FDR filtering.
SV_KERNEL_METHOD = "car"
SV_RHO = 0.9
SV_NEIGHBOR_DEGREE = 1
SV_FFT_SOLVER = "rfft2"
SV_CHUNK_SIZE = 16
SV_FFT_WORKERS = 4
SV_N_JOBS = 2

# --- spatial/spectral grid shared across species ----------------------------
# Stereo-seq bin50 means 50 x 50 DNBs. The DNB pitch is 0.5 µm, so one bin50
# observation spans 25 µm on each side.
STEREO_DNB_PITCH_UM = 0.5
BIN_SIZE_DNB = 50
BIN_SIZE_UM = STEREO_DNB_PITCH_UM * BIN_SIZE_DNB
N_RADIAL_BINS = 80

# --- radial-spectrum computation and PCA embedding --------------------------
FREQ_SPACING = "geom"
N_PC = 5

# --- embedding metrics, clustering, and GO enrichment -----------------------
LOCAL_MIXING_NEIGHBORS = 50
LOCAL_MIXING_QUANTILES = 5
N_PATTERN_CLUSTERS = 5
TEMPORAL_MAX_WITHIN_STAGE_PERCENTILE = 0.95
TEMPORAL_DYNAMIC_PERCENTILE_THRESHOLD = 0.9
TRANSITION_FDR_THRESHOLD = 0.05
GO_SOURCE = "GO:BP"
GO_ORGDB = {"mouse": "org.Mm.eg.db", "human": "org.Hs.eg.db"}
GO_SEMANTIC_SIMPLIFY = True
GO_TIMEOUT_SECONDS = 900

# Display filters are applied only to compact presentation tables; raw
# clusterProfiler results remain cached in full.
GO_FDR_THRESHOLD = 0.05
GO_MAX_TERM_UNIVERSE_FRACTION = 0.05
GO_DISPLAY_TERMS_PER_GROUP = 4

# Cache revisions are bumped only when result semantics change without a
# corresponding parameter change.  The parameter maps below are the canonical
# cache inputs documented by workflow.py and embedded in derived manifests.
CACHE_REVISIONS = {"metrics": 1, "temporal": 1}
SV_CACHE_PARAMETERS = {
    "expression_mode": ANALYSIS_EXPRESSION_MODE,
    "normalization_target_sum": ANALYSIS_NORMALIZATION_TARGET_SUM,
    "min_gene_detected_bin_fraction": MIN_GENE_DETECTED_BIN_FRACTION,
    "detector": {
        "kernel_method": SV_KERNEL_METHOD,
        "rho": SV_RHO,
        "neighbor_degree": SV_NEIGHBOR_DEGREE,
        "spacing_um": [BIN_SIZE_UM, BIN_SIZE_UM],
        "fft_solver": SV_FFT_SOLVER,
    },
    "fdr_filter_applied": False,
}
SPECTRAL_CACHE_PARAMETERS = {
    "expression_mode": ANALYSIS_EXPRESSION_MODE,
    "normalization_target_sum": ANALYSIS_NORMALIZATION_TARGET_SUM,
    "min_gene_detected_bin_fraction": MIN_GENE_DETECTED_BIN_FRACTION,
    "background_normalized": True,
    "bin_size_um": BIN_SIZE_UM,
    "n_radial_bins_requested": N_RADIAL_BINS,
    "frequency_spacing": FREQ_SPACING,
}
EMBEDDING_CACHE_PARAMETERS = {**SPECTRAL_CACHE_PARAMETERS, "n_pc": N_PC}
GO_CACHE_PARAMETERS = {
    "source": GO_SOURCE,
    "orgdb": GO_ORGDB,
    "semantic_simplify": GO_SEMANTIC_SIMPLIFY,
    "semantic_simplify_p_adjust_cutoff": GO_FDR_THRESHOLD,
    "maximum_term_universe_fraction": GO_MAX_TERM_UNIVERSE_FRACTION,
    "display_terms_per_group": GO_DISPLAY_TERMS_PER_GROUP,
    "timeout_seconds": GO_TIMEOUT_SECONDS,
}
METRICS_CACHE_PARAMETERS = {
    "n_pattern_clusters": N_PATTERN_CLUSTERS,
    "local_mixing_neighbors": LOCAL_MIXING_NEIGHBORS,
    "local_mixing_quantiles": LOCAL_MIXING_QUANTILES,
    "go": GO_CACHE_PARAMETERS,
}
TEMPORAL_CACHE_PARAMETERS = {
    "max_within_stage_percentile": TEMPORAL_MAX_WITHIN_STAGE_PERCENTILE,
    "dynamic_percentile_threshold": TEMPORAL_DYNAMIC_PERCENTILE_THRESHOLD,
    "transition_fdr_threshold": TRANSITION_FDR_THRESHOLD,
    "go": GO_CACHE_PARAMETERS,
}

# --- spatial-domain annotation ----------------------------------------------
# Generated comparator feature prefix, not an input adata column.
DOMAIN_CHANNEL_PREFIX = "__domain__"


def _species_config(
    *,
    panel_key: str,
    species: str,
    data_key: str,
    result_dir: Path,
    annotation_col: str,
    **extra,
) -> SimpleNamespace:
    data_root = DATA_ROOT / data_key
    sv_result_dir = result_dir / "spatial_variability"
    spectra_result_dir = result_dir / "spectral_embedding"
    cfg = SimpleNamespace(
        PANEL_KEY=panel_key,
        SPECIES=species,
        DATA_ROOT=data_root,
        RAW_DIR=data_root / "raw",
        SDATA_DIR=data_root / "sdata",
        META_CSV=data_root / "metadata.csv",
        RESULT_DIR=result_dir,
        SV_RESULT_DIR=sv_result_dir,
        SV_MANIFEST_JSON=sv_result_dir / "manifest.json",
        SPECTRA_RESULT_DIR=spectra_result_dir,
        GEOMETRY_CSV=result_dir / "section_geometry.csv",
        ANNOTATION_COL=annotation_col,
    )
    cfg.__dict__.update(extra)
    return cfg


def ensure_species_output_dirs(cfg: SimpleNamespace) -> None:
    """Create one species' raw, prepared, and cached-result directories."""
    ensure_directories(
        cfg.RAW_DIR,
        cfg.SDATA_DIR,
        cfg.RESULT_DIR,
        cfg.SV_RESULT_DIR,
        cfg.SPECTRA_RESULT_DIR,
    )


# --- mouse: Stereo-seq MOSTA -------------------------------------------------
STAGE_TO_DAY: dict[str, float] = {
    "E9.5": 9.5,
    "E10.5": 10.5,
    "E11.5": 11.5,
    "E12.5": 12.5,
    "E13.5": 13.5,
    "E14.5": 14.5,
    "E15.5": 15.5,
    "E16.5": 16.5,
}
MOUSE_KEPT_TIMEPOINTS: list[str] = [
    "E10.5",
    "E11.5",
    "E12.5",
    "E13.5",
    "E14.5",
    "E15.5",
]
MOSTA_BASE_URL = "https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics"
ALL_SECTIONS: dict[str, list[str]] = {
    "E10.5": ["E10.5_E1S1", "E10.5_E1S2", "E10.5_E1S3", "E10.5_E2S1"],
    "E11.5": ["E11.5_E1S1", "E11.5_E1S2", "E11.5_E1S3", "E11.5_E1S4"],
    "E12.5": [
        "E12.5_E1S1",
        "E12.5_E1S2",
        "E12.5_E1S3",
        "E12.5_E1S4",
        "E12.5_E1S5",
        "E12.5_E2S1",
    ],
    "E13.5": ["E13.5_E1S1", "E13.5_E1S2", "E13.5_E1S3", "E13.5_E1S4"],
    "E14.5": [
        "E14.5_E1S1",
        "E14.5_E1S2",
        "E14.5_E1S3",
        "E14.5_E1S4",
        "E14.5_E1S5",
        "E14.5_E2S1",
        "E14.5_E2S2",
    ],
    "E15.5": ["E15.5_E1S1", "E15.5_E1S2", "E15.5_E1S3", "E15.5_E1S4", "E15.5_E2S1"],
}
REPRESENTATIVE_SECTIONS: dict[str, list[str]] = {
    tp: [sections[0]] for tp, sections in ALL_SECTIONS.items()
}

MOUSE = _species_config(
    panel_key="mouse",
    species="mouse",
    data_key="mosta",
    result_dir=MOUSE_ONLY_RESULT_DIR,
    annotation_col="annotation",
    STAGE_TO_DAY=STAGE_TO_DAY,
    KEPT_TIMEPOINTS=MOUSE_KEPT_TIMEPOINTS,
    MOSTA_BASE_URL=MOSTA_BASE_URL,
    ALL_SECTIONS=ALL_SECTIONS,
    REPRESENTATIVE_SECTIONS=REPRESENTATIVE_SECTIONS,
)

# --- human: Stereo-seq HESTA -------------------------------------------------
# Supplementary Table 3 reports two representative sections per embryo. Use
# those pairs for the 12 configured embryos, yielding four sections per stage
# where two embryos are available. CS17 and CS18 have only one embryo, so
# supplement their reported pairs with the two highest-quality remaining
# sections, ranked within embryo by
#
#   sqrt(mean UMI * mean gene)
#   * valid-CID fraction * clean-read fraction
#   * uniquely-mapped fraction * annotated-read fraction
#
# This balances transcript capture and sequencing quality only where more than
# the publication's two representative sections are needed.
HUMAN_ALL_SECTIONS: dict[str, list[str]] = {
    "CS12-13": ["CS12-13_E1S2", "CS12-13_E1S3", "CS12-13_E2S3", "CS12-13_E2S4"],
    "CS14-15": [
        "CS14-15_E1S3",
        "CS14-15_E1S4",
        "CS14-15_E2S1",
        "CS14-15_E2S2",
    ],
    "CS17": ["CS17_E1S9", "CS17_E1S10", "CS17_E1S6", "CS17_E1S3"],
    "CS18": ["CS18_E1S2", "CS18_E1S3", "CS18_E1S6", "CS18_E1S5"],
    "CS19": ["CS19_E1S1", "CS19_E1S2", "CS19_E2S1", "CS19_E2S2"],
    "CS20": ["CS20_E1S6", "CS20_E1S7", "CS20_E2S2", "CS20_E2S3"],
    "CS23": ["CS23_E1S2", "CS23_E1S3", "CS23_E2S5", "CS23_E2S10"],
}
HUMAN_REPRESENTATIVE_SECTIONS: dict[str, list[str]] = {
    "CS12-13": ["CS12-13_E1S3"],
    "CS14-15": ["CS14-15_E1S3"],
    "CS17": ["CS17_E1S6"],
    "CS18": ["CS18_E1S2"],
    "CS19": ["CS19_E2S2"],
    "CS20": ["CS20_E1S7"],
    "CS23": ["CS23_E2S10"],
}

# Days post-conception for the configured HESTA sections, from Pan et al. 2026
# Supplementary Table 1a ("Summary of Spatiotemporal transcriptomic data").
HESTA_SECTION_DAYS: dict[str, float] = {
    "CS12-13_E1S2": 29.0,
    "CS12-13_E1S3": 29.0,
    "CS12-13_E2S3": 26.0,
    "CS12-13_E2S4": 26.0,
    "CS14-15_E1S3": 31.0,
    "CS14-15_E1S4": 31.0,
    "CS14-15_E2S1": 35.0,
    "CS14-15_E2S2": 35.0,
    "CS17_E1S9": 41.0,
    "CS17_E1S10": 41.0,
    "CS17_E1S6": 41.0,
    "CS17_E1S3": 41.0,
    "CS18_E1S2": 47.0,
    "CS18_E1S3": 47.0,
    "CS18_E1S6": 47.0,
    "CS18_E1S5": 47.0,
    "CS19_E1S1": 44.0,
    "CS19_E1S2": 44.0,
    "CS19_E2S1": 42.0,
    "CS19_E2S2": 42.0,
    "CS20_E1S6": 48.0,
    "CS20_E1S7": 48.0,
    "CS20_E2S2": 50.0,
    "CS20_E2S3": 50.0,
    "CS23_E1S2": 59.0,
    "CS23_E1S3": 59.0,
    "CS23_E2S5": 59.0,
    "CS23_E2S10": 59.0,
}
HESTA_EMBRYO_SEX: dict[str, str] = {
    "CS12-13_E1": "male",
    "CS12-13_E2": "female",
    "CS14-15_E1": "female",
    "CS14-15_E2": "female",
    "CS17_E1": "male",
    "CS18_E1": "female",
    "CS19_E1": "female",
    "CS19_E2": "male",
    "CS20_E1": "female",
    "CS20_E2": "male",
    "CS23_E1": "male",
    "CS23_E2": "male",
}
HESTA_SINGLE_EMBRYO_STAGES = frozenset({"CS17", "CS18"})
CS_TO_DAY: dict[str, float] = {
    cs: sum(HESTA_SECTION_DAYS[section] for section in sections) / len(sections)
    for cs, sections in HUMAN_ALL_SECTIONS.items()
}
GEO_SERIES = "GSE326326"
GEO_BRIEF_URL = (
    f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={GEO_SERIES}&targ=gsm&form=text&view=brief"
)

HUMAN = _species_config(
    panel_key="human",
    species="human",
    data_key="hesta",
    result_dir=HUMAN_ONLY_RESULT_DIR,
    annotation_col="celltype",
    CS_TO_DAY=CS_TO_DAY,
    KEPT_TIMEPOINTS=list(CS_TO_DAY),
    ALL_SECTIONS=HUMAN_ALL_SECTIONS,
    REPRESENTATIVE_SECTIONS=HUMAN_REPRESENTATIVE_SECTIONS,
    SECTION_TO_DAY=HESTA_SECTION_DAYS,
    EMBRYO_TO_SEX=HESTA_EMBRYO_SEX,
    SINGLE_EMBRYO_STAGES=HESTA_SINGLE_EMBRYO_STAGES,
    DAY_RANGE=(min(HESTA_SECTION_DAYS.values()), max(HESTA_SECTION_DAYS.values())),
    GEO_SERIES=GEO_SERIES,
    GEO_BRIEF_URL=GEO_BRIEF_URL,
)

SPECIES_CONFIGS = {"mouse": MOUSE, "human": HUMAN}


def ensure_output_dirs() -> None:
    """Create workflow-owned embryo_xspecies result directories."""
    ensure_directories(
        RESULT_DIR,
        COMPARISON_RESULT_DIR,
        MOUSE_ONLY_RESULT_DIR,
        HUMAN_ONLY_RESULT_DIR,
    )
    for cfg in SPECIES_CONFIGS.values():
        ensure_species_output_dirs(cfg)
