"""Configuration for embryo spatial-pattern spectral embeddings.

This module owns both shared analysis constants and the two species-specific
configuration objects:

- ``MOUSE`` — Stereo-seq MOSTA mouse embryo sections.
- ``HUMAN`` — Stereo-seq HESTA human embryo sections.

The adopted radial-spectrum embedding recipe is the winner of the richness
diagnosis (``figure_supp_richness``): geometric, low-frequency-dense bins at
``N_RADIAL_BINS=80`` (≈41 retained on the shared-supported grid), **L1-log**
shape scaling, **equal** frequency weighting, per-bin z-score, and ``N_PC=5``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

# --- repo-relative roots -----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "embryo_xspecies"
RESULT_DIR = PROJECT_ROOT / "results" / "embryo_xspecies"
FIGURE_DIR = PROJECT_ROOT / "figures" / "embryo_xspecies"
COMPARISON_RESULT_DIR = RESULT_DIR / "comparison"
MOUSE_ONLY_RESULT_DIR = RESULT_DIR / "mouse_only"
HUMAN_ONLY_RESULT_DIR = RESULT_DIR / "human_only"
COMPARISON_FIGURE_DIR = FIGURE_DIR / "comparison"
MOUSE_ONLY_FIGURE_DIR = FIGURE_DIR / "mouse_only"
HUMAN_ONLY_FIGURE_DIR = FIGURE_DIR / "human_only"

# --- spatial/spectral grid shared across species ----------------------------
# Stereo-seq bin50 means 50 x 50 DNBs. The DNB pitch is 0.5 µm, so one bin50
# observation spans 25 µm on each side.
STEREO_DNB_PITCH_UM = 0.5
BIN_SIZE_DNB = 50
BIN_SIZE_UM = STEREO_DNB_PITCH_UM * BIN_SIZE_DNB
N_RADIAL_BINS = 80
PRESENCE_THRESHOLD = 0.0

# --- adopted radial-spectrum embedding recipe --------------------------------
FREQ_SPACING = "geom"
EMBED_SCALING = "L1-log"
EMBED_WEIGHTING = "equal"
N_PC = 5

# --- spatial-domain annotation ----------------------------------------------
# Generated comparator feature prefix, not an input adata column.
DOMAIN_CHANNEL_PREFIX = "__domain__"


def _species_config(
    *,
    panel_key: str,
    species: str,
    data_key: str,
    result_dir: Path,
    figure_dir: Path,
    min_extent_um: float,
    annotation_col: str,
    **extra,
) -> SimpleNamespace:
    data_root = DATA_ROOT / data_key
    cfg = SimpleNamespace(
        PANEL_KEY=panel_key,
        SPECIES=species,
        DATA_ROOT=data_root,
        RAW_DIR=data_root / "raw",
        SDATA_DIR=data_root / "sdata",
        META_CSV=data_root / "metadata.csv",
        RESULT_DIR=result_dir,
        FIGURE_DIR=figure_dir,
        GEOMETRY_CSV=result_dir / "section_geometry.csv",
        MIN_EXTENT_UM=min_extent_um,
        ANNOTATION_COL=annotation_col,
    )
    cfg.__dict__.update(extra)
    return cfg


def ensure_species_output_dirs(cfg: SimpleNamespace) -> None:
    """Create one species' raw, prepared, result, and figure directories."""
    for d in (cfg.RAW_DIR, cfg.SDATA_DIR, cfg.RESULT_DIR, cfg.FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


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
ALL_TIMEPOINTS: list[str] = list(STAGE_TO_DAY)
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
    figure_dir=MOUSE_ONLY_FIGURE_DIR,
    min_extent_um=3400.0,
    annotation_col="annotation",
    STAGE_TO_DAY=STAGE_TO_DAY,
    ALL_TIMEPOINTS=ALL_TIMEPOINTS,
    KEPT_TIMEPOINTS=MOUSE_KEPT_TIMEPOINTS,
    MOSTA_BASE_URL=MOSTA_BASE_URL,
    ALL_SECTIONS=ALL_SECTIONS,
    REPRESENTATIVE_SECTIONS=REPRESENTATIVE_SECTIONS,
)

# --- human: Stereo-seq HESTA -------------------------------------------------
HUMAN_ALL_SECTIONS: dict[str, list[str]] = {
    "CS12-13": ["CS12-13_E1S1", "CS12-13_E1S2", "CS12-13_E1S3"],
    "CS14-15": ["CS14-15_E1S1", "CS14-15_E1S2", "CS14-15_E1S3"],
    "CS17": ["CS17_E1S1", "CS17_E1S2", "CS17_E1S3"],
    "CS18": ["CS18_E1S1", "CS18_E1S2", "CS18_E1S3"],
    "CS19": ["CS19_E1S1", "CS19_E1S2", "CS19_E1S3"],
    "CS20": ["CS20_E1S1", "CS20_E1S2", "CS20_E1S3"],
    "CS23": ["CS23_E1S1", "CS23_E1S2", "CS23_E1S3"],
}
HUMAN_REPRESENTATIVE_SECTIONS: dict[str, list[str]] = {
    cs: [sections[0]] for cs, sections in HUMAN_ALL_SECTIONS.items()
}

# Days post-conception for the configured HESTA sections, from Pan et al. 2026
# Supplementary Table 1a ("Summary of Spatiotemporal transcriptomic data").
HESTA_SECTION_DAYS: dict[str, float] = {
    "CS12-13_E1S1": 29.0,
    "CS12-13_E1S2": 29.0,
    "CS12-13_E1S3": 29.0,
    "CS14-15_E1S1": 31.0,
    "CS14-15_E1S2": 31.0,
    "CS14-15_E1S3": 31.0,
    "CS17_E1S1": 41.0,
    "CS17_E1S2": 41.0,
    "CS17_E1S3": 41.0,
    "CS18_E1S1": 47.0,
    "CS18_E1S2": 47.0,
    "CS18_E1S3": 47.0,
    "CS19_E1S1": 44.0,
    "CS19_E1S2": 44.0,
    "CS19_E1S3": 44.0,
    "CS20_E1S1": 48.0,
    "CS20_E1S2": 48.0,
    "CS20_E1S3": 48.0,
    "CS23_E1S1": 59.0,
    "CS23_E1S2": 59.0,
    "CS23_E1S3": 59.0,
}
CS_TO_DAY: dict[str, float] = {
    cs: HESTA_SECTION_DAYS[sections[0]] for cs, sections in HUMAN_ALL_SECTIONS.items()
}
ALL_CARNEGIE_STAGES: list[str] = list(CS_TO_DAY)
GEO_SERIES = "GSE326326"
GEO_BRIEF_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    f"?acc={GEO_SERIES}&targ=gsm&form=text&view=brief"
)

HUMAN = _species_config(
    panel_key="human",
    species="human",
    data_key="hesta",
    result_dir=HUMAN_ONLY_RESULT_DIR,
    figure_dir=HUMAN_ONLY_FIGURE_DIR,
    min_extent_um=4000.0,
    annotation_col="celltype",
    CS_TO_DAY=CS_TO_DAY,
    ALL_CARNEGIE_STAGES=ALL_CARNEGIE_STAGES,
    KEPT_TIMEPOINTS=list(CS_TO_DAY),
    ALL_SECTIONS=HUMAN_ALL_SECTIONS,
    REPRESENTATIVE_SECTIONS=HUMAN_REPRESENTATIVE_SECTIONS,
    SECTION_TO_DAY=HESTA_SECTION_DAYS,
    DAY_RANGE=(min(HESTA_SECTION_DAYS.values()), max(HESTA_SECTION_DAYS.values())),
    GEO_SERIES=GEO_SERIES,
    GEO_BRIEF_URL=GEO_BRIEF_URL,
)

SPECIES_CONFIGS = {"mouse": MOUSE, "human": HUMAN}


def ensure_output_dirs() -> None:
    """Create all embryo_xspecies output directories."""
    for d in (
        RESULT_DIR,
        FIGURE_DIR,
        COMPARISON_RESULT_DIR,
        MOUSE_ONLY_RESULT_DIR,
        HUMAN_ONLY_RESULT_DIR,
        COMPARISON_FIGURE_DIR,
        MOUSE_ONLY_FIGURE_DIR,
        HUMAN_ONLY_FIGURE_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
