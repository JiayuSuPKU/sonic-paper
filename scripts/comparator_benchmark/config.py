"""Static configuration for the comparator benchmark scripts.

Mirrors the structure of ``scripts/visiumhd_mouse_cbs/config.py``: top-level
path constants first, then magic thresholds, then a per-panel dataclass with
all dataset-specific paths / GEO accessions / curated gene lists, and finally
panel-specific download metadata (sample lists, GSM dictionaries) used by
the downloader scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Roots — mirror visiumhd_mouse_cbs/config.py:7-14
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

RESULT_DIR = PROJECT_ROOT / "results" / "comparator_benchmark"
FIGURE_DIR = PROJECT_ROOT / "figures" / "comparator_benchmark"

NUMBA_CACHE_DIR = Path(
    os.environ.get("NUMBA_CACHE_DIR", PROJECT_ROOT / ".cache" / "numba")
)
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

# ---------------------------------------------------------------------------
# Sweep / test thresholds
# ---------------------------------------------------------------------------

PRESENCE_THRESHOLD = 0.05
MAX_GENES = 3000
N_RADIAL_BINS = 30
EPS = 1e-12  # numerical floor for log/divide

# ---------------------------------------------------------------------------
# Comparator benchmark workflow settings and cache names
# ---------------------------------------------------------------------------

TEST_LABELS: tuple[str, ...] = ("DE", "DF", "DF_norm")
SPATIAL_TESTS: tuple[tuple[str, bool], ...] = (("DF", False), ("DF_norm", True))
RADIAL_BIN_DIAGNOSTIC_BINS: tuple[int, ...] = (8, 12, 16, 24, 30, 45)

AD_PIG_FULL_INPUTS: tuple[str, ...] = ("raw", "log1p")
AD_PIG_FULL_BINNINGS: tuple[str, ...] = ("linear", "log")
AD_PIG_FULL_FEATURE_MODES: tuple[str, ...] = ("radial", "2d")
AD_PIG_REFERENCE_INPUT = "raw"
AD_PIG_REFERENCE_BINNING = "linear"
AD_PIG_REFERENCE_FEATURE_MODE = "radial"
AD_PIG_REFERENCE_N_RADIAL_BINS = N_RADIAL_BINS
AD_PIG_FDR_ALPHA = 0.05
AD_PIG_ENRICHMENT_TOP_N = 100

AD_PIG_FULL_TESTS_FILE = "ad_pig_full_tests.csv"
AD_PIG_FULL_NULL_ECDF_FILE = "ad_pig_full_null_ecdf.csv"
AD_PIG_RADIAL_BIN_ROBUSTNESS_FILE = "ad_pig_radial_bin_robustness.csv"
AD_PIG_ANISOTROPY_FILE = "ad_pig_anisotropy.csv"
AD_PIG_SPECTRAL_REPLICATE_DISTANCE_FILE = "ad_pig_spectral_replicate_distance.csv"
AD_ENRICH_CONTRAST_FILE = "ad_pig_enrichment_contrasts.csv"

BRAIN_DEV_REFERENCE_INPUT = "raw"
BRAIN_DEV_REFERENCE_BINNING = "linear"
BRAIN_DEV_REFERENCE_FEATURE_MODE = "2d"
BRAIN_DEV_REFERENCE_N_RADIAL_BINS = N_RADIAL_BINS
BRAIN_DEV_FULL_INPUTS: tuple[str, ...] = ("raw", "log1p")
BRAIN_DEV_FULL_FEATURE_MODES: tuple[str, ...] = ("radial", "2d")
BRAIN_DEV_NULL_PERMUTATIONS = 50
BRAIN_DEV_TIMEPOINT_ORDER: tuple[str, ...] = ("P5", "P7", "P9", "P12", "P21", "P60")
BRAIN_DEV_ENRICHMENT_TOP_N = 100
BRAIN_DEV_CELL_TYPE_PROGRAM_TOP_N = 8
BRAIN_DEV_CELL_TYPE_CONTROL_TOP_N = 30
BRAIN_DEV_TESTS_FILE = "brain_dev_tests.csv"
BRAIN_DEV_NULL_ECDF_FILE = "brain_dev_null_ecdf.csv"
BRAIN_DEV_ANISOTROPY_FILE = "brain_dev_anisotropy.csv"
BRAIN_DEV_POSITIVE_CONTROLS_FILE = "brain_dev_positive_controls.csv"
BRAIN_DEV_ENRICH_CONTRAST_FILE = "brain_dev_enrichment_contrasts.csv"

CLUSTERPROFILER_RSCRIPT = PROJECT_ROOT / "scripts" / "run_clusterprofiler.R"

# ---------------------------------------------------------------------------
# Dataset and QC constants
# ---------------------------------------------------------------------------

# Stockholm Spatial Transcriptomics geometry (ad_pig only)
ST_SPOT_PITCH_UM = 200.0  # spot-to-spot distance, centre to centre

# AD/PIG probe QC from the original analysis notes. Three probes with design
# issues are removed; cross-reactive probes are assigned to the shared family
# symbols used for downstream ranking and enrichment.
AD_PIG_DROP_PROBES: tuple[str, ...] = ("H2-D1", "Cd63-ps", "RP23-269H21.1")
AD_PIG_RENAME_PROBES: dict[str, str] = {
    "C4a": "C4",
    "C4b": "C4",
    "Serpina3n": "Serpina3",
    "Lyz2": "Lyz",
}
AD_PIG_STALE_PROBE_NAMES = frozenset(AD_PIG_DROP_PROBES) | frozenset(
    AD_PIG_RENAME_PROBES
)

# 10x Visium geometry (brain_dev only)
BRAIN_DEV_SPOT_DIAMETER_UM = 55.0  # capture-spot diameter
BRAIN_DEV_MIN_GENE_SPOTS = 3
BRAIN_DEV_MIN_COUNTS_PER_SPOT = 3_000
BRAIN_DEV_MAX_COUNTS_PER_SPOT = 50_000

# ---------------------------------------------------------------------------
# Curated gene panels
# ---------------------------------------------------------------------------

# Full PIG (plaque-induced gene) module from Chen et al. *Cell* 2020 —
# genes co-induced around amyloid plaques in AppNL-G-F cortex (their
# Table S2 + Fig 4 module-A signature). Used as the ad_pig positive-
# control set. Includes the original 10-gene "core" subset (Cst7,
# Tyrobp, C4, Ctsd, Gfap, Trem2, Apoe, Hexb, Cd68, Itgax) plus the
# wider complement-cascade (C1q*, C4*), microglial-effector (Csf1r,
# Mpeg1, Plek), endo-lysosomal (Cts*, Hex*, Lgmn, Man2b1, Npc2),
# antigen-presentation (H2-K1, B2m), and additional Iba1/CD9-class
# markers from the broader module. Names below are post-QC symbols.
AD_PIG_POS_CONTROLS: tuple[str, ...] = (
    "Apoe",
    "Arpc1b",
    "Axl",
    "B2m",
    "C1qa",
    "C1qb",
    "C1qc",
    "C4",
    "Cd63",
    "Cd9",
    "Clu",
    "Csf1r",
    "Cst3",
    "Ctsa",
    "Ctsb",
    "Ctsd",
    "Ctsh",
    "Ctsl",
    "Ctss",
    "Ctsz",
    "Cx3cr1",
    "Cyba",
    "Fcer1g",
    "Fcgr3",
    "Fcrls",
    "Gfap",
    "Gns",
    "Gpx4",
    "Grn",
    "Gusb",
    "H2-K1",
    "Hexa",
    "Hexb",
    "Igfbp5",
    "Itgb5",
    "Itm2b",
    "Laptm5",
    "Lgals3bp",
    "Lgmn",
    "Ly86",
    "Lyz",
    "Man2b1",
    "Mpeg1",
    "Npc2",
    "Olfml3",
    "Plek",
    "Prdx6",
    "Rpl18a",
    "S100a6",
    "Serpina3",
    "Trem2",
    "Tyrobp",
    "Vsir",
)


# Local copy of the Cell 2026 brain-vascularization supplement used to define
# the brain-development positive controls.
BRAIN_DEV_TABLE_S4_CNMFS = (
    DATA_ROOT / "brain_dev" / "supplementary" / "table_s4_cnmf_programs.xlsx"
)

# Paper-derived positive-control groups for the continuous postnatal-time
# analysis (``brain_dev``). These map to Table S4 cNMF enrichment rows. The
# workflow selects the most enriched programs per group, then ranks their genes
# by mean cNMF score.
BRAIN_DEV_CELL_TYPE_CONTROL_GROUPS: dict[str, tuple[str, ...]] = {
    "immature_neuron": (
        "Immature neurons",
        "Cytoskeleton/axon scaffold",
    ),
    "mature_synaptic_neuron": (
        "Neurons",
        "Synaptic vesicle/exocytosis",
    ),
    "astrocyte": ("Astrocyte",),
    "oligodendrocyte": ("OPC", "Oligodendrocyte"),
    "vascular": ("Endothelial", "VSMC"),
}

BRAIN_DEV_CELL_TYPE_MANUAL_GENES: dict[str, tuple[str, ...]] = {
    "immature_neuron": (
        "Sox4",
        "Sox11",
        "Gap43",
        "Dcx",
        "Tubb3",
        "Kif1a",
        "Kif5a",
        "Mapt",
        "Nefh",
        "Nefl",
        "Nefm",
        "Stmn2",
        "Tubb2b",
        "Actb",
    ),
    "mature_synaptic_neuron": (
        "Camk2a",
        "Slc17a6",
        "Slc17a7",
        "Snap25",
        "Syt1",
        "Syt2",
        "Thy1",
        "Ppp1r1b",
        "Grin2c",
        "Gria1",
        "Gabra1",
        "Gabra2",
        "Gad1",
        "Gad2",
        "Nos1",
        "Pvalb",
        "Slc32a1",
        "Slc6a1",
        "Slc6a11",
        "Sst",
        "Vip",
        "Drd1",
        "Drd2",
        "Pantr1",
        "Calm2",
        "Ddc",
        "Nr4a2",
        "Pitx3",
        "Slc18a2",
        "Slc6a3",
        "Th",
        "Fev",
        "Gata2",
        "Gata3",
        "Slc6a4",
        "Tph2",
        "Dbh",
        "Slc6a2",
        "Chat",
        "Slc18a3",
        "Slc5a7",
        "Hdc",
        "Dlg4",
        "Slc1a2",
        "Slc1a3",
        "Stx1a",
        "Stx1b",
        "Syn1",
        "Syn2",
        "Syn3",
        "Syp",
        "Syt7",
        "Vamp1",
        "Vamp2",
    ),
    "astrocyte": (
        "Aldh1l1",
        "Apoe",
        "Aqp4",
        "Gja1",
        "Gfap",
        "S100b",
        "Slc1a2",
        "Slc1a3",
    ),
    "oligodendrocyte": (
        "Cspg4",
        "Gpr17",
        "Olig1",
        "Olig2",
        "Pdgfra",
        "Sox10",
        "Nfasc",
        "Sirt2",
        "Cldn11",
        "Cnp",
        "Gpr37",
        "Mag",
        "Mbp",
        "Mog",
        "Plp1",
    ),
    "vascular": (
        "Aplnr",
        "Cldn5",
        "Bsg",
        "Flt1",
        "Erg",
        "Slc38a5",
        "Fn1",
        "Eng",
        "Kdr",
        "Nos3",
        "Tfrc",
        "Pecam1",
        "Tek",
        "Vwf",
        "Rgs5",
        "Acta2",
        "Cald1",
        "Cnn1",
        "Myh11",
        "Myl9",
        "Tagln",
    ),
}

BRAIN_DEV_CELL_TYPE_LABELS: dict[str, str] = {
    "immature_neuron": "Immature neuron",
    "mature_synaptic_neuron": "Mature synaptic neuron",
    "astrocyte": "Astrocyte",
    "oligodendrocyte": "Oligodendrocyte",
    "vascular": "Vascular / mural",
}

# ---------------------------------------------------------------------------
# PanelConfig — per-panel paths + GEO accession + curated genes.
# Mirrors `DatasetConfig` in visiumhd_mouse_cbs/config.py:38-101.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelConfig:
    key: str
    data_root: Path  # e.g. data/ad_pig — has raw/, adata/, ...
    metadata_csv: Path  # row-per-sample metadata table the loader reads
    pos_controls: tuple[str, ...]
    geo_accession: str | None = None  # downloaders use these; benchmark ignores
    raw_url: str | None = None
    raw_min_bytes: int | None = None


PANELS: dict[str, PanelConfig] = {
    "ad_pig": PanelConfig(
        key="ad_pig",
        data_root=DATA_ROOT / "ad_pig",
        metadata_csv=DATA_ROOT / "ad_pig" / "metadata.csv",
        pos_controls=AD_PIG_POS_CONTROLS,
        geo_accession="GSE152506",
        raw_url=(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE152nnn/"
            "GSE152506/suppl/GSE152506_raw_counts.txt.gz"
        ),
        raw_min_bytes=200_000_000,  # ~258 MB expected
    ),
    # Continuous postnatal-time brain-development panel. ``build_sample_sets``
    # emits a single SampleSet with a continuous ``log_pn_day`` design over
    # all six timepoints (P5, P7, P9, P12, P21, P60) of the ANTERIOR plane
    # (n = 17). Positive controls are paper-derived Table S4 cNMF
    # cell-type/process genes.
    "brain_dev": PanelConfig(
        key="brain_dev",
        data_root=DATA_ROOT / "brain_dev",
        metadata_csv=DATA_ROOT / "brain_dev" / "metadata.csv",
        pos_controls=(),  # to be defined later from manual + Table S4 cNMF genes
        geo_accession="GSE313896",
        raw_url=(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE313nnn/"
            "GSE313896/suppl/GSE313896_RAW.tar"
        ),
        raw_min_bytes=2_000_000_000,  # ~2.4 GB expected
    ),
}


# ---------------------------------------------------------------------------
# AD/PIG (GSE152506) — Stockholm ST sample list
# (sample_title, genotype, age_months) parsed from GEO sample-level metadata.
# ---------------------------------------------------------------------------

AD_PIG_SAMPLES: tuple[tuple[str, str, int], ...] = (
    ("B02_D1", "WT", 3),
    ("B02_E1", "WT", 3),
    ("B03_C2", "WT", 3),
    ("B03_D2", "WT", 3),
    ("N02_C1", "TG", 3),
    ("N02_D1", "TG", 3),
    ("N03_C2", "TG", 3),
    ("N03_D2", "TG", 3),
    ("B07_C2", "WT", 6),
    ("N07_C1", "TG", 6),
    ("B06_E1", "WT", 12),
    ("N06_D2", "TG", 12),
    ("B04_D1", "WT", 18),
    ("B04_E1", "WT", 18),
    ("B05_D2", "WT", 18),
    ("B05_E2", "WT", 18),
    ("N04_D1", "TG", 18),
    ("N04_E1", "TG", 18),
    ("N05_C2", "TG", 18),
    ("N05_D2", "TG", 18),
)


# ---------------------------------------------------------------------------
# Postnatal brain (GSE313896) — GSM ID → canonical sample title
# ---------------------------------------------------------------------------

BRAIN_DEV_GSM_TITLES: dict[str, str] = {
    "GSM9377328": "P5_THALAMIC_PLANE_REP1",
    "GSM9377329": "P5_ANTERIOR_PLANE_REP1",
    "GSM9377330": "P7_THALAMIC_PLANE_REP1",
    "GSM9377331": "P7_ANTERIOR_PLANE_REP1",
    "GSM9377332": "P12_THALAMIC_PLANE_REP1",
    "GSM9377333": "P12_ANTERIOR_PLANE_REP1",
    "GSM9377334": "P9_THALAMIC_PLANE_REP1",
    "GSM9377335": "P9_ANTERIOR_PLANE_REP1",
    "GSM9377336": "P9_BRAINSTEM_PLANE_REP1",
    "GSM9377337": "P12_BRAINSTEM_PLANE_REP1",
    "GSM9377338": "P5_BRAINSTEM_PLANE_REP1",
    "GSM9377339": "P5_THALAMIC_PLANE_REP2",
    "GSM9377340": "P7_BRAINSTEM_PLANE_REP1",
    "GSM9377341": "P7_ANTERIOR_PLANE_REP2",
    "GSM9377342": "P9_THALAMIC_PLANE_REP2",
    "GSM9377343": "P9_BRAINSTEM_PLANE_REP2",
    "GSM9377344": "P9_ANTERIOR_PLANE_REP2",
    "GSM9377345": "P12_BRAINSTEM_PLANE_REP2",
    "GSM9377346": "P12_THALAMIC_PLANE_REP2",
    "GSM9377347": "P12_ANTERIOR_PLANE_REP2",
    "GSM9377348": "P60_ANTERIOR_PLANE_REP2",
    "GSM9377349": "P60_BRAINSTEM_PLANE_REP1",
    "GSM9377350": "P60_THALAMIC_PLANE_REP1",
    "GSM9377351": "P60_ANTERIOR_PLANE_REP1",
    "GSM9377352": "P5_BRAINSTEM_PLANE_REP2",
    "GSM9377353": "P5_ANTERIOR_PLANE_REP2",
    "GSM9377354": "P60_BRAINSTEM_PLANE_REP2",
    "GSM9377355": "P21_THALAMIC_PLANE_REP1",
    "GSM9377356": "P21_BRAINSTEM_PLANE_REP1",
    "GSM9377357": "P7_BRAINSTEM_PLANE_REP2",
    "GSM9377358": "P21_ANTERIOR_PLANE_REP2",
    "GSM9377359": "P21_THALAMIC_PLANE_REP2",
    "GSM9377360": "P21_BRAINSTEM_PLANE_REP2",
    "GSM9377361": "P21_ANTERIOR_PLANE_REP1",
    "GSM9377362": "P12_ANTERIOR_PLANE_REP3",
    "GSM9377363": "P5_ANTERIOR_PLANE_REP3",
    "GSM9377364": "P5_BRAINSTEM_PLANE_REP3",
    "GSM9377365": "P5_THALAMIC_PLANE_REP3",
    "GSM9377366": "P7_ANTERIOR_PLANE_REP3",
    "GSM9377367": "P7_THALAMIC_PLANE_REP2",
    "GSM9377368": "P12_BRAINSTEM_PLANE_REP3",
    "GSM9377369": "P12_THALAMIC_PLANE_REP3",
    "GSM9377370": "P60_ANTERIOR_PLANE_REP3",
    "GSM9377371": "P60_THALAMIC_PLANE_REP2",
    "GSM9377372": "P60_BRAINSTEM_PLANE_REP3",
    "GSM9377373": "P9_ANTERIOR_PLANE_REP3",
}
