"""Static configuration for the Visium human glioma IDH-status analysis."""

from __future__ import annotations

from scripts.paper_utils import PAPER_COLORS, PROJECT_ROOT

ANALYSIS_SLUG = "glioma_idh"
DATA_ROOT = PROJECT_ROOT / "data" / "visium_glioma_idh_status"

RESULT_DIR = PROJECT_ROOT / "results" / ANALYSIS_SLUG
FIGURE_DIR = PROJECT_ROOT / "figures" / ANALYSIS_SLUG

# Greenwald, Alissa C., et al. "Integrative spatial analysis reveals a multi-layered organization of glioblastoma." Cell 187.10 (2024): 2485-2501.
# Data to be downloaded from Zenodo
ZENODO_BASE_URL = "https://zenodo.org/records/12624860/files/{}"

N_GENES = 7000

# Use the same QC thresholds as in the original publication.
SPOT_MIN_COUNTS = 1000
SPOT_MAX_PCT_MITO = 20.0
SAMPLE_MAX_QC_EXCLUDED_FRAC = 0.50
MIN_FRAC_PER_SAMPLE = 0.02
MIN_SAMPLES_FRAC = 0.50
MIN_SAMPLES_PER_GROUP = 3
N_RADIAL_BINS = 30
N_LABEL_PERMUTATIONS = 50
FDR_ALPHA = 0.05

REAL_TESTS_FILE = RESULT_DIR / "idh_status_tests.csv"
PERMUTATION_NULL_FILE = RESULT_DIR / "idh_status_permutation_null.csv"
SPECTRA_FILE = RESULT_DIR / "idh_status_spectra.npz"
SPATIAL_DIAGNOSTICS_FILE = RESULT_DIR / "idh_status_spatial_diagnostics.csv"
MAIN_FIGURE_PDF = FIGURE_DIR / "idh_main.pdf"
SUPPLEMENTARY_DIAGNOSTICS_PDF = FIGURE_DIR / "idh_supp.pdf"
ENRICHMENT_TERMS_FILE = RESULT_DIR / "idh_status_clusterprofiler_enrichment.csv"
META_PROGRAM_GENE_MAP_FILE = RESULT_DIR / "idh_status_df_norm_top100_metaprogram_map.csv"

# Main spatial-profile genes ordered from established programs to novel findings.
FIGURE_GENES: tuple[str, ...] = ("NDRG1", "C3", "S100A10", "FN1")

# Canonical IDH-linked metabolism controls shown in the diagnostics supplement.
SUPPLEMENTARY_METABOLISM_GENES: tuple[str, ...] = ("BCAT1", "LDHA")

# Literature-prioritized meta-programs, representative top DF-norm genes, and
# compact DF-norm-preferred GO/KEGG terms retained in the main figure. Terms
# are stored as (source, stable identifier, display name).
META_PROGRAMS = (
    {
        "key": "metabolism_redox",
        "label": "Metabolism / redox",
        "short_label": "Metab",
        "color": PAPER_COLORS["ochre"],
        "genes": ("BCAT1", "MT3", "GAPDH", "TPI1", "LPL"),
        "terms": (
            ("KEGG", "hsa01230", "Biosynthesis of amino acids"),
            ("GO:BP", "GO:0032787", "monocarboxylic acid metabolic process"),
            ("GO:BP", "GO:1905954", "positive regulation of lipid localization"),
        ),
    },
    {
        "key": "hypoxia_vascular",
        "label": "Hypoxia / vascular",
        "short_label": "Hypoxia",
        "color": PAPER_COLORS["purple"],
        "genes": ("NDRG1", "VEGFA", "HILPDA", "SLC2A1", "DCN"),
        "terms": (
            ("GO:BP", "GO:0071456", "cellular response to hypoxia"),
            ("GO:BP", "GO:0001568", "blood vessel development"),
            ("GO:BP", "GO:0070482", "response to oxygen levels"),
        ),
    },
    {
        "key": "immune_complement",
        "label": "Immune / complement",
        "short_label": "Immune",
        "color": PAPER_COLORS["green"],
        "genes": ("C3", "C1QA", "C1QB", "LGALS3", "C1QC"),
        "terms": (
            ("GO:BP", "GO:0006959", "humoral immune response"),
            ("GO:BP", "GO:0006956", "complement activation"),
            ("GO:BP", "GO:0098883", "synapse pruning"),
        ),
    },
    {
        "key": "plasmin_coag",
        "label": "Plasminogen / coagulation",
        "short_label": "Coag",
        "color": PAPER_COLORS["pink"],
        "genes": ("S100A10", "ANXA2", "ENO1", "PGK1", "A2M"),
        "terms": (
            ("KEGG", "hsa04610", "Complement and coagulation cascades"),
            ("GO:BP", "GO:0031639", "plasminogen activation"),
            ("GO:BP", "GO:0010755", "regulation of plasminogen activation"),
        ),
    },
    {
        "key": "ecm_integrin",
        "label": "ECM / integrin",
        "short_label": "ECM",
        "color": PAPER_COLORS["charcoal"],
        "genes": ("FN1", "COL1A1", "COL6A1", "COL1A2", "TNC"),
        "terms": (
            ("KEGG", "hsa04512", "ECM-receptor interaction"),
            ("KEGG", "hsa04518", "Integrin signaling"),
            ("GO:BP", "GO:0009612", "response to mechanical stimulus"),
        ),
    },
)

# Conservative pathway-name rules used to assign enriched terms to programs.
META_PROGRAM_PATHWAY_KEYWORDS = {
    "ecm_integrin": (
        "ecm-receptor",
        "extracellular matrix",
        "integrin",
        "focal adhesion",
        "collagen",
        "response to mechanical stimulus",
        "skeletal system development",
        "muscle structure development",
        "connective tissue development",
        "regulation of cell migration",
        "cell-matrix adhesion",
    ),
    "plasmin_coag": (
        "plasminogen",
        "coagulation",
        "fibrinolysis",
        "hemostasis",
        "platelet activation",
    ),
    "immune_complement": (
        "immune",
        "complement",
        "interferon",
        "cytokine",
        "inflammatory",
        "granulocyte",
        "leukocyte",
        "phagocytosis",
        "chemotaxis",
    ),
    "hypoxia_vascular": (
        "hypoxia",
        "oxygen level",
        "vascular endothelial growth factor",
        "blood vessel",
        "angiogenesis",
        "vasculature",
        "hif-1",
    ),
    "metabolism_redox": (
        "glycolysis",
        "glucose catabolic",
        "amino acid",
        "hydrogen peroxide",
        "reactive oxygen species",
        "redox",
        "lipid",
        "fatty acid",
        "carbohydrate",
        "monocarboxylic acid",
        "atp metabolic",
        "pyridine nucleotide",
        "nicotinamide nucleotide",
    ),
}

GROUP_ORDER = ("IDH-mut", "IDH-wt")
GROUP_LABELS = {
    "IDH-mut": "IDH-mutant",
    "IDH-wt": "IDH-wildtype",
}
GROUP_COLORS = {
    "IDH-mut": PAPER_COLORS["cool"],
    "IDH-wt": PAPER_COLORS["warm"],
}

IDH_WILDTYPE_SAMPLE_IDS: tuple[str, ...] = (
    "GBM_MGH258",
    "GBM_ZH1007inf",
    "GBM_ZH1007nec",
    "GBM_ZH1019T1",
    "GBM_ZH8811Abulk",
    "GBM_ZH8811Bbulk",
    "GBM_ZH8812bulk",
    "GBM_ZH881_inf",
    "GBM_ZH881T1",
    "GBM_ZH916bulk",
    "GBM_ZH916inf",
    "GBM_ZH916T1",
)

# IDHm_BWH35A.tar.gz on Zenodo is a 45-byte placeholder and is skipped.
IDH_MUTANT_SAMPLE_IDS: tuple[str, ...] = (
    "IDHm_BWH23O",
    "IDHm_BWH24A",
    "IDHm_BWH25A",
    "IDHm_BWH28A",
    "IDHm_BWH35O",
    "IDHm_MGH259O",
)

SAMPLES: tuple[tuple[str, str], ...] = tuple(
    [(sample_id, GROUP_ORDER[0]) for sample_id in IDH_MUTANT_SAMPLE_IDS]
    + [(sample_id, GROUP_ORDER[1]) for sample_id in IDH_WILDTYPE_SAMPLE_IDS]
)
