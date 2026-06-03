from __future__ import annotations

from pathlib import Path

HOME = Path.home()
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Keep data/results/figures separate so generated outputs stay out of source.
DATA_ROOT = PROJECT_ROOT / "data" / "atera_ffpe_human_brca"
ATERA_DATA_DIR = DATA_ROOT / "atera"
TCGA_DATA_DIR = DATA_ROOT / "tcga_brca_ankrd30a"
PACBIO_DATA_DIR = DATA_ROOT / "pacbio_brca_lrseq"
ANKRD30A_TRPS1_PROSEQ_DATA_DIR = DATA_ROOT / "ankrd30a_trps1_proseq"

# Script-owned caches are flat under the Atera result root unless they are
# large raw/source files, which live under DATA_ROOT.
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "atera_ffpe_human_brca"
RESULT_DIR = PROJECT_ROOT / "results" / "atera_ffpe_human_brca"
FIGURE_DIR = PROJECT_ROOT / "figures" / "atera_ffpe_human_brca"
TOP20_FIGURE_DIR = FIGURE_DIR / "top20_svp_probe_transcript_panels"
TOP20_RESULT_DIR = RESULT_DIR / "top20_svp_probe_transcript_panels"
SVP_PROBE_RESULT_DIR = RESULT_DIR / "svp_probe_transcript_cache"

SDATA_ZARR = ATERA_DATA_DIR / "sdata_cw_nomorph.filtered.zarr"
SDATA_ZARR_ZIP = ATERA_DATA_DIR / "sdata_cw_nomorph.filtered.zarr.zip"
PROBE_CSV = ATERA_DATA_DIR / "atera__human_whole_transcriptome_v1__probe_info.csv"
ANNOTATION_CSV = ATERA_DATA_DIR / "WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv"

# Reference annotation inputs used for probe-to-transcript interpretation.
REFERENCE_ROOT = HOME / "reference"
HG38_REFERENCE_DIR = REFERENCE_ROOT / "hg38"
GENCODE_GTF = HG38_REFERENCE_DIR / "gencode.v49.annotation.sorted.gtf.gz"
PACBIO_GFF = (
    PACBIO_DATA_DIR / "PacBio_Breast_Cancer_all_QC_pass_transcripts_unique.gff.gz"
)
SQANTI_TXT = PACBIO_DATA_DIR / "Sqanti_annotation_QC_pass_transcripts_unique.txt.gz"
PACBIO_FL_COUNTS_ZIP = PACBIO_DATA_DIR / "BRCA_superPBID_counts.tsv.zip"
PACBIO_FL_COUNTS_TSV = PACBIO_DATA_DIR / "BRCA_superPBID_counts.tsv"
HG38_FASTA = HG38_REFERENCE_DIR / "hg38.genome.fa"

TCGA_BAM_SLICE_DIR = TCGA_DATA_DIR / "bam_slices"
TCGA_BAM_MANIFEST_TSV = TCGA_DATA_DIR / "tcga_brca_rnaseq_bam_manifest_with_samples.tsv"
TCGA_CBIO_CLINICAL_SAMPLE_TXT = (
    TCGA_DATA_DIR / "brca_tcga_pub2015_data_clinical_sample.txt"
)
TCGA_CBIO_RSEM_TXT = TCGA_DATA_DIR / "brca_tcga_pub2015_data_mrna_seq_v2_rsem.txt"
TCGA_GDC_CLINICAL_TSV = TCGA_DATA_DIR / "tcga_brca_gdc_clinical.tsv"
TCGA_GDC_CLINICAL_RAW_JSON = TCGA_DATA_DIR / "tcga_brca_gdc_cases_clinical.raw.json"
TCGA_GDC_STATUS_JSON = TCGA_DATA_DIR / "gdc_status.json"

# TCGA analysis tables are generated from the raw source files above and are
# small enough to live with other workflow results.
TCGA_ANALYSIS_DIR = RESULT_DIR / "tcga_brca_ankrd30a"
TCGA_METRIC_TABLE_TSV = (
    TCGA_ANALYSIS_DIR / "tcga_brca_ankrd30a_pb_probe_metric_table.tsv"
)
TCGA_ER_POSITIVE_CORRELATION_CSV = (
    RESULT_DIR / "tcga_er_positive_regulator_correlations.csv"
)

# ReMap is a shared reference cache under ~/reference because it can be reused
# by Atera DU and other human workflows.
REMAP_CACHE_DIR = REFERENCE_ROOT / "ReMap2022"
REMAP_NR_HG38_BED = REMAP_CACHE_DIR / "remap2022_nr_macs2_hg38_v1_0.bed.gz"
REMAP_TF_TARGETS_HG38 = REMAP_CACHE_DIR / "remap2022_tf_targets_hg38.csv"
REMAP_NR_HG38_URLS = (
    "https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/remap2022_nr_macs2_hg38_v1_0.bed.gz",
    "https://zenodo.org/records/10527088/files/remap2022_nr_macs2_hg38_v1_0.bed.gz?download=1",
)
POSTAR3_RBP_SOURCE = REFERENCE_ROOT / "POSTAR3" / "human.rbp_with_clip.txt"

BRCA_LRSEQ_BASE_URL = (
    "https://raw.githubusercontent.com/TheJacksonLaboratory/BRCA-LRseq-pipeline/main"
)
PACBIO_GFF_URL = f"{BRCA_LRSEQ_BASE_URL}/data/QC_pass/{PACBIO_GFF.name}"
SQANTI_TXT_URL = f"{BRCA_LRSEQ_BASE_URL}/data/QC_pass/{SQANTI_TXT.name}"
PACBIO_FL_COUNTS_URL = (
    f"{BRCA_LRSEQ_BASE_URL}/data/isoforms_before_QC/{PACBIO_FL_COUNTS_ZIP.name}"
)

TABLE_CODEWORD = "table_codeword"
TABLE_GENE = "table"
LAYER = "counts"
SPATIAL_KEY = "spatial"
GENE_COL_CODEWORD = "gene_symbol"
GENE_COL_GENE_TABLE = None

# Canonical statistical settings shared by SV and DU; workflow summaries record
# these values to prevent stale noncanonical caches from being reused silently.
FDR_THRESHOLD = 0.01
MIN_COUNTS = 100
MIN_BIN_PCT = 0.05
MIN_COMPONENT_SIZE = 100
K_NEIGHBORS = 4
RHO = 0.99
SV_METHOD = "hsic-ir"
DU_METHOD = "hsic-gp"
DU_GPR_BACKEND = "nufft"
DU_RESIDUALIZE = "cov_only"
DU_LML_APPROX_RANK = 32

# Primary workflow-owned result files consumed by visualization.ipynb.
SV_CSV = RESULT_DIR / "segmented_splisosmnp_svp_hsic_ir.csv"
SV_METRICS_CSV = RESULT_DIR / "segmented_splisosmnp_svp_metrics.csv"
SV_SUMMARY_JSON = RESULT_DIR / "segmented_splisosmnp_svp_summary.json"
DU_CSV = RESULT_DIR / "du_regulator_segmented_svp_hsic_gp_nufft.csv"
DU_METRICS_CSV = RESULT_DIR / "du_regulator_segmented_svp_hsic_gp_nufft_metrics.csv"
DU_REGULATOR_STATUS_CSV = RESULT_DIR / "du_regulator_status.csv"
DU_SUMMARY_JSON = RESULT_DIR / "du_regulator_segmented_svp_summary.json"
FINAL_FIGURE_PDF = FIGURE_DIR / "atera_ankrd30a.pdf"
DU_TCGA_FIGURE_PDF = FIGURE_DIR / "atera_ankrd30a_du_tcga.pdf"
ANKRD30A_VISUALIZATION_CACHE_DIR = RESULT_DIR / "ankrd30a_visualization_cache"
ANKRD30A_TRANSCRIPT_ANNOTATIONS_CSV = (
    ANKRD30A_VISUALIZATION_CACHE_DIR / "ankrd30a_transcript_annotations.csv"
)
ANKRD30A_PROBE_MAPPINGS_CSV = (
    ANKRD30A_VISUALIZATION_CACHE_DIR / "ankrd30a_probe_mappings.csv"
)
ANKRD30A_REGULATOR_PRIORITY_CSV = (
    ANKRD30A_VISUALIZATION_CACHE_DIR / "ankrd30a_regulator_priority.csv"
)

# PRO-seq results of TRPS1 knockdown in T47D cells from GSE236172
ANKRD30A_TRPS1_PROSEQ_RESULT_DIR = RESULT_DIR / "ankrd30a_trps1_proseq"
ANKRD30A_TRPS1_PROSEQ_SAMPLE_MANIFEST_CSV = (
    ANKRD30A_TRPS1_PROSEQ_RESULT_DIR / "gse236172_sample_manifest.csv"
)
ANKRD30A_TRPS1_PROSEQ_SIGNAL_WINDOWS_CSV = (
    ANKRD30A_TRPS1_PROSEQ_RESULT_DIR / "ankrd30a_proseq_signal_windows.csv"
)
ANKRD30A_TRPS1_PROSEQ_SIGNAL_CSV = (
    ANKRD30A_TRPS1_PROSEQ_RESULT_DIR / "ankrd30a_proseq_signal_changes.csv"
)

# POSTAR3 human CLIP-backed RBP symbols copied from
# ~/reference/POSTAR3/human.rbp_with_clip.txt on 2026-05-15.
RBP_GENES = {
    "AARS",
    "AATF",
    "ABCF1",
    "ACI",
    "AGGF1",
    "AGO2",
    "AKAP1",
    "AKAP8L",
    "ALKBH1",
    "ALKBH5",
    "ALYREF",
    "APOBEC3C",
    "AQR",
    "ATXN2",
    "BCCIP",
    "BCLAF1",
    "BUD13",
    "CAPRIN1",
    "CDC40",
    "CELF2",
    "CHTOP",
    "CNBP",
    "CPEB4",
    "CPSF1",
    "CPSF2",
    "CPSF3",
    "CPSF4",
    "CPSF6",
    "CPSF7",
    "CSTF2",
    "CSTF2T",
    "DDX21",
    "DDX24",
    "DDX3",
    "DDX3X",
    "DDX42",
    "DDX51",
    "DDX52",
    "DDX55",
    "DDX59",
    "DDX6",
    "DGCR8",
    "DHX30",
    "DICER1",
    "DIS3L2",
    "DKC1",
    "DROSHA",
    "EFTUD2",
    "EIF3A",
    "EIF3B",
    "EIF3D",
    "EIF3G",
    "EIF3H",
    "EIF4A3",
    "EIF4G2",
    "ELAVL1",
    "EWSR1",
    "EXOSC5",
    "EZH2",
    "FAM120A",
    "FASTKD2",
    "FBL",
    "FIP1L1",
    "FKBP4",
    "FMR1",
    "FTO",
    "FUBP3",
    "FUS",
    "FXR1",
    "FXR2",
    "G3BP1",
    "GEMIN5",
    "GNL3",
    "GPKOW",
    "GRSF1",
    "GRWD1",
    "GTF2F1",
    "HLTF",
    "HNRNPA1",
    "HNRNPA2B1",
    "HNRNPC",
    "HNRNPD",
    "HNRNPF",
    "HNRNPH",
    "HNRNPH1",
    "HNRNPK",
    "HNRNPL",
    "HNRNPM",
    "HNRNPU",
    "HNRNPUL1",
    "IGF2BP1",
    "IGF2BP2",
    "IGF2BP3",
    "ILF3",
    "IMP3",
    "KHDRBS1",
    "KHSRP",
    "LARP4",
    "LARP7",
    "LIN28A",
    "LIN28B",
    "LSM11",
    "MATR3",
    "MBNL2",
    "METAP2",
    "METTL14",
    "METTL3",
    "MKRN1",
    "MOV10",
    "MTPAP",
    "NCBP2",
    "NCBP3",
    "NIP7",
    "NIPBL",
    "NKRF",
    "NOL12",
    "NOLC1",
    "NONO",
    "NOP56",
    "NOP58",
    "NPM1",
    "NSUN2",
    "NUDT21",
    "NXF1",
    "PABPC4",
    "PABPN1",
    "PCBP1",
    "PCBP2",
    "PHF6",
    "POLR2G",
    "PPIG",
    "PPIL4",
    "PRKRA",
    "PRPF4",
    "PRPF8",
    "PTBP1",
    "PUM1",
    "PUM2",
    "PUS1",
    "QKI",
    "RBFOX1",
    "RBFOX2",
    "RBM10",
    "RBM15",
    "RBM15B",
    "RBM22",
    "RBM47",
    "RBM5",
    "RBPMS",
    "RBP_occupancy",
    "RPS11",
    "RPS3",
    "RTCB",
    "SAFB",
    "SAFB2",
    "SBDS",
    "SDAD1",
    "SDOS",
    "SERBP1",
    "SF3A3",
    "SF3B1",
    "SF3B4",
    "SFPQ",
    "SLBP",
    "SLTM",
    "SMN",
    "SMNDC1",
    "SND1",
    "SRRM4",
    "SRSF1",
    "SRSF10",
    "SRSF3",
    "SRSF7",
    "SRSF9",
    "SSB",
    "STAU1",
    "STAU2",
    "SUB1",
    "SUGP2",
    "SUPV3L1",
    "TAF15",
    "TARBP2",
    "TARDBP",
    "TBRG4",
    "TDP43",
    "TIA1",
    "TIAL1",
    "TNRC6A",
    "TNRC6B",
    "TNRC6C",
    "TRA2A",
    "TROVE2",
    "U2AF1",
    "U2AF2",
    "U2AF65",
    "UCHL5",
    "UNR",
    "UPF1",
    "UTP18",
    "UTP3",
    "WDR3",
    "WDR33",
    "WDR43",
    "WRN",
    "WTAP",
    "XPO5",
    "XRCC6",
    "XRN2",
    "YBX3",
    "YTHDC1",
    "YTHDC2",
    "YTHDF1",
    "YTHDF2",
    "YTHDF3",
    "YWHAG",
    "ZC3H11A",
    "ZC3H7B",
    "ZC3H8",
    "ZNF622",
    "ZNF800",
    "ZRANB2",
}
