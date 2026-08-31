from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.paper_utils import PROJECT_ROOT

HOME = Path.home()
DATA_ROOT = HOME / "Projects" / "SPLISOSM_paper" / "data"
REFERENCE_ROOT = HOME / "reference" / "mm39"

RESULT_DIR = PROJECT_ROOT / "results" / "splisosm_visiumhd_cbs"
FIGURE_DIR = PROJECT_ROOT / "figures" / "splisosm_visiumhd_cbs"
SV_METRICS_PATH = RESULT_DIR / "sv_runtime_memory_metrics.csv"

GTF_FILE = REFERENCE_ROOT / "gencode.vM33.annotation.gtf.gz"

FDR_THRESHOLD = 0.01
RHO = 0.99
NEIGHBOR_DEGREE = {"16um": 1, "8um": 4, "2um": 64}
SV_METHOD = "hsic-ir"
DU_METHOD = "hsic-gp"
RESOLUTION_UM = {"16um": 16, "8um": 8, "2um": 2}
RESOLUTION_TABLE = {
    "16um": "square_016um",
    "8um": "square_008um",
    "2um": "square_002um",
}
DU_COVARIATE_MIN_COUNTS = 10
DU_COVARIATE_MIN_BIN_PCT = 0.001

# Optional test subsets. Leave as None for genome-wide runs.
# If a subset run is desired, set explicit gene/RBP symbols here.
# SV_GENE_SUBSET: tuple[str, ...] | None = ("Mbp", "Gabbr1")
SV_GENE_SUBSET: tuple[str, ...] | None = None
DU_TARGET_GENE_SUBSET: tuple[str, ...] | None = None
DU_RBP_SUBSET: tuple[str, ...] | None = None
DU_TF_SUBSET: tuple[str, ...] | None = None

# Mouse TF ChIP evidence source. The ReMap BED is cached outside the project
# because the non-redundant mm39 catalogue is large.
TF_CHIP_SOURCE = "remap2022"
REMAP_CACHE_DIR = HOME / "reference" / "ReMap2022"
REMAP_NR_MM39_BED = REMAP_CACHE_DIR / "remap2022_nr_macs2_mm39_v1_0.bed.gz"
REMAP_TF_TARGETS = REMAP_CACHE_DIR / "remap2022_tf_targets_mm39.csv"
REMAP_INTERVAL_DIR = REMAP_CACHE_DIR / "intervals"
REMAP_NR_MM39_URLS = (
    "https://remap.univ-amu.fr/storage/remap2022/mm39/MACS2/remap2022_nr_macs2_mm39_v1_0.bed.gz",
    "https://zenodo.org/records/10527088/files/remap2022_nr_macs2_mm39_v1_0.bed.gz?download=1",
)


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    short_label: str
    feature_label: str
    sdata_zarr: Path
    group_iso_by: str | None
    gene_names: str | None
    min_counts: int
    min_bin_pct: float
    structure_kind: str
    bin_prefix: str
    bed12_file: Path | None = None
    gff_file: Path | None = None
    gtf_file: Path = GTF_FILE


DATASETS = {
    "trend_3prime": DatasetConfig(
        key="trend_3prime",
        label="Visium HD 3' TREND peak",
        short_label="3' TREND",
        feature_label="TREND peak",
        sdata_zarr=DATA_ROOT / "visiumhd_3p_mouse_cbs" / "sdata_peak.filtered.zarr",
        group_iso_by="gene_ids",
        gene_names="gene_ids",
        min_counts=10,
        min_bin_pct=0.001,
        structure_kind="trend_peak_bed12",
        bin_prefix="_",
        bed12_file=DATA_ROOT / "visiumhd_3p_mouse_cbs" / "sierra_peaks" / "peak.txt.bed12",
    ),
    "ffpe_probe": DatasetConfig(
        key="ffpe_probe",
        label="Visium HD FFPE probe",
        short_label="FFPE probe",
        feature_label="probe",
        sdata_zarr=DATA_ROOT / "visiumhd_ffpe_mouse_cbs" / "sdata_probe.filtered.zarr",
        group_iso_by="gene_ids",
        gene_names="gene_name",
        min_counts=10,
        min_bin_pct=0.001,
        structure_kind="probe_bed12",
        bin_prefix="Visium_HD_Mouse_Brain_",
        bed12_file=DATA_ROOT
        / "visiumhd_ffpe_mouse_cbs"
        / "Visium_Mouse_Transcriptome_Probe_Set_v2.1.0_GRCm39-2024-A.bed",
    ),
    "ont_tx": DatasetConfig(
        key="ont_tx",
        label="Visium HD ONT transcript",
        short_label="ONT transcript",
        feature_label="transcript",
        sdata_zarr=DATA_ROOT / "visiumhd_ont_mouse_cbs" / "sdata_tx.filtered.zarr",
        group_iso_by="gene_ids",
        gene_names="gene_name",
        min_counts=10,
        min_bin_pct=0.001,
        structure_kind="ont_gff",
        bin_prefix="_",
        gff_file=DATA_ROOT / "visiumhd_ont_mouse_cbs" / "SAMPLE_S1_L001.transcriptome.gff.gz",
    ),
}

# Mouse RBPs that have either CLIP-binding data from POSTAR3 or have motifs in CISBP-RNA.
RBP_GENES = {
    "A1cf",
    "Ago2",
    "Ankhd1",
    "Ankrd17",
    "Apc",
    "B020018G12Rik",
    "Cbp",
    "Celf",
    "Celf1",
    "Celf2",
    "Celf3",
    "Celf4",
    "Celf5",
    "Celf6",
    "Cirbp",
    "Cnot4",
    "Cpeb2",
    "Cpeb3",
    "Cpeb4",
    "Cpsf6",
    "Crebbp",
    "Csda",
    "Dazap1",
    "ENSMUSG00000021927",
    "ENSMUSG00000049235",
    "ENSMUSG00000056951",
    "ENSMUSG00000089986",
    "ENSMUSG00000090049",
    "Eif2s1",
    "Eif4b",
    "Elavl1",
    "Elavl2",
    "Elavl3",
    "Elavl4",
    "Enox1",
    "Enox2",
    "Esrp1",
    "Esrp2",
    "Ezh2",
    "Fam120a",
    "Fmr1",
    "Fus",
    "Fxr1",
    "Fxr2",
    "G3bp2",
    "Gm10110",
    "Gm12355",
    "Gm5145",
    "Gm7964",
    "Gm8991",
    "Hnrnpa1",
    "Hnrnpa2b1",
    "Hnrnpa3",
    "Hnrnpab",
    "Hnrnpc",
    "Hnrnpf",
    "Hnrnph1",
    "Hnrnph2",
    "Hnrnpk",
    "Hnrnpl",
    "Hnrnpr",
    "Hnrpdl",
    "Hnrpll",
    "Igf2bp1",
    "Igf2bp2",
    "Igf2bp3",
    "Khdrbs1",
    "Khdrbs2",
    "Khdrbs3",
    "Lin28a",
    "Lin28b",
    "Matr3",
    "Mbnl1",
    "Mbnl2",
    "Mbnl3",
    "Mex3b",
    "Mex3c",
    "Mex3d",
    "Msi1",
    "Msi2",
    "Ncl",
    "Nono",
    "Nova1",
    "Nova2",
    "Pabpc1",
    "Pabpc1l",
    "Pabpc2",
    "Pabpc4",
    "Pabpc5",
    "Pabpc6",
    "Pabpn1",
    "Pcbp1",
    "Pcbp2",
    "Pcbp3",
    "Pcbp4",
    "Pou5f1",
    "Pprc1",
    "Pspc1",
    "Ptbp1",
    "Ptbp2",
    "Pum1",
    "Pum2",
    "Qk",
    "Qki",
    "Raly",
    "Ralyl",
    "Rbfox1",
    "Rbfox2",
    "Rbfox3",
    "Rbm10",
    "Rbm24",
    "Rbm28",
    "Rbm3",
    "Rbm38",
    "Rbm41",
    "Rbm42",
    "Rbm45",
    "Rbm46",
    "Rbm47",
    "Rbm4b",
    "Rbm5",
    "Rbm8a",
    "Rbms1",
    "Rbms2",
    "Rbms3",
    "Rbmx",
    "Rbmxl2",
    "Rbmxrt",
    "Rod1",
    "Samd4",
    "Samd4b",
    "Sart3",
    "Sf3b4",
    "Sfpq",
    "Snrnp70",
    "Snrpa",
    "Snrpb2",
    "Srrm4",
    "Srsf1",
    "Srsf10",
    "Srsf12",
    "Srsf2",
    "Srsf3",
    "Srsf4",
    "Srsf5",
    "Srsf6",
    "Srsf7",
    "Srsf9",
    "Syncrip",
    "Taf15",
    "Tardbp",
    "Tia1",
    "Tial1",
    "Ttp",
    "Tut1",
    "U2af2",
    "Upf1",
    "Ybx1",
    "Ybx2",
    "Ythdc1",
    "Ythdc2",
    "Yy1",
    "Zc3h10",
    "Zcrb1",
    "Zfp36",
    "Zfp36l1",
    "Zfp36l2",
    "Zfp36l3",
}
