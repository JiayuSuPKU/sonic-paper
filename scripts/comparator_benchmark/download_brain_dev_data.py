"""Download + cache the postnatal mouse brain Visium atlas (GSE313896).

Source: de Launoit, Elisa, et al. "The spatiotemporal dynamics of postnatal
vascularization in the mouse brain." Cell (2026).

The samples comprise male C57BL/6J mouse brains at postnatal
days 5, 7, 9, 12, 21, and 60, sampled across three coronal planes
(ANTERIOR / THALAMIC / BRAINSTEM) with 2-3
biological replicates per (timepoint x plane). 46 sections total.

Idempotent pipeline:

1. Downloads ``GSE313896_RAW.tar`` (~2.4 GB) from GEO into
   ``<data_root>/raw/``.
2. Files inside the tarball are named ``GSM<id>_<encoded_token>_<suffix>``
   where the encoded token varies by submission batch and is unreliable
   for parsing. We group by GSM ID and map to canonical sample titles
   (e.g. ``P5_THALAMIC_PLANE_REP1``) via ``config.BRAIN_DEV_GSM_TITLES``.
3. Reshapes each sample's files into the canonical Space Ranger
   ``outs/`` + ``outs/spatial/`` layout so
   :func:`sonic.utils.load_visium_sample` can ingest them.
4. Writes ``<data_root>/adata/<sample_title>.h5ad`` carrying ``timepoint``,
   ``plane``, and ``replicate`` annotations on ``adata.obs``.

Reruns are cheap: every step skips work whose output already exists.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.comparator_benchmark"

from .config import (
    BRAIN_DEV_GSM_TITLES,
    BRAIN_DEV_MAX_COUNTS_PER_SPOT,
    BRAIN_DEV_MIN_COUNTS_PER_SPOT,
    BRAIN_DEV_MIN_GENE_SPOTS,
    BRAIN_DEV_SPOT_DIAMETER_UM,
    PANELS,
)
from .utils import (
    download_with_progress,
    gunzip_to,
    human_size,
    scale_visium_spatial_to_spot_diameter_um,
)

logger = logging.getLogger("download_brain_dev_data")


# Tarball suffixes we care about → relative path inside outs/.
# Listed longest-first so endswith() matches "list" before "_list" sub-strings.
SUFFIX_MAP: list[tuple[str, str]] = [
    ("_filtered_feature_bc_matrix.h5", "filtered_feature_bc_matrix.h5"),
    ("_tissue_positions_list.csv.gz", "spatial/tissue_positions_list.csv"),
    ("_tissue_positions_list.csv", "spatial/tissue_positions_list.csv"),
    ("_tissue_positions.csv.gz", "spatial/tissue_positions.csv"),
    ("_tissue_positions.csv", "spatial/tissue_positions.csv"),
    ("_scalefactors_json.json.gz", "spatial/scalefactors_json.json"),
    ("_scalefactors_json.json", "spatial/scalefactors_json.json"),
    ("_tissue_hires_image.png.gz", "spatial/tissue_hires_image.png"),
    ("_tissue_hires_image.png", "spatial/tissue_hires_image.png"),
    ("_tissue_lowres_image.png.gz", "spatial/tissue_lowres_image.png"),
    ("_tissue_lowres_image.png", "spatial/tissue_lowres_image.png"),
]

SAMPLE_TITLE_RE = re.compile(
    r"^(?P<timepoint>P\d+)_(?P<plane>ANTERIOR|THALAMIC|BRAINSTEM)_PLANE_REP(?P<rep>\d+)$"
)


def _apply_brain_dev_qc(adata, sample_title: str):
    """Apply the paper-level Visium spot/gene QC before writing AnnData."""
    import scanpy as sc

    before = (adata.n_obs, adata.n_vars)
    adata.var_names_make_unique()

    if "in_tissue" in adata.obs:
        in_tissue = adata.obs["in_tissue"].to_numpy().astype(bool)
        adata = adata[in_tissue].copy()

    sc.pp.filter_genes(adata, min_cells=BRAIN_DEV_MIN_GENE_SPOTS)
    sc.pp.filter_cells(adata, min_counts=BRAIN_DEV_MIN_COUNTS_PER_SPOT)
    sc.pp.filter_cells(adata, max_counts=BRAIN_DEV_MAX_COUNTS_PER_SPOT)
    sc.pp.filter_genes(adata, min_cells=BRAIN_DEV_MIN_GENE_SPOTS)

    logger.info(
        "[%s] QC spots %d -> %d, genes %d -> %d",
        sample_title,
        before[0],
        adata.n_obs,
        before[1],
        adata.n_vars,
    )
    return adata


def _list_samples(tarball: Path) -> dict[str, dict[str, str]]:
    """Group tar members by GSM ID. Returns {gsm_id: {file_role: tar_member}}."""
    samples: dict[str, dict[str, str]] = {}
    with tarfile.open(tarball, "r") as tf:
        for member in tf.getnames():
            base = Path(member).name
            if not base.startswith("GSM"):
                continue
            gsm_id = base.split("_", 1)[0]
            for src_suffix, role in SUFFIX_MAP:
                if base.endswith(src_suffix):
                    samples.setdefault(gsm_id, {})[role] = member
                    break
    return samples


def _extract_sample(
    tarball: Path,
    gsm_id: str,
    members: dict[str, str],
    sample_dir: Path,
) -> bool:
    outs = sample_dir / "outs"
    outs.mkdir(parents=True, exist_ok=True)
    (outs / "spatial").mkdir(parents=True, exist_ok=True)

    canonical = outs / "filtered_feature_bc_matrix.h5"
    pos_list = outs / "spatial" / "tissue_positions_list.csv"
    pos_v2 = outs / "spatial" / "tissue_positions.csv"
    if canonical.exists() and (pos_list.exists() or pos_v2.exists()):
        return True

    with tarfile.open(tarball, "r") as tf:
        for role, member in members.items():
            target = outs / role
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size > 0:
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                logger.warning("[%s] missing tar member %s", gsm_id, member)
                continue
            tmp_gz = target.with_suffix(target.suffix + ".gz.tmp")
            with tmp_gz.open("wb") as fh:
                shutil.copyfileobj(extracted, fh)
            if member.endswith(".gz"):
                gunzip_to(tmp_gz, target)
                tmp_gz.unlink()
            else:
                tmp_gz.rename(target)
    return canonical.exists() and (pos_list.exists() or pos_v2.exists())


def _convert_to_h5ad(
    sample_dir: Path,
    sample_title: str,
    gsm_id: str,
    h5ad_path: Path,
) -> bool:
    try:
        from sonic.utils import load_visium_sample
    except ImportError as e:
        logger.error("Cannot import sonic: %s.", e)
        return False
    try:
        adata = load_visium_sample(sample_dir / "outs")
    except Exception as e:  # noqa: BLE001
        logger.error("[%s] load_visium_sample failed: %s", sample_title, e)
        return False
    adata = _apply_brain_dev_qc(adata, sample_title)
    adata = scale_visium_spatial_to_spot_diameter_um(adata, BRAIN_DEV_SPOT_DIAMETER_UM)

    m = SAMPLE_TITLE_RE.match(sample_title)
    if m:
        adata.obs["timepoint"] = m.group("timepoint")
        adata.obs["plane"] = m.group("plane")
        adata.obs["replicate"] = int(m.group("rep"))
    else:
        adata.obs["timepoint"] = "unknown"
        adata.obs["plane"] = "unknown"
        adata.obs["replicate"] = 0
    adata.obs["sample"] = sample_title
    adata.obs["gsm_id"] = gsm_id

    adata.write_h5ad(h5ad_path)
    return True


def process_sample(
    gsm_id: str,
    members: dict[str, str],
    tarball: Path,
    data_root: Path,
) -> bool:
    sample_title = BRAIN_DEV_GSM_TITLES.get(gsm_id)
    if sample_title is None:
        logger.warning("[%s] not in BRAIN_DEV_GSM_TITLES; skipping.", gsm_id)
        return False

    sample_dir = data_root / "spaceranger" / gsm_id
    h5ad_path = data_root / "adata" / f"{sample_title}.h5ad"

    if h5ad_path.exists() and h5ad_path.stat().st_size > 0:
        logger.info("[%s] cached .h5ad already present", sample_title)
        return True

    matrix_present = "filtered_feature_bc_matrix.h5" in members
    positions_present = (
        "spatial/tissue_positions_list.csv" in members
        or "spatial/tissue_positions.csv" in members
    )
    if not (matrix_present and positions_present):
        logger.warning(
            "[%s] missing required files: matrix=%s, positions=%s",
            sample_title,
            matrix_present,
            positions_present,
        )
        return False

    if not _extract_sample(tarball, gsm_id, members, sample_dir):
        logger.warning("[%s] extraction failed", sample_title)
        return False

    logger.info("[%s] converting to AnnData", sample_title)
    return _convert_to_h5ad(sample_dir, sample_title, gsm_id, h5ad_path)


def _write_metadata_csv(adata_dir: Path) -> None:
    meta_path = adata_dir.parent / "metadata.csv"
    with meta_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["gsm_id", "sample_title", "timepoint", "plane", "replicate", "h5ad_path"]
        )
        for gsm_id, title in sorted(BRAIN_DEV_GSM_TITLES.items()):
            m = SAMPLE_TITLE_RE.match(title)
            if m:
                writer.writerow(
                    [
                        gsm_id,
                        title,
                        m.group("timepoint"),
                        m.group("plane"),
                        m.group("rep"),
                        f"adata/{title}.h5ad",
                    ]
                )
    logger.info("Wrote %s", meta_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PANELS["brain_dev"].data_root,
        help="Output root. Will contain raw/, spaceranger/, adata/ subfolders.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    raw_dir = args.data_root / "raw"
    sr_dir = args.data_root / "spaceranger"
    adata_dir = args.data_root / "adata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sr_dir.mkdir(parents=True, exist_ok=True)
    adata_dir.mkdir(parents=True, exist_ok=True)

    geo_acc = PANELS["brain_dev"].geo_accession
    raw_url = PANELS["brain_dev"].raw_url
    min_bytes = PANELS["brain_dev"].raw_min_bytes
    raw_path = raw_dir / f"{geo_acc}_RAW.tar"

    t0 = time.time()
    if raw_path.exists() and raw_path.stat().st_size >= min_bytes:
        logger.info(
            "Raw tarball already present (%s).", human_size(raw_path.stat().st_size)
        )
    else:
        if raw_path.exists():
            raw_path.unlink()
        logger.info("Downloading %s ...", raw_url)
        if not download_with_progress(raw_url, raw_path, min_bytes):
            return 1

    logger.info("Scanning tarball for samples ...")
    samples = _list_samples(raw_path)
    logger.info("Found %d samples in tarball.", len(samples))

    n_ok = 0
    n_fail = 0
    for gsm_id in sorted(samples):
        if process_sample(gsm_id, samples[gsm_id], raw_path, args.data_root):
            n_ok += 1
        else:
            n_fail += 1

    _write_metadata_csv(adata_dir)

    elapsed = time.time() - t0
    logger.info(
        "Done in %.1fs. OK: %d, failed: %d, total: %d.",
        elapsed,
        n_ok,
        n_fail,
        len(samples),
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
