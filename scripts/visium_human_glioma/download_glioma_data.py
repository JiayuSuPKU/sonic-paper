"""Download + cache the Greenwald et al. 2024 glioma Visium samples (Zenodo 12624860).

Source: Greenwald, Alissa C., et al. "Integrative spatial analysis reveals a multi-layered
organization of glioblastoma." Cell 187.10 (2024): 2485-2501.

Idempotent pipeline:

1. Downloads every ``IDHm_*.tar.gz`` / ``GBM_*.tar.gz`` from Zenodo record
   12624860 into ``<data_root>/raw/``.
2. Extracts each tarball into ``<data_root>/spaceranger/<sample>/``.
3. Loads each sample via :func:`sonic.utils.load_visium_sample`, converts
   Visium full-resolution pixel coordinates to micrometers using the 55 um
   capture-spot diameter, and writes ``<data_root>/adata/<sample>.h5ad`` for
   fast reloading in notebooks.

Reruns are cheap: every step skips work whose archive and completed output
have been validated.

Typical invocation::

    python -m scripts.visium_human_glioma.download_glioma_data
    # or with a custom location:
    python -m scripts.visium_human_glioma.download_glioma_data --data-root /elsewhere/visium_glioma_idh_status
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.visium_human_glioma"

from .config import (
    DATA_ROOT,
    IDH_MUTANT_SAMPLE_IDS,
    IDH_WILDTYPE_SAMPLE_IDS,
    SAMPLE_MAX_QC_EXCLUDED_FRAC,
    SPOT_MAX_PCT_MITO,
    SPOT_MIN_COUNTS,
    ZENODO_BASE_URL,
)

logger = logging.getLogger("download_glioma_data")

MIN_TARBALL_BYTES = 1_000_000  # 1 MB; anything smaller is presumed broken.
EXTRACTION_COMPLETE_MARKER = ".extract_complete"
VISIUM_SPOT_DIAMETER_UM = 55.0
H5AD_CACHE_VERSION = "visium_human_glioma_qc_um_v3"
ALL_SAMPLES = IDH_MUTANT_SAMPLE_IDS + IDH_WILDTYPE_SAMPLE_IDS


def _human(nbytes: int) -> str:
    """Format a byte count for logs."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} TB"


def _tarball_is_valid(path: Path, expected_min_bytes: int) -> bool:
    """Return whether a gzip tarball is large enough and readable end to end."""
    if not path.exists() or path.stat().st_size < expected_min_bytes:
        return False
    try:
        with tarfile.open(path, "r:gz") as tf:
            return bool(tf.getmembers())
    except (tarfile.TarError, OSError, EOFError):
        return False


def _download(url: str, dest: Path, expected_min_bytes: int) -> bool:
    """Stream ``url`` to ``dest`` with a simple progress display. Returns True on success."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            total = int(resp.headers.get("content-length", 0) or 0)
            if total and total < expected_min_bytes:
                logger.warning(
                    "Remote %s reports %s (< %s min); skipping.",
                    url,
                    _human(total),
                    _human(expected_min_bytes),
                )
                return False
            with tmp.open("wb") as fh:
                done = 0
                chunk = 1 << 20  # 1 MiB
                t0 = time.time()
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
                    done += len(buf)
                    if total:
                        pct = 100.0 * done / total
                        dt = max(time.time() - t0, 1e-3)
                        print(
                            f"  {dest.name}: {_human(done)}/{_human(total)} "
                            f"({pct:5.1f}%, {_human(done / dt)}/s)\r",
                            end="",
                            flush=True,
                        )
                print()
        if not _tarball_is_valid(tmp, expected_min_bytes):
            logger.warning(
                "Downloaded %s is not a valid tarball (%s; %s min); discarding.",
                url,
                _human(tmp.stat().st_size),
                _human(expected_min_bytes),
            )
            tmp.unlink()
            return False
        tmp.replace(dest)
        return True
    except (OSError, urllib.error.URLError, TimeoutError) as e:
        logger.error("Download failed for %s: %s", url, e)
        if tmp.exists():
            tmp.unlink()
        return False


def _safe_tar_members(tf: tarfile.TarFile, dest_dir: Path) -> list[tarfile.TarInfo]:
    """Return tar members that cannot write outside ``dest_dir``."""
    dest_root = dest_dir.resolve()
    safe = []
    for member in tf.getmembers():
        if not (member.isfile() or member.isdir()):
            raise tarfile.TarError(
                f"Refusing to extract link/special member: {member.name}"
            )
        target = (dest_root / member.name).resolve()
        if dest_root not in (target, *target.parents):
            raise tarfile.TarError(
                f"Refusing to extract path outside target dir: {member.name}"
            )
        safe.append(member)
    return safe


def _extract(tarball: Path, dest_dir: Path) -> bool:
    """Extract ``tarball`` into ``dest_dir/<sample>/``. Returns True if extracted."""
    try:
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(dest_dir, members=_safe_tar_members(tf, dest_dir))
        (dest_dir / EXTRACTION_COMPLETE_MARKER).write_text("complete\n")
        return True
    except (tarfile.TarError, OSError, EOFError) as e:
        logger.error("Extraction failed for %s: %s", tarball, e)
        return False


def _find_layout_files(extracted_root: Path) -> dict[str, Path | None]:
    """Locate the Greenwald files needed to mimic a Space Ranger ``outs`` tree."""
    suffixes = {
        "matrix": "filtered_feature_bc_matrix.h5",
        "positions": "tissue_positions_list.csv",
        "scalefactors": "scalefactors_json.json",
        "hires_image": "tissue_hires_image.png",
        "lowres_image": "tissue_lowres_image.png",
    }
    matches: dict[str, Path | None] = dict.fromkeys(suffixes)
    for p in extracted_root.rglob("*"):
        if not p.is_file() or p.name.startswith("._"):
            continue
        name = p.name.lower()
        for key, suffix in suffixes.items():
            if matches[key] is None and name.endswith(suffix):
                matches[key] = p
                break
    return matches


def _spaceranger_layout_ready(outs: Path) -> bool:
    """Return True when the normalized ``outs`` layout has required files."""
    spatial = outs / "spatial"
    return (
        (outs / "filtered_feature_bc_matrix.h5").exists()
        and (spatial / "tissue_positions_list.csv").exists()
        and (spatial / "scalefactors_json.json").exists()
    )


def _extraction_ready(extracted_root: Path) -> bool:
    """Return whether extraction finished and contains the required Visium files."""
    if not (extracted_root / EXTRACTION_COMPLETE_MARKER).exists():
        return False
    matches = _find_layout_files(extracted_root)
    return all(
        matches[key] is not None for key in ("matrix", "positions", "scalefactors")
    )


def _normalize_to_spaceranger_layout(extracted_root: Path) -> Path | None:
    """Reshape a Greenwald-style flat tarball into a standard Space Ranger layout.

    The Zenodo tarballs (Greenwald et al., *Cell* 2024) do **not** follow the
    canonical Space Ranger ``outs/`` layout: every file lives in a single inner
    directory and is prefixed with a per-sample lowercase tag (e.g.
    ``mgh258_filtered_feature_bc_matrix.h5``). We copy (hard-link when possible)
    the relevant files into an ``outs/`` + ``outs/spatial/`` tree with the
    standard filenames so :func:`sonic.utils.load_visium_sample` can ingest them.
    Idempotent — skips the reshape if ``outs/filtered_feature_bc_matrix.h5``
    already exists.
    """
    outs = extracted_root / "outs"
    target_matrix = outs / "filtered_feature_bc_matrix.h5"
    if _spaceranger_layout_ready(outs):
        return outs

    matches = _find_layout_files(extracted_root)
    if (
        matches["matrix"] is None
        or matches["positions"] is None
        or matches["scalefactors"] is None
    ):
        logger.error(
            "could not locate required files under %s (found matrix=%s positions=%s scalefactors=%s)",
            extracted_root,
            matches["matrix"],
            matches["positions"],
            matches["scalefactors"],
        )
        return None

    spatial = outs / "spatial"
    spatial.mkdir(parents=True, exist_ok=True)

    dest_map = {
        matches["matrix"]: target_matrix,
        matches["positions"]: spatial / "tissue_positions_list.csv",
        matches["scalefactors"]: spatial / "scalefactors_json.json",
    }
    if matches["hires_image"] is not None:
        dest_map[matches["hires_image"]] = spatial / "tissue_hires_image.png"
    if matches["lowres_image"] is not None:
        dest_map[matches["lowres_image"]] = spatial / "tissue_lowres_image.png"

    for src, dst in dest_map.items():
        if dst.exists():
            continue
        try:
            # Hard-link if on same filesystem; otherwise copy.
            import os as _os

            _os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return outs


def _h5ad_is_current(path: Path) -> bool:
    """Return whether ``path`` is a micrometer-scaled glioma cache file."""
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        import h5py
    except ImportError:
        return False
    try:
        with h5py.File(path, "r") as h5:
            node = h5.get("uns/visium_human_glioma_h5ad_version")
            if node is None:
                return False
            value = node[()]
    except OSError:
        return False
    if isinstance(value, bytes):
        value = value.decode()
    elif hasattr(value, "tolist"):
        value = value.tolist()
        if isinstance(value, bytes):
            value = value.decode()
    return str(value) == H5AD_CACHE_VERSION


def _set_physical_coordinates(adata, sample: str):
    """Store Visium coordinates in micrometers, matching comparator_benchmark."""
    import numpy as np

    from scripts.comparator_benchmark.utils import (
        scale_visium_spatial_to_spot_diameter_um,
    )

    if "spatial" in adata.obsm:
        adata.obsm["spatial_fullres_px"] = np.asarray(
            adata.obsm["spatial"], dtype=float
        ).copy()
    adata = scale_visium_spatial_to_spot_diameter_um(adata, VISIUM_SPOT_DIAMETER_UM)

    spatial_uns = adata.uns.get("spatial")
    if isinstance(spatial_uns, dict):
        spatial_uns["path"] = f"spaceranger/{sample}/outs"
    adata.uns["visium_human_glioma_h5ad_version"] = H5AD_CACHE_VERSION
    return adata


def _apply_glioma_qc(adata, sample: str):
    """Apply the Greenwald/Ravi Visium spot QC before caching AnnData."""
    import numpy as np
    import scanpy as sc

    # Filter out spots with <1000 total counts and/or >20% mitochondrial reads
    adata.var_names_make_unique()
    adata.var["mt"] = [str(gene).upper().startswith("MT-") for gene in adata.var_names]
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )
    total_counts = adata.obs["total_counts"].to_numpy(float)
    pct_mito = adata.obs["pct_counts_mt"].to_numpy(float)
    keep = (
        np.isfinite(total_counts)
        & np.isfinite(pct_mito)
        & (total_counts >= SPOT_MIN_COUNTS)
        & (pct_mito <= SPOT_MAX_PCT_MITO)
    )
    before = int(adata.n_obs)
    excluded = before - int(keep.sum())
    excluded_frac = excluded / max(before, 1)
    if excluded_frac > SAMPLE_MAX_QC_EXCLUDED_FRAC:
        raise ValueError(
            f"{sample} failed spot QC: excluded {excluded}/{before} spots ({excluded_frac:.1%})"
        )

    adata.obs["total_counts_raw"] = total_counts
    adata = adata[keep].copy()
    adata.uns["glioma_spot_qc"] = {
        "min_counts": int(SPOT_MIN_COUNTS),
        "max_pct_mito": float(SPOT_MAX_PCT_MITO),
        "max_excluded_fraction": float(SAMPLE_MAX_QC_EXCLUDED_FRAC),
        "n_spots_before_qc": before,
        "n_spots_after_qc": int(adata.n_obs),
        "n_spots_excluded": excluded,
        "excluded_fraction": float(excluded_frac),
    }
    logger.info(
        "[%s] QC kept %d/%d spots (excluded %.1f%%)",
        sample,
        adata.n_obs,
        before,
        100.0 * excluded_frac,
    )
    return adata


def _convert_to_h5ad(spaceranger_dir: Path, out_path: Path, sample: str) -> bool:
    """Load a Space Ranger sample dir and write it as .h5ad. Returns True on success."""
    try:
        # local import so --help doesn't need it
        from sonic.utils import load_visium_sample
    except ImportError as e:
        logger.error(
            "Cannot import sonic: %s. Install the package in dev mode first.", e
        )
        return False

    # The Greenwald tarballs need a one-time reshape into the Space Ranger layout.
    reshaped = _normalize_to_spaceranger_layout(spaceranger_dir)
    if reshaped is None:
        return False

    try:
        adata = load_visium_sample(reshaped)
        adata = _apply_glioma_qc(adata, sample)
        adata = _set_physical_coordinates(adata, sample)
        tmp = out_path.with_name(out_path.name + ".tmp.h5ad")
        if tmp.exists():
            tmp.unlink()
        adata.write_h5ad(tmp)
        tmp.replace(out_path)
        logger.info(
            "  -> %s (%d spots x %d genes)",
            out_path.name,
            adata.n_obs,
            adata.n_vars,
        )
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to convert %s -> .h5ad: %s", spaceranger_dir, e)
        return False


def process_sample(sample: str, data_root: Path) -> tuple[str, bool]:
    """Run the download → extract → convert pipeline for one sample (idempotent)."""
    raw = data_root / "raw" / f"{sample}.tar.gz"
    extracted_root = data_root / "spaceranger" / sample
    h5ad = data_root / "adata" / f"{sample}.h5ad"

    # 1. Download.
    if _tarball_is_valid(raw, MIN_TARBALL_BYTES):
        logger.info(
            "[%s] raw tarball already present (%s)", sample, _human(raw.stat().st_size)
        )
    else:
        if raw.exists():
            raw.unlink()
        url = ZENODO_BASE_URL.format(f"{sample}.tar.gz")
        logger.info("[%s] downloading from %s", sample, url)
        if not _download(url, raw, MIN_TARBALL_BYTES):
            return sample, False

    # 2. Extract.
    if _extraction_ready(extracted_root):
        logger.info("[%s] spaceranger dir already extracted", sample)
    else:
        if extracted_root.exists():
            shutil.rmtree(extracted_root)
        extracted_root.mkdir(parents=True, exist_ok=True)
        logger.info("[%s] extracting %s", sample, raw.name)
        if not _extract(raw, extracted_root):
            return sample, False

    # 3. Convert to .h5ad.
    if _h5ad_is_current(h5ad):
        logger.info("[%s] cached micrometer-scale .h5ad already present", sample)
    else:
        if h5ad.exists():
            logger.info("[%s] regenerating stale .h5ad cache", sample)
        logger.info("[%s] converting to AnnData", sample)
        if not _convert_to_h5ad(extracted_root, h5ad, sample):
            return sample, False

    return sample, True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Output root. Will contain raw/, spaceranger/, adata/ subfolders.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    for sub in ("raw", "spaceranger", "adata"):
        (args.data_root / sub).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    succeeded: list[str] = []
    failed: list[str] = []
    for sample in ALL_SAMPLES:
        name, ok = process_sample(sample, args.data_root)
        (succeeded if ok else failed).append(name)

    elapsed = time.time() - t0
    logger.info(
        "Done in %.1fs. OK: %d (%d GBM + %d IDHm). Failed: %d.",
        elapsed,
        len(succeeded),
        sum(1 for s in succeeded if s.startswith("GBM_")),
        sum(1 for s in succeeded if s.startswith("IDHm_")),
        len(failed),
    )
    if failed:
        logger.warning("Failed samples: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
