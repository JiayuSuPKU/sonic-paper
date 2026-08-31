"""Download + cache the mouse Alzheimer Stockholm ST atlas (GSE152506).

Source: Chen, Wei-Ting, et al. "Spatial transcriptomics and in situ sequencing to
study Alzheimer's disease." Cell 182.4 (2020): 976-991.

The 20 samples comprise AppNL-G-F (TG, prefix N) and C57Bl/6J (WT, prefix B)
mice profiled at 3, 6, 12, and 18 months.

Idempotent pipeline:

1. Downloads ``GSE152506_raw_counts.txt.gz`` (~258 MB) from GEO into
   ``<data_root>/raw/``.
2. Splits the merged spot-by-gene matrix into per-sample slices using the
   ``<sample>__<x>_<y>`` row index convention.
3. Builds a per-sample :class:`anndata.AnnData` with
   ``obsm['spatial']`` set to physical micrometre coordinates
   (Stockholm ST array, 200 μm spot pitch) and ``obs`` carrying genotype +
   age annotations from ``config.AD_PIG_SAMPLES``.
4. Writes ``<data_root>/adata/<sample>.h5ad`` for fast reloading. Each .h5ad is ~30 MB.

Reruns are cheap: every step skips work whose output already exists.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.comparator_benchmark"

from .config import (
    AD_PIG_SAMPLES,
    PANELS,
    ST_SPOT_PITCH_UM,
)
from .utils import (
    download_with_progress,
    human_size,
)

logger = logging.getLogger("download_ad_pig_data")


def _split_and_convert(raw_counts: Path, adata_dir: Path) -> tuple[int, int]:
    """Read the merged spot×gene CSV, split per sample, write one .h5ad each.

    Returns (n_written, n_skipped_existing).
    """
    try:
        import anndata as ad
        import numpy as np
        import pandas as pd
        from scipy import sparse
    except ImportError as e:
        raise ImportError(
            "Conversion requires pandas, numpy, scipy, anndata. "
            "Install via `pip install -e .[spatial]`."
        ) from e

    expected_h5ads = [adata_dir / f"{s[0]}.h5ad" for s in AD_PIG_SAMPLES]
    if all(p.exists() and p.stat().st_size > 0 for p in expected_h5ads):
        logger.info(
            "All %d per-sample .h5ad files already present.", len(AD_PIG_SAMPLES)
        )
        return 0, len(AD_PIG_SAMPLES)

    logger.info("Reading %s (this may take a minute) ...", raw_counts.name)
    with gzip.open(raw_counts, "rt") as fh:
        df = pd.read_csv(fh, index_col=0, low_memory=False)
    logger.info(
        "Loaded merged matrix: %d spots x %d genes. Splitting by sample ...",
        df.shape[0],
        df.shape[1],
    )

    n_written = 0
    n_skipped = 0
    geo_acc = PANELS["ad_pig"].geo_accession

    for sample, genotype, age in AD_PIG_SAMPLES:
        out_path = adata_dir / f"{sample}.h5ad"
        if out_path.exists() and out_path.stat().st_size > 0:
            n_skipped += 1
            continue

        prefix = f"{sample}__"
        mask = df.index.str.startswith(prefix)
        sub = df.loc[mask].copy()
        if sub.empty:
            logger.warning("No rows for sample %s; skipping.", sample)
            continue

        # Spot IDs follow `<sample>__<x>_<y>` (Stockholm ST array indices, 1-based).
        # NB: pandas Index.str.split returns a MultiIndex when expand=True; cast
        # through a Series first so we get a proper 2-column DataFrame.
        coord_strs = pd.Series(sub.index, index=sub.index).str.removeprefix(prefix)
        coords_df = coord_strs.str.split("_", expand=True)
        if coords_df.shape[1] != 2:
            logger.warning(
                "Sample %s has unexpected spot ID format (e.g. %s); skipping.",
                sample,
                sub.index[0],
            )
            continue
        try:
            xs = coords_df[0].astype(float).to_numpy()
            ys = coords_df[1].astype(float).to_numpy()
        except (ValueError, TypeError) as exc:
            logger.warning("Sample %s coord parse failed (%s); skipping.", sample, exc)
            continue

        values = sub.fillna(0.0).to_numpy(dtype=np.float32)
        X = sparse.csr_matrix(values)

        adata = ad.AnnData(
            X=X,
            obs=pd.DataFrame(
                {
                    "array_col": xs.astype(int),
                    "array_row": ys.astype(int),
                    "sample": sample,
                    "genotype": genotype,
                    "age_months": age,
                },
                index=sub.index.astype(str),
            ),
            var=pd.DataFrame(index=df.columns.astype(str)),
        )
        adata.obsm["spatial"] = np.column_stack(
            [xs * ST_SPOT_PITCH_UM, ys * ST_SPOT_PITCH_UM]
        ).astype(np.float32)
        adata.uns["spatial"] = {
            "scalefactors": {"spot_pitch_um": ST_SPOT_PITCH_UM},
            "platform": "Stockholm Spatial Transcriptomics",
            "source": geo_acc,
        }

        adata.write_h5ad(out_path)
        n_written += 1
        logger.info(
            "  -> %s (%d spots x %d genes, %s/%dM)",
            out_path.name,
            adata.n_obs,
            adata.n_vars,
            genotype,
            age,
        )

    return n_written, n_skipped


def _write_metadata_csv(adata_dir: Path) -> None:
    meta_path = adata_dir.parent / "metadata.csv"
    with meta_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample", "genotype", "age_months", "h5ad_path"])
        for sample, genotype, age in AD_PIG_SAMPLES:
            writer.writerow([sample, genotype, age, f"adata/{sample}.h5ad"])
    logger.info("Wrote %s", meta_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PANELS["ad_pig"].data_root,
        help="Output root. Will contain raw/ and adata/ subfolders.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    raw_dir = args.data_root / "raw"
    adata_dir = args.data_root / "adata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    adata_dir.mkdir(parents=True, exist_ok=True)

    geo_acc = PANELS["ad_pig"].geo_accession
    raw_url = PANELS["ad_pig"].raw_url
    min_bytes = PANELS["ad_pig"].raw_min_bytes
    raw_path = raw_dir / f"{geo_acc}_raw_counts.txt.gz"

    t0 = time.time()
    if raw_path.exists() and raw_path.stat().st_size >= min_bytes:
        logger.info(
            "Raw counts already present (%s).", human_size(raw_path.stat().st_size)
        )
    else:
        if raw_path.exists():
            raw_path.unlink()
        logger.info("Downloading %s ...", raw_url)
        if not download_with_progress(raw_url, raw_path, min_bytes):
            return 1

    n_written, n_skipped = _split_and_convert(raw_path, adata_dir)
    _write_metadata_csv(adata_dir)

    elapsed = time.time() - t0
    logger.info(
        "Done in %.1fs. .h5ad written: %d, already present: %d, total samples: %d.",
        elapsed,
        n_written,
        n_skipped,
        len(AD_PIG_SAMPLES),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
