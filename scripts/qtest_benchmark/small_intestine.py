"""Small-intestine kernel analyses loaded directly from SpatialData."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.fft
from scipy.ndimage import gaussian_filter
from sonic.kernels.fft import FFTKernel
from sonic.statistics import spatial_q_test

from .config import (
    ARTIFACT_GENES,
    CAR_K_NEIGHBORS,
    CAR_RHOS,
    FAST_DECAY_RHO,
    IMMUNE_GENES,
    PIXEL_SIZE_UM,
    SLOW_DECAY_RHO,
    SMALL_INTESTINE_PATTERN_RANKS_FILE,
    SMALL_INTESTINE_SCORES_FILE,
    ensure_output_dirs,
)
from .utils import cache_is_reusable

LOGGER = logging.getLogger(__name__)
CACHE_INPUTS = (Path(__file__), Path(__file__).with_name("config.py"))


def _gene_indices(table, genes: tuple[str, ...]) -> np.ndarray:
    indices = table.var_names.get_indexer(genes)
    if (indices < 0).any():
        missing = [
            gene for gene, index in zip(genes, indices, strict=True) if index < 0
        ]
        raise ValueError(f"Missing configured genes in square_016um: {missing}")
    return indices


def load_gene_maps(source: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the eight plotted 16 µm gene rasters directly from SpatialData."""
    if not source.exists():
        raise FileNotFoundError(
            f"SpatialData source not found: {source}. Pass --source to override it."
        )

    import spatialdata as sd

    table = sd.read_zarr(source).tables["square_016um"]
    genes = ARTIFACT_GENES + IMMUNE_GENES
    gene_indices = _gene_indices(table, genes)

    rows_raw = table.obs["array_row"].to_numpy(dtype=int)
    cols_raw = table.obs["array_col"].to_numpy(dtype=int)
    rows, cols = rows_raw - rows_raw.min(), cols_raw - cols_raw.min()
    shape = (int(rows.max()) + 1, int(cols.max()) + 1)
    maps = np.zeros((len(genes), *shape), dtype=np.float32)
    maps[:, rows, cols] = table.X[:, gene_indices].toarray().T.astype(np.float32)
    mask = np.zeros(shape, dtype=bool)
    mask[rows, cols] = True

    return maps.astype(float), mask, list(genes)


def mean_fill_structural_holes(maps: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill off-tissue bins by each gene mean, matching DetectorGrid behavior."""
    filled = np.asarray(maps, dtype=float).copy()
    means = filled[:, mask].mean(axis=1)
    filled[:, ~mask] = means[:, None]
    return filled


def radial_profile(
    values: np.ndarray, *, pixel_size_um: float = PIXEL_SIZE_UM, n_bins: int = 72
) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally average an FFT-layout spectrum on a normalized radial grid."""
    ny, nx = values.shape
    fy = scipy.fft.fftfreq(ny, d=pixel_size_um)
    fx = scipy.fft.fftfreq(nx, d=pixel_size_um)
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    max_radius = 1.0 / (2.0 * pixel_size_um)
    edges = np.linspace(0.0, max_radius, n_bins + 1)
    idx = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=np.asarray(values).ravel(), minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    profile = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = (counts > 0) & (centers > 0)
    return centers[valid], profile[valid]


def rho_key(rho: float) -> str:
    """Return the stable cache-column key for a CAR autocorrelation value."""
    return f"rho_{rho:.2f}"


def cache_small_intestine_kernel_results(
    source: Path,
    *,
    force: bool = False,
    chunk_size: int = 64,
) -> pd.DataFrame:
    """Cache transcriptome-wide scores and pattern ranks across kernel choices."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    ensure_output_dirs()
    outputs = (SMALL_INTESTINE_SCORES_FILE, SMALL_INTESTINE_PATTERN_RANKS_FILE)
    if cache_is_reusable(
        *outputs,
        force=force,
        inputs=(*CACHE_INPUTS, source),
        logger=LOGGER,
        label="Small-intestine analysis",
    ):
        return pd.read_csv(SMALL_INTESTINE_SCORES_FILE)
    if not source.exists():
        raise FileNotFoundError(
            f"SpatialData source not found: {source}. It is required once to build "
            f"{SMALL_INTESTINE_SCORES_FILE.name}."
        )

    import spatialdata as sd

    # Load the 16 µm square SpatialData and test all genes expressed in at least 10 bins.
    table = sd.read_zarr(source).tables["square_016um"]
    panel_genes = ARTIFACT_GENES + IMMUNE_GENES
    totals = np.asarray(table.X.sum(axis=0)).ravel()
    keep = np.flatnonzero(totals >= 10)
    genes = table.var_names[keep].astype(str).to_numpy()

    rows_raw = table.obs["array_row"].to_numpy(dtype=int)
    cols_raw = table.obs["array_col"].to_numpy(dtype=int)
    rows, cols = rows_raw - rows_raw.min(), cols_raw - cols_raw.min()
    shape = (int(rows.max()) + 1, int(cols.max()) + 1)
    mask = np.zeros(shape, dtype=bool)
    mask[rows, cols] = True

    # Prepare a set of CAR kernels with predefined varying autocorrelation decay.
    kernels = {
        rho_key(rho): FFTKernel(
            shape,
            spacing=(PIXEL_SIZE_UM, PIXEL_SIZE_UM),
            method="car",
            rho=rho,
            k_neighbors=CAR_K_NEIGHBORS,
            fft_solver="fft2",
            workers=1,
        )
        for rho in CAR_RHOS
    }

    # For selected example genes, prepare expression with mean-filling similar to
    # DetectorGrid for later hyperparameter fitting.
    panel_indices = _gene_indices(table, panel_genes)
    panel_maps = np.zeros((len(panel_genes), *shape), dtype=np.float32)
    panel_maps[:, rows, cols] = table[:, panel_indices].X.toarray().T
    standardized = mean_fill_structural_holes(panel_maps, mask)
    standardized -= standardized.mean(axis=(1, 2), keepdims=True)
    standardized /= (
        standardized.std(axis=(1, 2), ddof=1, keepdims=True) + np.finfo(float).eps
    )
    pattern_templates = {
        "Swiss-roll artifact": standardized[
            [panel_genes.index(gene) for gene in ARTIFACT_GENES]
        ].mean(axis=0),
        "Immune cluster": standardized[
            [panel_genes.index(gene) for gene in IMMUNE_GENES]
        ].mean(axis=0),
    }
    fit_metrics: dict[str, tuple[float, float]] = {}
    for target, contrast in (
        ("Swiss-roll artifact", "Immune cluster"),
        ("Immune cluster", "Swiss-roll artifact"),
    ):
        # Retain frequencies enriched in the target relative to the contrast.
        key = "fit_artifact" if target == "Swiss-roll artifact" else "fit_immune"
        kernel, fit_seconds = design_contrastive_pattern_kernel(
            pattern_templates[target], pattern_templates[contrast], mask
        )
        kernels[key] = kernel
        fit_metrics[key] = (fit_seconds, kernel.spectrum.nbytes / 1e6)

    # For each kernel, run the spatial Q-test on all genes.
    z_scores = {key: np.empty(len(keep), dtype=float) for key in kernels}

    for start in range(0, len(keep), chunk_size):
        end = min(start + chunk_size, len(keep))
        maps = np.zeros((end - start, *shape), dtype=np.float32)
        maps[:, rows, cols] = table.X[:, keep[start:end]].toarray().T
        maps[:, ~mask] = maps[:, mask].mean(axis=1)[:, None]
        data = np.moveaxis(maps, 0, -1)
        for key, kernel in kernels.items():
            q = np.asarray(spatial_q_test(data, kernel, return_pval=False))
            z_scores[key][start:end] = (q - kernel.trace()) / np.sqrt(
                2.0 * kernel.square_trace()
            )

    result = pd.DataFrame(
        {
            "gene": genes,
            f"Z_{rho_key(SLOW_DECAY_RHO)}": z_scores[rho_key(SLOW_DECAY_RHO)],
            f"Z_{rho_key(FAST_DECAY_RHO)}": z_scores[rho_key(FAST_DECAY_RHO)],
        }
    )
    priority_rows: list[dict] = []
    for key in kernels:
        ranks = pd.Series(z_scores[key]).rank(method="average", ascending=False)
        for group, group_genes in (
            ("Swiss-roll artifact", ARTIFACT_GENES),
            ("Immune cluster", IMMUNE_GENES),
        ):
            selected = np.isin(genes, group_genes)
            is_target_fit = (key, group) in {
                ("fit_artifact", "Swiss-roll artifact"),
                ("fit_immune", "Immune cluster"),
            }
            priority_rows.append(
                {
                    "kernel": key,
                    "group": group,
                    "median_transcriptome_rank": float(ranks[selected].median()),
                    "fit_seconds": fit_metrics[key][0] if is_target_fit else np.nan,
                    "spectral_storage_mb": (
                        fit_metrics[key][1] if is_target_fit else np.nan
                    ),
                }
            )
    result.to_csv(SMALL_INTESTINE_SCORES_FILE, index=False)
    pd.DataFrame(priority_rows).to_csv(SMALL_INTESTINE_PATTERN_RANKS_FILE, index=False)
    LOGGER.info("Small-intestine analysis: cached scores for %d genes", len(result))
    return result


def design_contrastive_pattern_kernel(
    target: np.ndarray, contrast: np.ndarray, mask: np.ndarray
) -> tuple[FFTKernel, float]:
    """Fit a PSD spectrum to frequencies enriched in one pattern versus another."""
    start = time.perf_counter()

    # Compare normalized power spectra after mean-filling off-tissue bins.
    powers = []
    for pattern in (target, contrast):
        filled = pattern.astype(float).copy()
        filled[~mask] = float(pattern[mask].mean())
        filled -= filled.mean()
        filled /= filled.std(ddof=1) + np.finfo(float).eps
        power = gaussian_filter(
            np.abs(scipy.fft.fft2(filled, workers=1)) ** 2, 1.25, mode="wrap"
        )
        power[0, 0] = 0.0
        power_total = power.sum()
        if not np.isfinite(power_total) or power_total <= 0:
            raise ValueError("Pattern has no non-constant spectral power")
        powers.append(power / power_total)

    # Compute the contrast emphasizing frequencies enriched in the target pattern.
    spectrum = np.maximum(powers[0] - powers[1], 0.0)
    total = spectrum.sum()
    if total <= 0:
        raise ValueError("Target spectrum has no power beyond the contrast spectrum")
    spectrum *= (target.size - 1) / total

    # Use the rescaled spectrum to define a new FFTKernel.
    kernel = FFTKernel(
        target.shape,
        spacing=(PIXEL_SIZE_UM, PIXEL_SIZE_UM),
        method="car",
        rho=SLOW_DECAY_RHO,
        k_neighbors=CAR_K_NEIGHBORS,
        fft_solver="fft2",
        workers=1,
    )
    kernel.spectrum = spectrum.ravel()
    return kernel, time.perf_counter() - start
