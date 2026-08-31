"""Small plotting/data helpers shared by comparator benchmark figures."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import sparse
from sonic.statistics import apply_bh_correction

ENRICHMENT_NUMERIC_COLUMNS: tuple[str, ...] = (
    "abs_delta",
    "stronger_score",
    "score_DF",
    "score_DF_norm",
    "delta_norm_minus_DF",
    "intersection_size_DF",
    "intersection_size_DF_norm",
    "p_adjust_DF",
    "p_adjust_DF_norm",
)


def configure_paper_matplotlib() -> None:
    """Apply the benchmark figure font and line defaults."""
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 6,
            "axes.titlesize": 7,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def neglog10(values) -> np.ndarray:
    """Return clipped -log10 p-values for stable plotting."""
    return -np.log10(np.clip(np.asarray(values, dtype=float), 1e-300, 1.0))


def gene_test_stats(
    wide_tests: pd.DataFrame,
    gene: str,
    tests: tuple[str, ...] = ("DE", "DF", "DF_norm"),
) -> dict[str, float]:
    """Return p-values for one gene from a wide test table."""
    row = wide_tests.set_index("Feature").loc[gene]
    return {test: float(row[f"P_{test}"]) for test in tests}


def fdr_counts(
    wide_tests: pd.DataFrame,
    tests: tuple[str, ...] = ("DE", "DF", "DF_norm"),
    alpha: float = 0.05,
) -> dict[str, int]:
    """Count BH-FDR significant genes for each test in a wide p-value table."""
    return {
        test: int(
            (apply_bh_correction(wide_tests[f"P_{test}"].to_numpy(float)) < alpha).sum()
        )
        for test in tests
    }


def gene_values(adata: Any, gene: str, transform: str | None = "log1p") -> np.ndarray:
    """Return expression values for one AnnData gene, optionally log1p-transformed."""
    x = adata[:, gene].X
    if sparse.issparse(x):
        x = x.toarray()
    values = np.asarray(x).ravel().astype(float)
    if transform in {None, "raw", "none"}:
        return values
    if transform == "log1p":
        return np.log1p(values)
    raise ValueError(f"Unknown expression transform: {transform!r}")


def gene_vmax(
    samples,
    gene: str,
    q: float = 0.99,
    transform: str | None = "log1p",
) -> float:
    """Compute a shared upper color limit for one gene across samples."""
    arrays = [gene_values(adata, gene, transform=transform) for adata in samples]
    if not arrays:
        return 1e-6
    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-6
    return max(float(np.quantile(values, q)), 1e-6)


def scenario_wide(
    tests: pd.DataFrame,
    *,
    panel: str,
    input_transform: str,
    binning: str,
    feature_mode: str,
    requested_n_radial_bins: int,
) -> pd.DataFrame:
    """Filter a long test cache for one scenario and pivot tests to P_* columns."""
    sub = tests[
        tests["panel"].eq(panel)
        & tests["input"].eq(input_transform)
        & tests["binning"].eq(binning)
        & tests["feature_mode"].eq(feature_mode)
        & tests["requested_n_radial_bins"].eq(requested_n_radial_bins)
    ].copy()
    if sub.empty:
        return pd.DataFrame(columns=["panel", "Feature", "P_DE", "P_DF", "P_DF_norm"])
    wide = sub.pivot_table(
        index="Feature", columns="test", values="P_value", aggfunc="first"
    )
    wide = wide.rename(columns={test: f"P_{test}" for test in wide.columns})
    wide = wide.reset_index()
    wide.insert(0, "panel", panel)
    return wide


def scenario_subset(
    df: pd.DataFrame,
    scenario: tuple[str, str, str],
    *,
    requested_n_radial_bins: int,
) -> pd.DataFrame:
    """Filter a long cache for one input/binning/feature-mode scenario."""
    input_transform, binning, feature_mode = scenario
    return df[
        df["input"].astype(str).eq(input_transform)
        & df["binning"].astype(str).eq(binning)
        & df["feature_mode"].astype(str).eq(feature_mode)
        & df["requested_n_radial_bins"].eq(requested_n_radial_bins)
    ].copy()


def ecdf_values(p_values) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y values for an empirical CDF without smoothing."""
    x = np.sort(np.clip(np.asarray(p_values, dtype=float), 0.0, 1.0))
    y = np.arange(1, x.size + 1, dtype=float) / max(x.size, 1)
    return x, y


def clip_for_scatter(values, cap: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Clip plotted values and return a mask of clipped points."""
    values = np.asarray(values, dtype=float)
    if cap is None:
        return values, np.zeros(values.shape, dtype=bool)
    clipped = values > cap
    return np.minimum(values, cap), clipped


def scatter_spearman(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    """Compute Spearman correlation between two -log10 p-value columns."""
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if valid.sum() < 3:
        return np.nan
    x_score = -np.log10(np.maximum(x[valid], 1e-300))
    y_score = -np.log10(np.maximum(y[valid], 1e-300))
    return float(pd.Series(x_score).corr(pd.Series(y_score), method="spearman"))


def load_enrichment_contrasts(
    path: Path,
    *,
    panel: str,
    variant: str,
    numeric_cols: tuple[str, ...] = ENRICHMENT_NUMERIC_COLUMNS,
) -> pd.DataFrame:
    """Load one panel/variant from a cached DF-vs-DF-norm enrichment table."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["panel"].eq(panel) & df["variant"].eq(variant)].copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def wrap_term_label(name, width: int = 40, max_lines: int = 2) -> str:
    """Wrap pathway labels to at most a few compact lines."""
    lines = textwrap.wrap(str(name), width=width)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[: max_lines - 1] + [lines[max_lines - 1] + "..."])


def enrichment_bar_colors(
    plot_df: pd.DataFrame,
    test: str,
    test_colors: dict[str, str],
    alpha: float = 0.05,
) -> list[tuple[float, float, float]]:
    """Return darker bars for significant terms and pale bars otherwise."""
    base = mpl.colors.to_rgb(test_colors[test])
    pale = tuple(np.asarray(base) + (1 - np.asarray(base)) * 0.7)
    p_adj = pd.to_numeric(plot_df[f"p_adjust_{test}"], errors="coerce").fillna(1.0)
    return [base if p <= alpha else pale for p in p_adj]
