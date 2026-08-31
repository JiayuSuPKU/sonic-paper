"""AD/PIG benchmark helpers for QC-aware cache validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import AD_PIG_STALE_PROBE_NAMES
from .utils import cache_has_columns


def ad_pig_cache_has_current_qc(path: Path, gene_col: str = "Feature") -> bool:
    """Return whether a cache no longer contains pre-QC AD/PIG probe names."""
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, usecols=lambda col: col in {"panel", gene_col})
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False
    if "panel" not in df.columns or gene_col not in df.columns:
        return False
    ad_genes = set(df.loc[df["panel"] == "ad_pig", gene_col].dropna().astype(str))
    return bool(ad_genes) and not (ad_genes & AD_PIG_STALE_PROBE_NAMES)


def ad_pig_anisotropy_cache_ready(path: Path) -> bool:
    """Return whether the AD/PIG anisotropy cache has the current schema."""
    if not ad_pig_cache_has_current_qc(path, gene_col="gene"):
        return False
    if not cache_has_columns(
        path,
        {
            "panel",
            "sample",
            "group",
            "gene",
            "gene_set",
            "anisotropy",
            "dominant_angle_deg",
        },
    ):
        return False
    try:
        long = pd.read_csv(path, usecols=["gene_set"])
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False
    return {"PIG", "non-PIG"}.issubset(set(long["gene_set"].astype(str)))


def ad_pig_spectral_replicate_distance_cache_ready(path: Path) -> bool:
    """Return whether the AD/PIG replicate-distance cache has the current schema."""
    if not ad_pig_cache_has_current_qc(path, gene_col="gene"):
        return False
    required = {
        "panel",
        "condition",
        "relation",
        "comparison",
        "sample_a",
        "sample_b",
        "animal_a",
        "animal_b",
        "gene",
        "gene_set",
        "bin",
        "bin_index",
        "bin_type",
        "relative_distance",
    }
    if not cache_has_columns(path, required):
        return False
    try:
        cols = ["panel", "condition", "relation", "bin_type", "relative_distance"]
        long = pd.read_csv(path, usecols=cols)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False
    if long.empty:
        return False
    return (
        set(long["panel"].astype(str)) == {"ad_pig"}
        and {"WT", "AppNL-G-F"}.issubset(set(long["condition"].astype(str)))
        and {"within_animal", "cross_animal"}.issubset(set(long["relation"].astype(str)))
        and {"DC", "radial"}.issubset(set(long["bin_type"].astype(str)))
        and np.isfinite(long["relative_distance"].to_numpy(float)).all()
    )
