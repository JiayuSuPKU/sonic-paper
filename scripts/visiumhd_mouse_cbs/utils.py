from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex_mpl_config")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/codex_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import numpy as np
import pandas as pd

from .config import (
    DATASETS,
    FIGURE_DIR,
    NEIGHBOR_DEGREE,
    PROJECT_ROOT,
    RESOLUTION_TABLE,
    RESOLUTION_UM,
    RESULT_DIR,
    RHO,
    SCRIPT_DIR,
    SPLISOSM_SRC,
    DatasetConfig,
)

if str(SPLISOSM_SRC) not in sys.path:
    sys.path.insert(0, str(SPLISOSM_SRC))

import spatialdata as sd
from splisosm import SplisosmFFT


def log(message: str) -> None:
    """Timestamped stdout logging for long workflow runs."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def ensure_dirs() -> None:
    """Create the workflow output directories."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)


def max_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 10**9:
        return rss / 1024**3
    return rss / 1024**2


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write a CSV and log a project-relative path when possible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    try:
        display_path = path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = path
    log(f"wrote {display_path} ({df.shape[0]:,} rows)")


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    log(f"wrote {path.relative_to(PROJECT_ROOT)}")


def normalize_gene_symbol(value: object) -> str:
    """Normalize gene symbols/IDs for case-insensitive joins."""
    return str(value).strip().casefold()


def neighbor_degrees_by_res() -> dict[tuple[str, str], int]:
    """Expand the fixed per-resolution degree map across all datasets."""
    return {
        (cfg.key, resolution): int(NEIGHBOR_DEGREE[resolution])
        for cfg in DATASETS.values()
        for resolution in RESOLUTION_UM
    }


def load_sdata(cfg: DatasetConfig):
    """Open one dataset's SpatialData zarr store."""
    if not cfg.sdata_zarr.exists():
        raise FileNotFoundError(f"Missing SpatialData store: {cfg.sdata_zarr}")
    return sd.read_zarr(cfg.sdata_zarr)


def resolve_bin_element(sdata, cfg: DatasetConfig, resolution: str) -> str:
    """Handle dataset-specific SpatialData shape-name prefixes."""
    table = RESOLUTION_TABLE[resolution]
    shapes = set(getattr(sdata, "shapes", {}).keys())
    candidates = [
        f"{cfg.bin_prefix}{table}",
        table,
        f"_{table}",
        f"Visium_HD_Mouse_Brain_{table}",
    ]
    for candidate in candidates:
        if candidate in shapes:
            return candidate
    suffix_matches = sorted(name for name in shapes if name.endswith(table))
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise ValueError(f"Could not resolve bins for {cfg.key} {resolution}: {sorted(shapes)}")


def resolve_table_config(cfg: DatasetConfig, sdata, resolution: str) -> dict[str, Any]:
    """Resolve the table, shape element, and annotation columns for one grid."""
    table = RESOLUTION_TABLE[resolution]
    if table not in sdata.tables:
        raise ValueError(f"{table!r} not found in {cfg.key}: {sorted(sdata.tables.keys())}")

    adata = sdata.tables[table]
    var_cols = list(adata.var.columns)
    group_iso_by = cfg.group_iso_by if cfg.group_iso_by in var_cols else None
    if group_iso_by is None:
        raise ValueError(f"Could not infer group_iso_by for {cfg.key} {resolution}: {var_cols}")

    gene_names = cfg.gene_names if cfg.gene_names in var_cols else None

    return {
        "dataset_key": cfg.key,
        "resolution": resolution,
        "table_name": table,
        "bins": resolve_bin_element(sdata, cfg, resolution),
        "group_iso_by": group_iso_by,
        "gene_names": gene_names,
        "min_counts": cfg.min_counts,
        "min_bin_pct": cfg.min_bin_pct,
        "spacing_um": RESOLUTION_UM[resolution],
    }


def setup_fft_model(
    sdata,
    table_cfg: dict[str, Any],
    *,
    table_name: str | None = None,
    min_counts: int | None = None,
    min_bin_pct: float | None = None,
    neighbor_degree: int,
    design_mtx: str | None = None,
    filter_single_iso_genes: bool = True,
) -> SplisosmFFT:
    """Construct SplisosmFFT with the settled kernel and table settings."""
    model = SplisosmFFT(
        neighbor_degree=neighbor_degree,
        rho=RHO,
        spacing=(float(table_cfg["spacing_um"]), float(table_cfg["spacing_um"])),
    )
    model.setup_data(
        sdata=sdata,
        bins=table_cfg["bins"],
        table_name=table_name or table_cfg["table_name"],
        col_key="array_col",
        row_key="array_row",
        layer="counts",
        group_iso_by=table_cfg["group_iso_by"],
        gene_names=table_cfg["gene_names"],
        min_counts=table_cfg["min_counts"] if min_counts is None else min_counts,
        min_bin_pct=table_cfg["min_bin_pct"] if min_bin_pct is None else min_bin_pct,
        filter_single_iso_genes=filter_single_iso_genes,
        design_mtx=design_mtx,
    )
    return model


def model_feature_rows(model: SplisosmFFT, table_cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    var = model._adata.var
    group_col = table_cfg["group_iso_by"]
    for gene_name, iso_names in zip(model.gene_names, model._gene_iso_names, strict=True):
        group_id = str(var.loc[iso_names[0], group_col])
        for iso in iso_names:
            rows.append(
                {
                    "gene": str(gene_name),
                    "gene_key": normalize_gene_symbol(gene_name),
                    "group_id": group_id,
                    "isoform": str(iso),
                }
            )
    return pd.DataFrame(rows)


def time_call(label: str, fn) -> tuple[Any, float, float]:
    """Run a callable and return `(value, elapsed_seconds, max_rss_gb)`."""
    log(f"START {label}")
    t0 = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - t0
    rss = max_rss_gb()
    log(f"END {label}: {elapsed:.2f}s, max_rss={rss:.2f} GB")
    return value, elapsed, rss


def minus_log10_p(values: Any) -> np.ndarray:
    """Numerically stable `-log10(p)` for plotting and correlations."""
    p = np.asarray(values, dtype=float)
    p = np.clip(p, np.finfo(float).tiny, 1.0)
    return -np.log10(p)
