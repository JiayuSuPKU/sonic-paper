from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spatialdata as sd
from splisosm import SplisosmFFT

from scripts.paper_utils import (
    ensure_directories,
    log,
    max_rss_gb,
    neglog10,
    normalize_gene_symbol,
    time_call,
)
from scripts.paper_utils import (
    save_csv as _save_csv,
)
from scripts.paper_utils import (
    save_json as _save_json,
)

from .config import (
    DATASETS,
    FIGURE_DIR,
    NEIGHBOR_DEGREE,
    PROJECT_ROOT,
    RESOLUTION_TABLE,
    RESOLUTION_UM,
    RESULT_DIR,
    RHO,
    DatasetConfig,
)

__all__ = ["log", "max_rss_gb", "normalize_gene_symbol", "time_call"]


def ensure_dirs() -> None:
    """Create the workflow output directories."""
    ensure_directories(RESULT_DIR, FIGURE_DIR)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write a CSV and log a project-relative path when possible."""
    _save_csv(df, path, project_root=PROJECT_ROOT, index=index)


def save_json(data: dict[str, Any], path: Path) -> None:
    _save_json(data, path, project_root=PROJECT_ROOT)


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


def minus_log10_p(values: Any) -> np.ndarray:
    """Numerically stable `-log10(p)` for plotting and correlations."""
    return neglog10(values, floor=np.finfo(float).tiny)
