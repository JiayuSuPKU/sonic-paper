"""Shared Stereo-seq preprocessing for embryo_xspecies.

Builds bin50 SpatialData zarr stores from the downloaded per-section h5ad files,
records section geometry/QC, aligns samples to the shared Carnegie-stage axis,
and writes one species-level prepared gene set: genes present in and detected
in every selected section. Prepared zarr stores contain only observed bins:
missing lattice positions are not filled, dropped, or aggregated here.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config as C
from .config import SPECIES_CONFIGS, ensure_species_output_dirs

logger = logging.getLogger("embryo_xspecies.preprocessing")


def _preprocessing_paths(cfg: Any) -> dict[str, Path]:
    """Return the small QC/manifest artifacts owned by preprocessing."""
    return {
        "gene_qc": cfg.RESULT_DIR / "gene_qc.csv",
        "gene_summary_cache": cfg.RESULT_DIR / "gene_qc_cache",
        "section_qc": cfg.RESULT_DIR / "section_qc.csv",
        "summary": cfg.RESULT_DIR / "preprocessing_summary.json",
    }


_MOSTA_INDEX_RE = re.compile(r"^(\d+)_(\d+)(?:_.*)?$")
_MOUSE_E_ANCHORS = np.array([10.5, 11.0, 11.5, 12.0, 12.75, 13.75])
_CARNEGIE_ANCHORS = np.array([14.0, 15.0, 16.0, 17.0, 18.5, 21.5])
_GENE_SUMMARY_CACHE_VERSION = 1


def _trim_heavy(adata: Any, *, drop_raw: bool = False) -> None:
    """Drop unused AnnData slots after the intended count matrix is in ``X``."""
    if adata.uns:
        adata.uns.clear()
    for key in list(adata.obsm.keys()):
        if key != "spatial":
            del adata.obsm[key]
    for key in list(getattr(adata, "obsp", {}).keys()):
        del adata.obsp[key]
    if getattr(adata, "layers", None):
        adata.layers.clear()
    for key in list(getattr(adata, "varm", {}).keys()):
        del adata.varm[key]
    if drop_raw:
        adata.raw = None


def _annotation_col(adata: Any, *, cfg: Any, override: str | None) -> str:
    """Resolve and validate the configured organ/domain label column."""
    col = override or cfg.ANNOTATION_COL
    if col not in adata.obs.columns:
        available = ", ".join(map(str, adata.obs.columns[:20]))
        suffix = " ..." if len(adata.obs.columns) > 20 else ""
        raise ValueError(
            f"configured annotation column {col!r} missing from obs; "
            f"available columns: {available}{suffix}"
        )
    return col


def _domains_of_table(table: Any) -> list[str]:
    """Return non-empty organ/domain labels from a prepared AnnData table."""
    if "annotation" not in table.obs.columns:
        return []
    vals = table.obs["annotation"].astype("string")
    vals = vals[vals.notna() & (vals != "")]
    return sorted(map(str, vals.unique()))


def _is_mosta_index(adata: Any, *, sample: int = 200) -> bool:
    """Return whether ``adata.obs.index`` looks like the MOSTA bin50 lattice."""
    n = min(int(sample), int(adata.n_obs))
    return n > 0 and all(
        _MOSTA_INDEX_RE.match(str(s)) is not None for s in adata.obs.index[:n]
    )


def _coords_from_mosta_index(adata: Any) -> np.ndarray:
    """Recover physical bin50 coordinates from MOSTA ``"<col>_<row>[...]"`` indices."""
    coords: list[tuple[int, int]] = []
    for s in adata.obs.index:
        m = _MOSTA_INDEX_RE.match(str(s))
        if m is None:
            raise ValueError(
                f"obs.index entry {s!r} doesn't match MOSTA '<col>_<row>[_suffix]' format"
            )
        coords.append((int(m.group(1)), int(m.group(2))))
    return np.asarray(coords, dtype=np.float64) * C.BIN_SIZE_UM


def _mouse_e_to_carnegie(e_day: float | np.ndarray) -> float | np.ndarray:
    """Map mouse embryonic day to the shared Carnegie-stage axis."""
    e = np.asarray(e_day, dtype=float)
    cs = np.interp(e, _MOUSE_E_ANCHORS, _CARNEGIE_ANCHORS)
    lo_slope = (_CARNEGIE_ANCHORS[1] - _CARNEGIE_ANCHORS[0]) / (
        _MOUSE_E_ANCHORS[1] - _MOUSE_E_ANCHORS[0]
    )
    hi_slope = (_CARNEGIE_ANCHORS[-1] - _CARNEGIE_ANCHORS[-2]) / (
        _MOUSE_E_ANCHORS[-1] - _MOUSE_E_ANCHORS[-2]
    )
    cs = np.where(
        e < _MOUSE_E_ANCHORS[0],
        _CARNEGIE_ANCHORS[0] + (e - _MOUSE_E_ANCHORS[0]) * lo_slope,
        cs,
    )
    cs = np.where(
        e > _MOUSE_E_ANCHORS[-1],
        _CARNEGIE_ANCHORS[-1] + (e - _MOUSE_E_ANCHORS[-1]) * hi_slope,
        cs,
    )
    return float(cs) if cs.ndim == 0 else cs


def _human_label_to_carnegie(label: str) -> float:
    """Parse a HESTA Carnegie-stage label; ranges use their midpoint."""
    nums = [int(x) for x in re.findall(r"\d+", str(label))]
    return float(np.mean(nums)) if nums else float("nan")


def _carnegie_stage(
    species: str, *, timepoint: str | None = None, day: float | None = None
) -> float:
    """Return the metadata Carnegie-stage value used in preprocessing outputs."""
    if species == "human":
        return _human_label_to_carnegie(str(timepoint))
    if day is None:
        raise ValueError("mouse Carnegie-stage mapping requires day.")
    return float(_mouse_e_to_carnegie(day))


def _count_matrix(adata: Any, *, cfg: Any) -> Any:
    """Return the raw count matrix source for downloaded h5ad QC.

    HESTA stores counts in ``layers['counts']``. Failing here is intentional:
    silently using normalized ``X`` would change gene QC and the shared gene
    universe.
    """
    if cfg.PANEL_KEY == "human":
        if "counts" not in getattr(adata, "layers", {}):
            raise KeyError("HESTA h5ad is missing layers['counts'].")
        return adata.layers["counts"]
    return adata.X


def _as_csr(x: Any) -> sp.csr_matrix:
    """Return ``x`` as CSR, preserving sparse inputs."""
    return x.tocsr() if sp.issparse(x) else sp.csr_matrix(np.asarray(x))


def _subset_to_genes(adata: Any, genes: list[str]) -> Any:
    """Subset to ``genes`` and keep ``.X`` sparse."""
    names = list(map(str, adata.var_names))
    idx_by_name = {g: i for i, g in enumerate(names)}
    missing = [g for g in genes if g not in idx_by_name]
    if missing:
        raise KeyError(
            f"prepared genes missing from section after raw scan: {missing[:10]}"
        )
    out = adata[:, np.array([idx_by_name[g] for g in genes], dtype=int)].copy()
    out.X = _as_csr(out.X)
    return out


def _require_existing_bin_lattice(
    adata: Any,
    *,
    annotation_col: str,
    bin_size_um: float = C.BIN_SIZE_UM,
) -> Any:
    """Normalize an already-binned section onto the shared bin50 lattice.

    The released MOSTA/HESTA inputs are expected to contain one observation per
    observed bin50 location. Unexpected off-lattice coordinates or duplicate
    lattice positions are hard errors so preprocessing cannot silently reshape
    the tissue geometry before ``quadsv`` handles structural missing bins.
    """
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"obsm['spatial'] must be (n, 2); got {xy.shape}.")
    if xy.shape[0] == 0:
        raise ValueError("Cannot prepare an AnnData object with zero observations.")

    x0, y0 = float(xy[:, 0].min()), float(xy[:, 1].min())
    col = np.round((xy[:, 0] - x0) / bin_size_um).astype(np.int64)
    row = np.round((xy[:, 1] - y0) / bin_size_um).astype(np.int64)
    snapped = np.column_stack([x0 + col * bin_size_um, y0 + row * bin_size_um])
    n_col = int(col.max()) + 1
    n_unique = int(np.unique(row * n_col + col).size)
    if n_unique != adata.n_obs:
        raise ValueError(
            "Downloaded h5ad does not have one observation per bin50 lattice "
            f"site ({n_unique} unique sites for {adata.n_obs} observations)."
        )
    max_dev = float(np.max(np.abs(xy - snapped)))
    if not np.allclose(xy, snapped, atol=1e-6, rtol=0.0):
        raise ValueError(
            "Downloaded h5ad coordinates are not on the configured bin50 "
            f"lattice ({bin_size_um:g} um; max snap deviation {max_dev:.3g} um)."
        )

    adata.obs["array_row"] = row.astype(int)
    adata.obs["array_col"] = col.astype(int)
    labels = adata.obs[annotation_col].astype("string").fillna("")
    adata.obs["annotation"] = pd.Categorical(labels.astype(str))
    adata.obsm["spatial"] = np.column_stack(
        [
            col * bin_size_um + bin_size_um / 2.0,
            row * bin_size_um + bin_size_um / 2.0,
        ]
    )
    adata.X = _as_csr(adata.X)
    return adata


def _geometry_from_sdata(sdata: Any, *, bin_size_um: float = C.BIN_SIZE_UM) -> dict:
    """Return geometry/QC for a prepared SpatialData section."""
    table = sdata.tables["bins_table"]
    rows = table.obs["array_row"].to_numpy(dtype=np.int64)
    cols = table.obs["array_col"].to_numpy(dtype=np.int64)
    ny = int(rows.max() - rows.min() + 1)
    nx = int(cols.max() - cols.min() + 1)
    extent_y = ny * bin_size_um
    extent_x = nx * bin_size_um
    min_extent = float(min(extent_y, extent_x))
    n_bins = int(table.n_obs)
    return {
        "ny": ny,
        "nx": nx,
        "extent_y_um": float(extent_y),
        "extent_x_um": float(extent_x),
        "min_extent_um": min_extent,
        "n_bins": n_bins,
        "fill_frac": float(n_bins / (ny * nx)),
        "f_lo_cyc_per_um": float(1.0 / min_extent),
        "f_nyq_cyc_per_um": float(1.0 / (2.0 * bin_size_um)),
    }


def _binned_adata_to_spatialdata(
    adata_binned: Any,
    *,
    bin_size_um: float = C.BIN_SIZE_UM,
    region_name: str = "bins",
) -> Any:
    """Wrap observed bin-level AnnData as SpatialData shapes plus table."""
    import anndata as ad
    import geopandas as gpd
    import spatialdata as sd
    from shapely.geometry import box
    from spatialdata.models import ShapesModel, TableModel

    rows = adata_binned.obs["array_row"].to_numpy(dtype=np.int64)
    cols = adata_binned.obs["array_col"].to_numpy(dtype=np.int64)
    polys = [
        box(
            c * bin_size_um,
            r * bin_size_um,
            (c + 1) * bin_size_um,
            (r + 1) * bin_size_um,
        )
        for r, c in zip(rows, cols, strict=True)
    ]
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame({"geometry": polys}, index=np.arange(len(polys)))
    )
    obs_keep = [
        c
        for c in ("array_row", "array_col", "annotation")
        if c in adata_binned.obs.columns
    ]
    # SpatialData table parsing rejects several upstream var metadata column
    # names, so only the gene index is carried into the prepared zarr.
    var = pd.DataFrame(
        index=pd.Index(adata_binned.var.index, name=adata_binned.var.index.name)
    )
    table = ad.AnnData(
        X=sp.csc_matrix(adata_binned.X),
        obs=adata_binned.obs[obs_keep].copy(),
        var=var,
    )
    table.obs["region"] = pd.Categorical([region_name] * table.n_obs)
    table.obs["instance_id"] = np.arange(table.n_obs, dtype=np.int64)
    table = TableModel.parse(
        table, region=region_name, region_key="region", instance_key="instance_id"
    )
    table.uns["bin_size_um"] = float(bin_size_um)
    table.uns["coordinate_unit"] = "um"
    if "spatial" in getattr(adata_binned, "obsm", {}):
        table.obsm["spatial"] = np.asarray(adata_binned.obsm["spatial"], dtype=float)
    return sd.SpatialData(
        shapes={region_name: shapes}, tables={f"{region_name}_table": table}
    )


def _read_selected_metadata(cfg: Any, stages: list[str] | None) -> pd.DataFrame:
    """Load downloaded-section metadata and attach absolute paths/Carnegie values."""
    if not cfg.META_CSV.exists():
        raise FileNotFoundError(
            f"Missing {cfg.META_CSV} - run the {cfg.PANEL_KEY} download step first."
        )
    meta = pd.read_csv(cfg.META_CSV)
    if stages:
        meta = meta[meta["timepoint"].isin(stages)].copy()
    if meta.empty:
        raise RuntimeError("No sections in metadata.csv match the requested stages.")
    meta = _sort_by_stage(cfg, meta).copy()
    meta["h5ad_path_abs"] = [
        p if (p := Path(str(v))).is_absolute() else C.PROJECT_ROOT / p
        for v in meta["h5ad_path"]
    ]
    missing = meta.loc[~meta["h5ad_path_abs"].map(Path.exists), "h5ad_path_abs"]
    if len(missing):
        raise FileNotFoundError(
            "Missing downloaded h5ad files: "
            + ", ".join(str(p) for p in missing.head(5))
            + (" ..." if len(missing) > 5 else "")
        )
    if cfg.PANEL_KEY == "human":
        meta["carnegie"] = [
            _carnegie_stage("human", timepoint=str(t)) for t in meta["timepoint"]
        ]
    else:
        meta["carnegie"] = [_carnegie_stage("mouse", day=float(d)) for d in meta["day"]]
    return meta.reset_index(drop=True)


def _gene_summary_cache_path(cache_dir: Path, section_id: str) -> Path:
    """Return the per-section gene-QC summary cache path."""
    return cache_dir / f"{section_id}.npz"


def _gene_summary_cache_is_current(path: Path, h5ad_path: Path) -> bool:
    """Return whether a per-section gene summary matches the source h5ad."""
    if not path.exists():
        return False
    stat = h5ad_path.stat()
    with np.load(path, allow_pickle=False) as z:
        return (
            int(z["cache_version"]) == _GENE_SUMMARY_CACHE_VERSION
            and int(z["h5ad_size"]) == int(stat.st_size)
            and int(z["h5ad_mtime_ns"]) == int(stat.st_mtime_ns)
        )


def _read_gene_summary_cache(path: Path, *, include_arrays: bool) -> dict[str, Any]:
    """Read a cached section gene summary, loading only the arrays requested."""
    with np.load(path, allow_pickle=False) as z:
        out: dict[str, Any] = {
            "genes": z["genes"].astype(str),
            "section_qc": json.loads(str(z["section_qc_json"].item())),
        }
        if include_arrays:
            out["total_counts"] = z["total_counts"].astype(np.float64, copy=False)
            out["detected_bins"] = z["detected_bins"].astype(np.int64, copy=False)
        return out


def _write_gene_summary_cache(cfg: Any, r: Any, cache_path: Path) -> None:
    """Compute and cache one section's per-gene totals/detection summaries."""
    import anndata as ad

    h5ad_path = Path(r.h5ad_path_abs)
    stat = h5ad_path.stat()
    adata = ad.read_h5ad(h5ad_path)
    if cfg.PANEL_KEY == "human":
        adata.var_names_make_unique()
    genes = np.asarray(list(map(str, adata.var_names)), dtype=str)
    x = _as_csr(_count_matrix(adata, cfg=cfg))
    bin_counts = np.asarray(x.sum(axis=1)).ravel()
    gene_total = np.asarray(x.sum(axis=0)).ravel().astype(np.float64, copy=False)
    if x.nnz and np.any(x.data <= 0):
        positive = x.copy()
        positive.data = positive.data > 0
        positive.eliminate_zeros()
        bin_genes = np.asarray(positive.getnnz(axis=1)).ravel()
        gene_detected = np.asarray(positive.getnnz(axis=0)).ravel()
        del positive
    else:
        bin_genes = np.asarray(x.getnnz(axis=1)).ravel()
        gene_detected = np.asarray(x.getnnz(axis=0)).ravel()

    section_qc = {
        "section_id": r.section_id,
        "timepoint": r.timepoint,
        "day": float(r.day),
        "carnegie": float(r.carnegie),
        "n_bins_raw": int(adata.n_obs),
        "n_genes_raw": int(adata.n_vars),
        "zero_count_bins": int((bin_counts <= 0).sum()),
        "median_counts_per_bin": float(np.median(bin_counts)),
        "median_genes_per_bin": float(np.median(bin_genes)),
        "total_counts": float(bin_counts.sum()),
        "n_genes_detected": int((gene_detected > 0).sum()),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(
            f,
            cache_version=np.array(_GENE_SUMMARY_CACHE_VERSION, dtype=np.int16),
            h5ad_size=np.array(stat.st_size, dtype=np.int64),
            h5ad_mtime_ns=np.array(stat.st_mtime_ns, dtype=np.int64),
            genes=genes,
            total_counts=gene_total,
            detected_bins=gene_detected.astype(np.int64, copy=False),
            section_qc_json=np.array(json.dumps(section_qc), dtype=str),
        )
    tmp.replace(cache_path)
    del adata, x, bin_counts, bin_genes, gene_total, gene_detected
    gc.collect()


def _ensure_gene_summary_cache(cfg: Any, r: Any, cache_dir: Path) -> Path:
    """Build one per-section gene summary cache if missing or stale."""
    h5ad_path = Path(r.h5ad_path_abs)
    cache_path = _gene_summary_cache_path(cache_dir, str(r.section_id))
    if not _gene_summary_cache_is_current(cache_path, h5ad_path):
        logger.info("[%s] caching per-section gene QC summary.", r.section_id)
        _write_gene_summary_cache(cfg, r, cache_path)
    return cache_path


def _build_prepared_genes(
    cfg: Any,
    meta: pd.DataFrame,
    *,
    cache_dir: Path,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Return genes common to and detected in every selected section.

    The QC table also carries pooled counts and detection totals used by
    downstream ``utils.select_gene_panel``. Matrix reads stay sparse; only 1D summaries are
    materialized as NumPy arrays.
    """
    common: set[str] | None = None
    section_qc_rows: list[dict[str, Any]] = []
    n_total_bins = 0
    cache_paths: list[Path] = []

    # First pass: establish the common gene names and section-level QC.
    for r in meta.itertuples(index=False):
        cache_path = _ensure_gene_summary_cache(cfg, r, cache_dir)
        summary = _read_gene_summary_cache(cache_path, include_arrays=False)
        common = (
            set(map(str, summary["genes"]))
            if common is None
            else common & set(map(str, summary["genes"]))
        )
        section_qc_rows.append(summary["section_qc"])
        n_total_bins += int(summary["section_qc"]["n_bins_raw"])
        cache_paths.append(cache_path)
        del summary
        gc.collect()

    common_genes = sorted(common or set())
    if not common_genes:
        raise RuntimeError(
            f"No genes are common to all selected {cfg.PANEL_KEY} sections."
        )

    total = np.zeros(len(common_genes), dtype=np.float64)
    detected = np.zeros(len(common_genes), dtype=np.int64)
    sections_detected = np.zeros(len(common_genes), dtype=np.int64)
    # Second pass: aggregate cached per-section summaries on the common axis.
    for cache_path in cache_paths:
        summary = _read_gene_summary_cache(cache_path, include_arrays=True)
        idx_by_name = {str(g): i for i, g in enumerate(summary["genes"])}
        idx = np.array([idx_by_name[g] for g in common_genes], dtype=int)
        section_sum = summary["total_counts"][idx]
        section_detected = summary["detected_bins"][idx]
        total += section_sum
        detected += section_detected
        sections_detected += (section_detected > 0).astype(int)
        del summary, idx, idx_by_name, section_sum, section_detected
        gc.collect()

    gene_qc = pd.DataFrame(
        {
            "gene": common_genes,
            "present_in_all_sections": True,
            "sections_detected": sections_detected,
            "detected_bins": detected,
            "total_counts": total,
        }
    )
    gene_qc["mean_expr_all_bins"] = gene_qc["total_counts"] / max(n_total_bins, 1)
    gene_qc["qc_pass"] = gene_qc["sections_detected"].eq(len(meta))
    gene_qc["selected"] = gene_qc["qc_pass"]

    selected = gene_qc.loc[gene_qc["selected"], "gene"].astype(str).tolist()
    if not selected:
        raise RuntimeError("No common genes pass the configured gene QC.")
    section_qc = pd.DataFrame(section_qc_rows)
    section_qc["n_common_genes"] = len(common_genes)
    section_qc["n_selected_genes"] = len(selected)
    return selected, gene_qc, section_qc


def _prepare_mouse_adata(
    adata: Any,
    *,
    cfg: Any,
    annotation_col: str | None,
    prepared_genes: list[str],
) -> Any:
    """Convert one MOSTA h5ad section into the prepared binned AnnData form."""
    _trim_heavy(adata)
    adata = _subset_to_genes(adata, prepared_genes)
    annotation_col = _annotation_col(adata, cfg=cfg, override=annotation_col)
    if _is_mosta_index(adata):
        adata.obsm["spatial"] = _coords_from_mosta_index(adata)
    if "spatial" not in adata.obsm:
        raise ValueError(
            "no obsm['spatial'] and obs.index is not a MOSTA lattice index"
        )

    binned = _require_existing_bin_lattice(adata, annotation_col=annotation_col)
    logger.info(
        "input: %d MOSTA bin50 bins x %d genes; spatial-domain col = %s; using observed lattice.",
        binned.n_obs,
        binned.n_vars,
        annotation_col,
    )
    return binned


def _prepare_human_adata(
    adata: Any,
    *,
    cfg: Any,
    annotation_col: str | None,
    prepared_genes: list[str],
) -> Any:
    """Convert one HESTA h5ad section into the prepared binned AnnData form."""
    if "counts" not in getattr(adata, "layers", {}):
        raise KeyError("HESTA h5ad is missing layers['counts'].")
    adata.X = _as_csr(adata.layers["counts"])
    adata.var_names_make_unique()
    _trim_heavy(adata, drop_raw=True)
    adata = _subset_to_genes(adata, prepared_genes)
    annotation_col = _annotation_col(adata, cfg=cfg, override=annotation_col)
    if "spatial" not in adata.obsm:
        raise ValueError("no obsm['spatial'] - cannot recover coordinates")
    adata.obsm["spatial"] = (
        np.asarray(adata.obsm["spatial"], dtype=np.float64) * C.STEREO_DNB_PITCH_UM
    )

    binned = _require_existing_bin_lattice(adata, annotation_col=annotation_col)
    logger.info(
        "input: %d HESTA bin50 bins x %d genes; organ col = %s; using observed lattice.",
        binned.n_obs,
        binned.n_vars,
        annotation_col,
    )
    return binned


def _convert_one_section(
    cfg: Any,
    h5ad_path: Path,
    section_id: str,
    *,
    prepared_genes: list[str],
    force: bool,
    annotation_col: str | None,
) -> dict[str, Any]:
    """Build or reuse one prepared zarr store and return its geometry row."""
    zpath = cfg.SDATA_DIR / f"{section_id}.zarr"
    if zpath.exists() and not force:
        import spatialdata as sd

        sdata = sd.read_zarr(zpath)
        table = sdata.tables["bins_table"]
        current_bin_size = float(table.uns.get("bin_size_um", float("nan")))
        if list(map(str, table.var_names)) == list(prepared_genes) and np.isclose(
            current_bin_size, C.BIN_SIZE_UM
        ):
            logger.info("[%s] zarr exists with current prepared genes.", section_id)
            geom = _geometry_from_sdata(sdata, bin_size_um=C.BIN_SIZE_UM)
            domains = _domains_of_table(table)
            return {
                "section_id": section_id,
                **geom,
                "n_genes": int(table.n_vars),
                "n_domains": len(domains),
                "domains": ";".join(domains),
            }
        logger.info(
            "[%s] zarr is stale (%d genes; expected %d; bin_size_um=%s; expected %.1f) - rebuilding.",
            section_id,
            table.n_vars,
            len(prepared_genes),
            table.uns.get("bin_size_um", "missing"),
            C.BIN_SIZE_UM,
        )

    import anndata as ad

    logger.info("[%s] reading %s ...", section_id, h5ad_path.name)
    adata = ad.read_h5ad(h5ad_path)
    if cfg.PANEL_KEY == "mouse":
        binned = _prepare_mouse_adata(
            adata,
            cfg=cfg,
            annotation_col=annotation_col,
            prepared_genes=prepared_genes,
        )
    elif cfg.PANEL_KEY == "human":
        binned = _prepare_human_adata(
            adata,
            cfg=cfg,
            annotation_col=annotation_col,
            prepared_genes=prepared_genes,
        )
    else:
        raise ValueError(f"unknown species config: {cfg.PANEL_KEY!r}")
    del adata
    gc.collect()

    sdata = _binned_adata_to_spatialdata(binned, bin_size_um=C.BIN_SIZE_UM)
    table = sdata.tables["bins_table"]
    geom = _geometry_from_sdata(sdata, bin_size_um=C.BIN_SIZE_UM)
    doms = _domains_of_table(table)
    n_genes = int(table.n_vars)
    domains = ";".join(doms)
    logger.info(
        "[%s] bin50: %d bins on %dx%d lattice (%.1fx%.1f mm, fill %.2f), "
        "%d genes retained, %d spatial domains %s",
        section_id,
        geom["n_bins"],
        geom["ny"],
        geom["nx"],
        geom["extent_y_um"] / 1000,
        geom["extent_x_um"] / 1000,
        geom["fill_frac"],
        n_genes,
        len(doms),
        "(observed bin50 lattice; no bin fill/drop/aggregation)",
    )
    if zpath.exists():
        shutil.rmtree(zpath)
    # Keep the current no-autosharding Zarr v3 layout explicit; AnnData warns
    # when this setting is left as None.
    with ad.settings.override(auto_shard_zarr_v3=False):
        sdata.write(zpath)
    logger.info("[%s] wrote %s", section_id, zpath)
    del sdata, binned, table
    gc.collect()
    return {
        "section_id": section_id,
        **geom,
        "n_genes": n_genes,
        "n_domains": len(doms),
        "domains": domains,
    }


def _stage_order(cfg: Any, values: pd.Series) -> list[str]:
    """Return configured stage order restricted to observed values."""
    if hasattr(cfg, "ALL_TIMEPOINTS"):
        return [t for t in cfg.ALL_TIMEPOINTS if t in set(values)]
    if hasattr(cfg, "ALL_CARNEGIE_STAGES"):
        return [t for t in cfg.ALL_CARNEGIE_STAGES if t in set(values)]
    return sorted(map(str, values.unique()))


def _sort_by_stage(cfg: Any, df: pd.DataFrame) -> pd.DataFrame:
    """Sort a metadata/geometry table by configured stage order, then section."""
    rank = {tp: i for i, tp in enumerate(_stage_order(cfg, df["timepoint"]))}
    return (
        df.assign(_stage_rank=df["timepoint"].map(rank).fillna(len(rank)))
        .sort_values(["_stage_rank", "section_id"])
        .drop(columns="_stage_rank")
        .reset_index(drop=True)
    )


def run_preprocessing(
    cfg: Any,
    *,
    stages: list[str] | None = None,
    annotation_col: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    """Write preprocessing QC artifacts and optionally rebuild prepared zarrs."""
    # Create output directories and load per-section metadata.
    ensure_species_output_dirs(cfg)
    paths = _preprocessing_paths(cfg)
    meta = _read_selected_metadata(cfg, stages)

    # Compute detected genes common to all selected sections and write QC tables.
    prepared_genes, gene_qc, section_qc = _build_prepared_genes(
        cfg, meta, cache_dir=paths["gene_summary_cache"]
    )
    gene_qc.to_csv(paths["gene_qc"], index=False)
    section_qc.to_csv(paths["section_qc"], index=False)
    summary = {
        "species": cfg.PANEL_KEY,
        "n_sections": int(len(meta)),
        "n_common_genes": int(len(gene_qc)),
        "n_selected_genes": int(len(prepared_genes)),
        "prepared_gene_rule": "common_to_all_sections_and_detected_in_every_section",
        "gene_summary_cache_dir": str(paths["gene_summary_cache"]),
        "bin_geometry_rule": "observed_bin50_bins_only_no_fill_drop_or_aggregation",
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(
        "%s prepared genes: %d/%d common genes detected in every section; wrote %s",
        cfg.PANEL_KEY,
        len(prepared_genes),
        len(gene_qc),
        paths["gene_qc"],
    )
    if dry_run:
        logger.info("dry-run requested; zarr conversion skipped.")
        return 0

    # Convert each section to a prepared zarr store and collect geometry/QC.
    geom_rows: list[dict[str, Any]] = []
    for _, r in meta.iterrows():
        h5ad_path = Path(r["h5ad_path_abs"])
        grow = _convert_one_section(
            cfg,
            h5ad_path,
            str(r["section_id"]),
            prepared_genes=prepared_genes,
            force=force,
            annotation_col=annotation_col,
        )
        grow["timepoint"] = r["timepoint"]
        grow["day"] = r["day"]
        grow["carnegie"] = float(r["carnegie"])
        geom_rows.append(grow)

    if not geom_rows:
        logger.error("No sections converted - section_geometry.csv not written.")
        return 1

    # Save the geometry/QC table for all converted sections.
    geom = pd.DataFrame(geom_rows)
    cols = [
        "section_id",
        "timepoint",
        "day",
        "carnegie",
        "ny",
        "nx",
        "extent_y_um",
        "extent_x_um",
        "min_extent_um",
        "n_bins",
        "fill_frac",
        "f_lo_cyc_per_um",
        "f_nyq_cyc_per_um",
        "n_genes",
        "n_domains",
        "domains",
    ]
    geom = geom[[c for c in cols if c in geom]].pipe(lambda df: _sort_by_stage(cfg, df))
    geom.to_csv(cfg.GEOMETRY_CSV, index=False)
    logger.info("Wrote %s (%d sections).", cfg.GEOMETRY_CSV, len(geom))

    print("\n=== per-stage geometry (bin50) ===")
    agg = {
        "n_sections": ("section_id", "count"),
        "min_extent_mm_med": ("min_extent_um", lambda s: s.median() / 1000),
        "n_bins_med": ("n_bins", "median"),
        "genes": ("n_genes", "median"),
        "domains_med": ("n_domains", "median"),
    }
    summ = (
        geom.groupby("timepoint")
        .agg(**agg)
        .reindex(_stage_order(cfg, geom["timepoint"]))
    )
    print(summ.to_string(float_format="%.2f"))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the preprocessing CLI parser."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--species",
        choices=["mouse", "human", "both"],
        default="both",
        help="Species to preprocess (default: both).",
    )
    p.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Restrict to these timepoints/Carnegie stages.",
    )
    p.add_argument(
        "--annotation-col",
        default=None,
        help="Override the configured organ/domain .obs column.",
    )
    p.add_argument(
        "--force", action="store_true", help="Rebuild zarr stores even if present."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write QC files without converting zarr stores.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Run preprocessing for one species, or both species by default."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argv = sys.argv[1:] if argv is None else list(argv)
    args = _build_parser().parse_args(argv)
    cfgs = (
        list(SPECIES_CONFIGS.values())
        if args.species == "both"
        else [SPECIES_CONFIGS[args.species]]
    )
    statuses = [
        run_preprocessing(
            cfg,
            stages=args.stages,
            annotation_col=args.annotation_col,
            force=args.force,
            dry_run=args.dry_run,
        )
        for cfg in cfgs
    ]
    return 0 if all(status == 0 for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
