from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.paper_utils import (
    cache_entry_is_current,
    cache_signature,
    installed_version,
    path_fingerprint,
    update_cache_entry,
)

from .config import (
    FDR_THRESHOLD,
    GENE_COL_CODEWORD,
    K_NEIGHBORS,
    LAYER,
    MIN_BIN_PCT,
    MIN_COMPONENT_SIZE,
    MIN_COUNTS,
    RESULT_DIR,
    RHO,
    SDATA_ZARR,
    SPATIAL_KEY,
    SV_CSV,
    SV_METHOD,
    SV_METRICS_CSV,
    SV_SUMMARY_JSON,
    TABLE_CODEWORD,
)
from .utils import (
    load_sdata,
    log,
    normalize_gene_symbol,
    save_csv,
    save_json,
    time_call,
)


def _sv_cache_signature() -> str:
    return cache_signature(
        {
            "inputs": [
                path_fingerprint(SDATA_ZARR),
                path_fingerprint(Path(__file__)),
                path_fingerprint(Path(__file__).with_name("config.py")),
                path_fingerprint(Path(__file__).with_name("utils.py")),
            ],
            "parameters": {
                "method": SV_METHOD,
                "table": TABLE_CODEWORD,
                "layer": LAYER,
                "group_iso_by": GENE_COL_CODEWORD,
                "min_counts": MIN_COUNTS,
                "min_bin_pct": MIN_BIN_PCT,
                "min_component_size": MIN_COMPONENT_SIZE,
                "k_neighbors": K_NEIGHBORS,
                "rho": RHO,
            },
            "splisosm_version": installed_version("splisosm"),
        }
    )


def _valid_canonical_metrics(metrics: pd.DataFrame) -> bool:
    # Reuse SV only when the cached metrics prove canonical filters/backend.
    if metrics.empty:
        return False
    row = metrics.iloc[0]
    return (
        str(row.get("backend", "")) == "SplisosmNP"
        and int(row.get("min_counts", -1)) == MIN_COUNTS
        and abs(float(row.get("min_bin_pct", -1)) - MIN_BIN_PCT) < 1e-12
        and int(row.get("min_component_size", -1)) == MIN_COMPONENT_SIZE
    )


def _load_cached(path: Path, metrics_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    svp = pd.read_csv(path).sort_values(["pvalue_adj", "statistic"], ascending=[True, False])
    # svp_rank is derived from the current sort order rather than trusted from disk.
    svp["svp_rank"] = np.arange(1, svp.shape[0] + 1)
    if "gene_key" not in svp.columns:
        svp["gene_key"] = svp["gene"].map(normalize_gene_symbol)
    return svp, pd.read_csv(metrics_path)


def _write_summary(svp: pd.DataFrame, metrics: pd.DataFrame) -> None:
    summary: dict[str, Any] = {
        "analysis": "segmented_splisosmnp_svp",
        "backend": "SplisosmNP",
        "method": SV_METHOD,
        "n_genes_tested": int(svp.shape[0]),
        "n_sig_fdr_0_01": int((svp["pvalue_adj"] < FDR_THRESHOLD).sum()),
        "parameters": {
            "table": TABLE_CODEWORD,
            "layer": LAYER,
            "group_iso_by": GENE_COL_CODEWORD,
            "gene_names": GENE_COL_CODEWORD,
            "min_counts": MIN_COUNTS,
            "min_bin_pct": MIN_BIN_PCT,
            "min_component_size": MIN_COMPONENT_SIZE,
            "k_neighbors": K_NEIGHBORS,
            "rho": RHO,
        },
        "metrics": metrics.to_dict("records"),
    }
    save_json(summary, SV_SUMMARY_JSON)


def run_sv(*, force: bool = False, n_jobs: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run/load canonical segmented SVP calls."""
    signature = _sv_cache_signature()
    outputs = (SV_CSV, SV_METRICS_CSV, SV_SUMMARY_JSON)
    if not force and cache_entry_is_current(
        RESULT_DIR,
        "sv",
        signature,
        outputs=outputs,
    ):
        svp, metrics = _load_cached(SV_CSV, SV_METRICS_CSV)
        if _valid_canonical_metrics(metrics):
            return svp, metrics

    import splisosm
    from splisosm import SplisosmNP

    sdata = load_sdata()
    adata = sdata.tables[TABLE_CODEWORD]
    model = SplisosmNP(k_neighbors=K_NEIGHBORS, rho=RHO, standardize_cov=False)
    # setup_data applies the same feature and connected-component filters used
    # downstream for DU so the SV target universe stays canonical.
    _, setup_seconds, setup_rss = time_call(
        "segmented SV setup",
        partial(
            model.setup_data,
            adata=adata,
            spatial_key=SPATIAL_KEY,
            layer=LAYER,
            group_iso_by=GENE_COL_CODEWORD,
            gene_names=GENE_COL_CODEWORD,
            min_counts=MIN_COUNTS,
            min_bin_pct=MIN_BIN_PCT,
            filter_single_iso_genes=True,
            min_component_size=MIN_COMPONENT_SIZE,
        ),
    )
    _, test_seconds, test_rss = time_call(
        "segmented SV hsic-ir",
        partial(model.test_spatial_variability, n_jobs=n_jobs, print_progress=True),
    )
    # Keep the gene summary table as the canonical SVP target list.
    svp = model.get_formatted_test_results("sv", with_gene_summary=True).sort_values(
        ["pvalue_adj", "statistic"], ascending=[True, False]
    )
    svp["svp_rank"] = np.arange(1, svp.shape[0] + 1)
    svp["gene_key"] = svp["gene"].map(normalize_gene_symbol)
    metrics = pd.DataFrame(
        [
            {
                "analysis": "segmented_splisosmnp_svp",
                "backend": "SplisosmNP",
                "table": TABLE_CODEWORD,
                "method": SV_METHOD,
                "null_method": "liu",
                "ratio_transformation": "none",
                "filter_single_iso_genes": True,
                "min_counts": MIN_COUNTS,
                "min_bin_pct": MIN_BIN_PCT,
                "min_component_size": MIN_COMPONENT_SIZE,
                "k_neighbors": K_NEIGHBORS,
                "rho": RHO,
                "standardize_cov": False,
                "low_rank_path": False,
                "n_jobs": n_jobs,
                "n_spots": int(model.n_spots),
                "n_genes": int(model.n_genes),
                "avg_isos_per_gene": float(np.mean(model.n_isos_per_gene)),
                "n_sig_fdr_0_01": int((svp["pvalue_adj"] < FDR_THRESHOLD).sum()),
                "setup_seconds": setup_seconds,
                "setup_max_rss_gb": setup_rss,
                "test_seconds": test_seconds,
                "test_max_rss_gb": test_rss,
                "splisosm_version": getattr(splisosm, "__version__", "unknown"),
                "splisosm_file": splisosm.__file__,
            }
        ]
    )
    save_csv(svp.drop(columns=["gene_key"], errors="ignore"), SV_CSV)
    save_csv(metrics, SV_METRICS_CSV)
    _write_summary(svp, metrics)
    update_cache_entry(RESULT_DIR, "sv", signature, outputs)
    log(f"SVP genes at FDR < {FDR_THRESHOLD:g}: {(svp['pvalue_adj'] < FDR_THRESHOLD).sum():,}")
    return svp, metrics


def svp_target_genes(svp: pd.DataFrame) -> list[str]:
    # Return unique significant gene symbols while preserving the ranked SV order.
    sig = svp[pd.to_numeric(svp["pvalue_adj"], errors="coerce") < FDR_THRESHOLD].copy()
    sig = sig.sort_values(["pvalue_adj", "statistic"], ascending=[True, False])
    out: list[str] = []
    seen: set[str] = set()
    for gene in sig["gene"].astype(str):
        key = normalize_gene_symbol(gene)
        if key not in seen:
            out.append(gene)
            seen.add(key)
    return out
