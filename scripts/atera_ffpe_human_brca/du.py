from __future__ import annotations

import gc
from functools import partial

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from .config import (
    DU_CSV,
    DU_GPR_BACKEND,
    DU_LML_APPROX_RANK,
    DU_METHOD,
    DU_METRICS_CSV,
    DU_REGULATOR_STATUS_CSV,
    DU_RESIDUALIZE,
    DU_SUMMARY_JSON,
    FDR_THRESHOLD,
    GENE_COL_CODEWORD,
    K_NEIGHBORS,
    LAYER,
    MIN_BIN_PCT,
    MIN_COMPONENT_SIZE,
    MIN_COUNTS,
    RHO,
    SPATIAL_KEY,
    TABLE_CODEWORD,
    TABLE_GENE,
)
from .sv import svp_target_genes
from .utils import (
    load_sdata,
    max_rss_gb,
    normalize_gene_symbol,
    regulator_class_keys,
    regulator_gene_sets,
    save_csv,
    save_json,
    stable_hash_values,
    time_call,
)


def _gene_row_order(adata_gene: AnnData, codeword_cell_ids: pd.Index) -> np.ndarray:
    # Gene-level regulator expression must be aligned to the segmented codeword table.
    if "cell_id" in adata_gene.obs:
        gene_cell_ids = adata_gene.obs["cell_id"].astype(str)
    else:
        gene_cell_ids = pd.Index(adata_gene.obs_names).astype(str)
    indexer = pd.Series(np.arange(adata_gene.n_obs), index=gene_cell_ids)
    missing = codeword_cell_ids[~codeword_cell_ids.isin(indexer.index)]
    if len(missing):
        raise ValueError(f"{len(missing):,} codeword cells are absent from the gene table")
    return indexer.loc[codeword_cell_ids].to_numpy()


def _available_gene_symbol_map(adata_gene: AnnData) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in adata_gene.var_names.astype(str):
        out.setdefault(normalize_gene_symbol(name), name)
    return out


def _request_regulator_table() -> pd.DataFrame:
    gene_sets = regulator_gene_sets()
    class_keys = {
        regulator_class: {normalize_gene_symbol(gene) for gene in genes}
        for regulator_class, genes in gene_sets.items()
    }
    symbols_by_key: dict[str, set[str]] = {}
    # Collapse aliases differing only by case while retaining class membership flags.
    for genes in gene_sets.values():
        for gene in genes:
            symbols_by_key.setdefault(normalize_gene_symbol(gene), set()).add(str(gene))
    rows = []
    for key in sorted(symbols_by_key):
        symbol = sorted(symbols_by_key[key])[0]
        rows.append(
            {
                "requested_symbol": symbol,
                "regulator_key": key,
                "is_rbp": key in class_keys["rbp"],
                "is_tf_chip": key in class_keys["tf_chip"],
            }
        )
    return pd.DataFrame(rows)


def _regulator_class_from_flags(is_rbp: bool, is_tf_chip: bool) -> str:
    if is_rbp and is_tf_chip:
        return "rbp;tf_chip"
    if is_rbp:
        return "rbp"
    if is_tf_chip:
        return "tf_chip"
    return "unclassified"


def build_regulator_design(
    adata_gene: AnnData,
    codeword_cell_ids: pd.Index,
) -> tuple[sp.csr_matrix, list[str], pd.DataFrame]:
    """Build sparse log1p-CP10K regulator covariates aligned to codeword cells."""
    requested = _request_regulator_table()
    available = _available_gene_symbol_map(adata_gene)
    requested["present_in_segmented_gene_table"] = requested["regulator_key"].isin(available)
    requested["gene_table_symbol"] = requested["regulator_key"].map(available)
    present = requested[requested["present_in_segmented_gene_table"]].copy()
    if present.empty:
        raise ValueError("No configured RBP/ReMap regulator genes are present in the gene table.")

    row_order = _gene_row_order(adata_gene, codeword_cell_ids)
    gene_symbols = present["gene_table_symbol"].astype(str).tolist()
    x_raw = adata_gene[row_order, gene_symbols].X
    x_raw = x_raw.tocsr() if sp.issparse(x_raw) else sp.csr_matrix(np.asarray(x_raw))
    x_raw = x_raw.astype(np.float32, copy=False)

    # Apply the same expression filters to covariates that are used for features.
    total_counts = np.asarray(x_raw.sum(axis=0)).ravel()
    bin_pct = np.asarray((x_raw > 0).sum(axis=0)).ravel() / max(1, x_raw.shape[0])
    keep = (total_counts >= MIN_COUNTS) & (bin_pct >= MIN_BIN_PCT)
    present["total_counts"] = total_counts
    present["bin_pct"] = bin_pct
    present["tested_in_splisosmnp_du"] = keep

    if not keep.any():
        raise ValueError(
            "No configured regulator genes passed expression filtering "
            f"(min_counts={MIN_COUNTS}, min_bin_pct={MIN_BIN_PCT})."
        )

    kept_symbols = present.loc[keep, "gene_table_symbol"].astype(str).tolist()
    x_keep = x_raw[:, keep].tocsr()
    if "total_counts" in adata_gene.obs:
        cell_totals = adata_gene.obs.iloc[row_order]["total_counts"].astype(float).to_numpy()
    else:
        cell_totals = np.asarray(adata_gene[row_order, :].X.sum(axis=1)).ravel().astype(float)
    scale = np.divide(
        10_000.0,
        cell_totals,
        out=np.zeros_like(cell_totals, dtype=np.float32),
        where=cell_totals > 0,
    )
    x_norm = x_keep.multiply(scale[:, None]).tocsr()
    x_norm.data = np.log1p(x_norm.data).astype(np.float32, copy=False)

    # Status records every requested regulator, including missing/filtered genes.
    status = requested.merge(
        present[
            [
                "regulator_key",
                "gene_table_symbol",
                "total_counts",
                "bin_pct",
                "tested_in_splisosmnp_du",
            ]
        ],
        on="regulator_key",
        how="left",
        suffixes=("", "_present"),
    )
    for col in ["total_counts", "bin_pct"]:
        status[col] = pd.to_numeric(status[col], errors="coerce")
    status["tested_in_splisosmnp_du"] = (
        status["tested_in_splisosmnp_du"]
        .where(status["tested_in_splisosmnp_du"].notna(), False)
        .astype(bool)
    )
    status["regulator_class"] = [
        _regulator_class_from_flags(bool(rbp), bool(tf))
        for rbp, tf in zip(status["is_rbp"], status["is_tf_chip"], strict=False)
    ]
    return x_norm, kept_symbols, status


def _target_adata(sdata, svp: pd.DataFrame) -> tuple[AnnData, list[str]]:
    adata = sdata.tables[TABLE_CODEWORD]
    target_genes = svp_target_genes(svp)
    # DU is restricted to codeword features whose parent gene passed SV FDR.
    target_keys = {normalize_gene_symbol(gene) for gene in target_genes}
    mask = adata.var[GENE_COL_CODEWORD].astype(str).map(normalize_gene_symbol).isin(target_keys)
    if not mask.any():
        raise ValueError("No codeword features found for SVP target genes.")
    return adata[:, mask.to_numpy()].copy(), target_genes


def _cached_du_matches(svp: pd.DataFrame, covariates: list[str]) -> bool:
    if not all(
        path.exists()
        for path in [
            DU_CSV,
            DU_SUMMARY_JSON,
            DU_METRICS_CSV,
            DU_REGULATOR_STATUS_CSV,
        ]
    ):
        return False

    import json

    summary = json.loads(DU_SUMMARY_JSON.read_text())
    target_keys = [normalize_gene_symbol(g) for g in svp_target_genes(svp)]
    covariate_keys = [normalize_gene_symbol(c) for c in covariates]
    # Hash target/covariate identities so cache reuse is tied to the tested design.
    return (
        summary.get("target_gene_key_hash") == stable_hash_values(target_keys)
        and summary.get("covariate_key_hash") == stable_hash_values(covariate_keys)
        and summary.get("parameters", {}).get("gpr_backend") == DU_GPR_BACKEND
        and summary.get("parameters", {}).get("min_counts") == MIN_COUNTS
        and summary.get("parameters", {}).get("min_bin_pct") == MIN_BIN_PCT
        and summary.get("parameters", {}).get("min_component_size") == MIN_COMPONENT_SIZE
    )


def _annotate_du(du: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    class_keys = regulator_class_keys()
    out = du.copy()
    out["gene_key"] = out["gene"].map(normalize_gene_symbol)
    out["covariate_key"] = out["covariate"].map(normalize_gene_symbol)
    out["is_rbp"] = out["covariate_key"].isin(class_keys["rbp"])
    out["is_tf_chip"] = out["covariate_key"].isin(class_keys["tf_chip"])
    out["regulator_class"] = [
        _regulator_class_from_flags(bool(rbp), bool(tf))
        for rbp, tf in zip(out["is_rbp"], out["is_tf_chip"], strict=False)
    ]
    source_cols = [
        "regulator_key",
        "requested_symbol",
        "total_counts",
        "bin_pct",
    ]
    available_source_cols = [col for col in source_cols if col in status.columns]
    out = out.merge(
        status[available_source_cols].drop_duplicates("regulator_key"),
        left_on="covariate_key",
        right_on="regulator_key",
        how="left",
    ).drop(columns=["regulator_key"], errors="ignore")
    out["method"] = DU_METHOD
    out["residualize"] = DU_RESIDUALIZE
    out["gpr_backend"] = DU_GPR_BACKEND
    return out.sort_values(["pvalue_adj", "statistic"], ascending=[True, False]).reset_index(
        drop=True
    )


def _write_du_summary(du: pd.DataFrame, status: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Write compact metadata for validating the canonical all-SVP DU cache."""
    target_keys = sorted(du["gene_key"].dropna().astype(str).unique())
    covariate_keys = sorted(du["covariate_key"].dropna().astype(str).unique())
    summary = {
        "analysis": "du_regulator_segmented_svp_hsic_gp_nufft",
        "n_target_genes": len(target_keys),
        "n_covariates": len(covariate_keys),
        "n_pairs": int(du.shape[0]),
        "n_rbp_covariates": int(
            status.loc[
                status["tested_in_splisosmnp_du"] & status["is_rbp"],
                "regulator_key",
            ].nunique()
        ),
        "n_tf_chip_covariates": int(
            status.loc[
                status["tested_in_splisosmnp_du"] & status["is_tf_chip"],
                "regulator_key",
            ].nunique()
        ),
        "n_sig_pairs_fdr_0_01": int((du["pvalue_adj"] < FDR_THRESHOLD).sum()),
        "target_gene_key_hash": stable_hash_values(target_keys),
        "covariate_key_hash": stable_hash_values(covariate_keys),
        "parameters": {
            "backend": "SplisosmNP",
            "method": DU_METHOD,
            "residualize": DU_RESIDUALIZE,
            "gpr_backend": DU_GPR_BACKEND,
            "gpr_lml_approx_rank": DU_LML_APPROX_RANK,
            "min_counts": MIN_COUNTS,
            "min_bin_pct": MIN_BIN_PCT,
            "min_component_size": MIN_COMPONENT_SIZE,
            "k_neighbors": K_NEIGHBORS,
            "rho": RHO,
        },
        "metrics": metrics.to_dict("records"),
    }
    save_json(summary, DU_SUMMARY_JSON)


def run_du(
    svp: pd.DataFrame,
    *,
    force: bool = False,
    n_jobs: int = 8,
    lml_approx_rank: int = DU_LML_APPROX_RANK,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run/load DU for all segmented SVP genes against all configured regulators."""
    sdata = load_sdata()
    target_adata, target_genes = _target_adata(sdata, svp)
    codeword_cell_ids = pd.Index(target_adata.obs_names).astype(str)
    design, covariates, status = build_regulator_design(
        sdata.tables[TABLE_GENE],
        codeword_cell_ids,
    )

    if DU_CSV.exists() and not force and _cached_du_matches(svp, covariates):
        du = pd.read_csv(DU_CSV)
        metrics = pd.read_csv(DU_METRICS_CSV)
        cached_status = pd.read_csv(DU_REGULATOR_STATUS_CSV)
        return du, cached_status, metrics

    import splisosm
    from splisosm import SplisosmNP

    model = SplisosmNP(k_neighbors=K_NEIGHBORS, rho=RHO, standardize_cov=False)
    # skip_spatial_kernel=True keeps DU focused on regulator associations after
    # SV has already selected spatially variable target genes.
    _, setup_seconds, setup_rss = time_call(
        "all-SVP regulator DU setup",
        partial(
            model.setup_data,
            adata=target_adata,
            spatial_key=SPATIAL_KEY,
            layer=LAYER,
            group_iso_by=GENE_COL_CODEWORD,
            gene_names=GENE_COL_CODEWORD,
            design_mtx=design,
            covariate_names=covariates,
            min_counts=MIN_COUNTS,
            min_bin_pct=MIN_BIN_PCT,
            filter_single_iso_genes=True,
            min_component_size=MIN_COMPONENT_SIZE,
            skip_spatial_kernel=True,
        ),
    )
    _, test_seconds, test_rss = time_call(
        "all-SVP regulator DU hsic-gp nufft",
        partial(
            model.test_differential_usage,
            method=DU_METHOD,
            residualize=DU_RESIDUALIZE,
            ratio_transformation="none",
            nan_filling="mean",
            gpr_backend=DU_GPR_BACKEND,
            gpr_configs={"covariate": {"lml_approx_rank": lml_approx_rank}},
            n_jobs=n_jobs,
            print_progress=True,
        ),
    )
    du = model.get_formatted_test_results("du", with_gene_summary=True)
    # Add regulator source flags and canonical method metadata before caching.
    du = _annotate_du(du, status)
    metrics = pd.DataFrame(
        [
            {
                "analysis": "du_regulator_segmented_svp_hsic_gp_nufft",
                "backend": "SplisosmNP",
                "table": TABLE_CODEWORD,
                "method": DU_METHOD,
                "residualize": DU_RESIDUALIZE,
                "ratio_transformation": "none",
                "gpr_backend": DU_GPR_BACKEND,
                "gpr_lml_approx_rank": lml_approx_rank,
                "target_scope": "all segmented SVP genes with SV FDR < 0.01",
                "n_requested_svp_genes": len(target_genes),
                "n_model_genes": int(model.n_genes),
                "n_covariates": int(len(covariates)),
                "n_pairs": int(du.shape[0]),
                "min_counts": MIN_COUNTS,
                "min_bin_pct": MIN_BIN_PCT,
                "min_component_size": MIN_COMPONENT_SIZE,
                "k_neighbors": K_NEIGHBORS,
                "rho": RHO,
                "skip_spatial_kernel": True,
                "low_rank_path": False,
                "n_jobs": n_jobs,
                "setup_seconds": setup_seconds,
                "setup_max_rss_gb": setup_rss,
                "test_seconds": test_seconds,
                "test_max_rss_gb": test_rss,
                "max_rss_gb": max_rss_gb(),
                "splisosm_version": getattr(splisosm, "__version__", "unknown"),
                "splisosm_file": splisosm.__file__,
            }
        ]
    )
    save_csv(du, DU_CSV)
    save_csv(status, DU_REGULATOR_STATUS_CSV)
    save_csv(metrics, DU_METRICS_CSV)
    _write_du_summary(du, status, metrics)
    del model, sdata
    gc.collect()
    return du, status, metrics
