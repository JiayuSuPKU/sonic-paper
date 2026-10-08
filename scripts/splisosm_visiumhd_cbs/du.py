from __future__ import annotations

import gc
import warnings
from functools import partial
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.stats import spearmanr
from spatialdata.models import TableModel

from scripts.paper_utils import (
    cache_entry_is_current,
    cache_signature,
    path_fingerprint,
    update_cache_entry,
)
from scripts.splisosm_utils import source_provenance

from .config import (
    CACHE_REVISIONS,
    DATASETS,
    DU_CACHE_PARAMETERS,
    DU_COVARIATE_MIN_BIN_PCT,
    DU_COVARIATE_MIN_COUNTS,
    DU_METHOD,
    DU_RBP_SUBSET,
    DU_RESIDUALIZE,
    DU_TARGET_GENE_SUBSET,
    DU_TF_SUBSET,
    FDR_THRESHOLD,
    METRICS_PATH,
    RBP_GENES,
    RESULT_DIR,
    DatasetConfig,
)
from .reference_annotations import remap_tf_symbols
from .utils import (
    load_sdata,
    minus_log10_p,
    neighbor_degrees_by_res,
    normalize_gene_symbol,
    resolve_table_config,
    save_csv,
    setup_fft_model,
    time_call,
)


def _du_cache_signature_data(
    cfg: DatasetConfig,
    svp_keys: set[str],
    neighbor_degree: int,
) -> dict[str, Any]:
    target_keys = sorted(svp_keys)
    class_keys = regulator_class_keys()
    return {
        "cache_revision": CACHE_REVISIONS["du"],
        "source": source_provenance(__file__),
        "inputs": [
            path_fingerprint(cfg.sdata_zarr, recursive=True),
            path_fingerprint(RESULT_DIR / f"eligible_16um_features_{cfg.key}.csv"),
        ],
        "parameters": {
            **DU_CACHE_PARAMETERS,
            "dataset": cfg.key,
            "bin_prefix": cfg.bin_prefix,
            "neighbor_degree": neighbor_degree,
            "group_iso_by": cfg.group_iso_by,
            "gene_names": cfg.gene_names,
            "n_target_genes": len(target_keys),
            "target_gene_key_hash": cache_signature({"values": target_keys}),
            "rbp_gene_key_hash": cache_signature({"values": sorted(class_keys["rbp"])}),
            "tf_chip_gene_key_hash": cache_signature({"values": sorted(class_keys["tf_chip"])}),
        },
    }


def build_gene_expression_table(
    adata: AnnData,
    gene_col: str,
    genes: list[str],
    *,
    min_counts: int = DU_COVARIATE_MIN_COUNTS,
    min_bin_pct: float = DU_COVARIATE_MIN_BIN_PCT,
) -> AnnData:
    """Aggregate counts into filtered gene-level regulator covariates.

    Filtering is done on raw aggregated counts before log normalization so DU
    only tests RBP/TF genes with enough observed expression in the same dataset.
    """
    gene_arr = adata.var[gene_col].astype(str).to_numpy()
    targets = {normalize_gene_symbol(g) for g in genes}
    gene_keys = np.array([normalize_gene_symbol(gene) for gene in gene_arr])
    mask = np.isin(gene_keys, list(targets))
    if not mask.any():
        raise ValueError("No requested regulator genes were present in this dataset.")
    selected_gene_arr = gene_arr[mask]
    selected_keys = gene_keys[mask]
    key_to_label = {
        key: selected_gene_arr[np.flatnonzero(selected_keys == key)[0]]
        for key in set(selected_keys)
    }
    unique_keys = sorted(key_to_label)
    unique_genes = [key_to_label[key] for key in unique_keys]
    key_to_idx = {key: i for i, key in enumerate(unique_keys)}
    feature_to_gene = np.array([key_to_idx[key] for key in selected_keys])
    agg_mat = sp.csr_matrix(
        (np.ones(mask.sum()), (feature_to_gene, np.arange(mask.sum()))),
        shape=(len(unique_genes), int(mask.sum())),
    )
    counts = adata.layers["counts"].tocsr()
    x_gene = (counts[:, mask] @ agg_mat.T).tocsr()
    gene_total_counts = np.asarray(x_gene.sum(axis=0)).ravel()
    gene_bin_pct = np.asarray((x_gene > 0).sum(axis=0)).ravel() / max(1, x_gene.shape[0])
    keep = (gene_total_counts >= min_counts) & (gene_bin_pct > min_bin_pct)
    if not keep.any():
        raise ValueError(
            "No requested regulator genes passed the DU covariate expression filter "
            f"(min_counts={min_counts}, min_bin_pct>{min_bin_pct})."
        )
    x_gene = x_gene[:, keep].tocsr()
    unique_genes = [gene for gene, keep_gene in zip(unique_genes, keep, strict=False) if keep_gene]
    unique_keys = [key for key, keep_gene in zip(unique_keys, keep, strict=False) if keep_gene]
    gene_total_counts = gene_total_counts[keep]
    gene_bin_pct = gene_bin_pct[keep]
    total_counts = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.divide(
        1e4,
        total_counts,
        out=np.zeros_like(total_counts, dtype=float),
        where=total_counts > 0,
    )
    x_norm = x_gene.multiply(scale[:, None]).tocsr()
    x_norm.data = np.log1p(x_norm.data)
    x_norm = x_norm.tocsc()
    var = pd.DataFrame(
        {
            "gene_name": unique_genes,
            "gene_ids": unique_genes,
            "gene_key": unique_keys,
            "total_counts": gene_total_counts,
            "bin_pct": gene_bin_pct,
        },
        index=unique_genes,
    )
    out = AnnData(
        X=x_norm,
        obs=adata.obs.copy(),
        var=var,
        uns=adata.uns.copy(),
        obsm=adata.obsm.copy(),
    )
    out.layers["counts"] = x_norm
    return out


def target_genes_for_du(cfg: DatasetConfig, sv16: pd.DataFrame) -> list[str]:
    """Use explicit DU targets if configured; otherwise use dataset-specific SVP genes."""
    _ = cfg
    if DU_TARGET_GENE_SUBSET is not None:
        return list(DU_TARGET_GENE_SUBSET)

    sv16 = _ensure_sv_keys(sv16)
    svp = sv16.loc[sv16["pvalue_adj"] <= FDR_THRESHOLD].sort_values("pvalue_adj")
    ordered: list[str] = []
    seen: set[str] = set()
    for gene in svp["gene"].astype(str):
        key = normalize_gene_symbol(gene)
        if key not in seen:
            ordered.append(gene)
            seen.add(key)
    return ordered


def rbp_genes_for_du() -> list[str]:
    """Use the configured RBP subset or the full curated RBP list."""
    genes = DU_RBP_SUBSET if DU_RBP_SUBSET is not None else tuple(sorted(RBP_GENES))
    return list(genes)


def tf_chip_genes_for_du() -> list[str]:
    """Use the configured TF subset or all ReMap mouse TFs."""
    genes = DU_TF_SUBSET if DU_TF_SUBSET is not None else tuple(remap_tf_symbols())
    return list(genes)


def regulator_gene_sets_for_du() -> dict[str, list[str]]:
    """Return the requested RBP and TF gene sets for the combined DU run."""
    return {"rbp": rbp_genes_for_du(), "tf_chip": tf_chip_genes_for_du()}


def regulator_covariate_genes_for_du() -> list[str]:
    """Return the de-duplicated combined RBP + TF covariate gene list."""
    gene_sets = regulator_gene_sets_for_du()
    return sorted({gene for genes in gene_sets.values() for gene in genes})


def regulator_class_keys() -> dict[str, set[str]]:
    """Normalized covariate keys for post-hoc class-specific summaries."""
    gene_sets = regulator_gene_sets_for_du()
    return {
        class_name: {normalize_gene_symbol(gene) for gene in genes}
        for class_name, genes in gene_sets.items()
    }


def annotate_regulator_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Mark whether each DU covariate is in the RBP and/or TF-ChIP set."""
    df = _ensure_du_keys(df)
    class_keys = regulator_class_keys()
    out = df.copy()
    out["is_rbp"] = out["covariate_key"].isin(class_keys["rbp"])
    out["is_tf_chip"] = out["covariate_key"].isin(class_keys["tf_chip"])
    return out


def subset_du_results_by_class(
    du_results: dict[str, pd.DataFrame], regulator_class: str
) -> dict[str, pd.DataFrame]:
    """Filter combined DU results to RBP-only or TF-ChIP-only covariates."""
    flag_col = {"rbp": "is_rbp", "tf_chip": "is_tf_chip"}[regulator_class]
    subset = {}
    for key, df in du_results.items():
        df = annotate_regulator_classes(df) if flag_col not in df.columns else df.copy()
        subset[key] = df[df[flag_col]].copy()
    return subset


def _ensure_sv_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized gene keys when loading older cached SV tables."""
    if "gene_key" in df.columns:
        return df
    df = df.copy()
    df["gene_key"] = df["gene"].map(normalize_gene_symbol)
    return df


def _ensure_du_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized join keys when loading older cached DU tables."""
    df = df.copy()
    if "gene_key" not in df.columns:
        df["gene_key"] = df["gene"].map(normalize_gene_symbol)
    if "covariate_key" not in df.columns:
        df["covariate_key"] = df["covariate"].map(normalize_gene_symbol)
    return df


def _svp_gene_keys(sv16: pd.DataFrame) -> set[str]:
    """Dataset-specific 16um SVP target set used for DU target restriction."""
    sv16 = _ensure_sv_keys(sv16)
    return set(sv16.loc[sv16["pvalue_adj"] <= FDR_THRESHOLD, "gene_key"])


def _du_output_columns(df: pd.DataFrame) -> list[str]:
    """Keep only stable, downstream-needed DU columns in a predictable order."""
    preferred = [
        "gene",
        "gene_key",
        "covariate",
        "covariate_key",
        "covariate_set",
        "is_rbp",
        "is_tf_chip",
        "statistic",
        "pvalue",
        "pvalue_adj",
        "dataset_key",
        "dataset",
        "resolution",
        "method",
        "residualize",
    ]
    return [col for col in preferred if col in df.columns]


def run_du_suite(
    sv_results: dict[tuple[str, str], pd.DataFrame],
    *,
    force: bool = False,
    n_jobs: int = 8,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Run/load canonical 16um DU tests against combined RBP + TF covariates.

    Each regulator gene is tested once. RBP-only and TF-only summaries are
    derived later by filtering the `is_rbp` and `is_tf_chip` flags.
    """
    covariate_set = "regulator"
    metric_path = METRICS_PATH
    degrees = neighbor_degrees_by_res()
    results: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, Any]] = []
    covariate_genes = regulator_covariate_genes_for_du()
    class_keys = regulator_class_keys()
    for cfg in DATASETS.values():
        out_path = RESULT_DIR / f"du_{covariate_set}_{cfg.key}_16um_hsic_gp.csv"
        sv16 = sv_results.get((cfg.key, "16um"))
        if sv16 is None or sv16.empty:
            continue
        sv16 = _ensure_sv_keys(sv16)
        targets = target_genes_for_du(cfg, sv16)
        svp_keys = {normalize_gene_symbol(gene) for gene in targets}

        signature_data = _du_cache_signature_data(cfg, svp_keys, degrees[(cfg.key, "16um")])
        signature = cache_signature(signature_data)
        section = f"du:{cfg.key}:16um"
        if not force and cache_entry_is_current(
            RESULT_DIR, section, signature, outputs=(out_path,)
        ):
            results[cfg.key] = _ensure_du_keys(pd.read_csv(out_path))
            continue

        sdata = load_sdata(cfg, "16um")
        table_cfg = resolve_table_config(cfg, sdata, "16um")
        adata = sdata.tables[table_cfg["table_name"]]
        covariate_adata = build_gene_expression_table(
            adata, table_cfg["gene_names"], covariate_genes
        )

        # DU target matrix: same-dataset 16um SVP features only.
        gene_mask = (
            adata.var[table_cfg["gene_names"]]
            .astype(str)
            .map(normalize_gene_symbol)
            .isin(svp_keys)
            .to_numpy()
        )
        if gene_mask.sum() == 0:
            warnings.warn(f"No DU target features found for {cfg.key}.", stacklevel=2)
            continue
        eligible = pd.read_csv(RESULT_DIR / f"eligible_16um_features_{cfg.key}.csv")[
            "isoform"
        ].astype(str)
        missing = eligible[~eligible.isin(adata.var_names)]
        if not missing.empty:
            raise ValueError(f"SV eligible features missing from {cfg.key}: {missing.tolist()[:5]}")
        gene_mask &= adata.var_names.astype(str).isin(eligible)
        if not gene_mask.any():
            raise ValueError(f"No SV-eligible DU target features for {cfg.key}")
        target_adata = adata[:, gene_mask].copy()
        target_key = f"_du_targets_{cfg.key}_16um"
        design_key = f"_du_{covariate_set}_{cfg.key}_16um"
        sdata[target_key] = TableModel.parse(target_adata)
        sdata[design_key] = TableModel.parse(covariate_adata)

        model, setup_seconds, setup_rss = time_call(
            f"{cfg.key} 16um {covariate_set} DU setup",
            partial(
                setup_fft_model,
                sdata,
                table_cfg,
                table_name=target_key,
                min_counts=0,
                min_bin_pct=0.0,
                neighbor_degree=degrees[(cfg.key, "16um")],
                design_mtx=design_key,
            ),
        )
        covariate_keys = [normalize_gene_symbol(name) for name in model.covariate_names]
        n_svp_genes = int(model.n_genes)
        n_covariates = len(covariate_keys)
        n_rbps = sum(key in class_keys["rbp"] for key in covariate_keys)
        n_tfs = sum(key in class_keys["tf_chip"] for key in covariate_keys)
        n_pairs = n_svp_genes * n_covariates
        metric_rows.append(
            {
                "covariate_set": covariate_set,
                "dataset_key": cfg.key,
                "dataset": cfg.short_label,
                "resolution": "16um",
                "phase": "du_setup",
                "seconds": setup_seconds,
                "max_rss_gb": setup_rss,
                "neighbor_degree": degrees[(cfg.key, "16um")],
                "projected": False,
                "loaded_existing": False,
                "n_genes": n_svp_genes,
                "n_svp_genes": n_svp_genes,
                "n_covariates": n_covariates,
                "n_rbps": n_rbps,
                "n_tfs": n_tfs,
                "n_pairs": n_pairs,
            }
        )
        _, test_seconds, test_rss = time_call(
            f"{cfg.key} 16um {covariate_set} DU {DU_METHOD}",
            partial(
                model.test_differential_usage,
                method=DU_METHOD,
                residualize=DU_RESIDUALIZE,
                ratio_transformation="none",
                n_jobs=n_jobs,
                print_progress=True,
            ),
        )
        metric_rows.append(
            {
                "covariate_set": covariate_set,
                "dataset_key": cfg.key,
                "dataset": cfg.short_label,
                "resolution": "16um",
                "phase": "du_test",
                "seconds": test_seconds,
                "max_rss_gb": test_rss,
                "neighbor_degree": degrees[(cfg.key, "16um")],
                "projected": False,
                "loaded_existing": False,
                "n_genes": n_svp_genes,
                "n_svp_genes": n_svp_genes,
                "n_covariates": n_covariates,
                "n_rbps": n_rbps,
                "n_tfs": n_tfs,
                "n_pairs": n_pairs,
            }
        )
        du = model.get_formatted_test_results("du", with_gene_summary=False).sort_values(
            "pvalue_adj"
        )
        du["gene_key"] = du["gene"].map(normalize_gene_symbol)
        du["covariate_key"] = du["covariate"].map(normalize_gene_symbol)
        du["covariate_set"] = covariate_set
        du["is_rbp"] = du["covariate_key"].isin(class_keys["rbp"])
        du["is_tf_chip"] = du["covariate_key"].isin(class_keys["tf_chip"])
        du["dataset_key"] = cfg.key
        du["dataset"] = cfg.short_label
        du["resolution"] = "16um"
        du["method"] = DU_METHOD
        du["residualize"] = DU_RESIDUALIZE
        du = du[_du_output_columns(du)]
        save_csv(du, out_path)
        update_cache_entry(
            RESULT_DIR,
            section,
            signature,
            (out_path,),
            signature_data=signature_data,
        )
        results[cfg.key] = du
        del model, sdata
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    if metrics.empty and metric_path.exists():
        # Full cache hit: reuse the last real timing/memory benchmark.
        cached = pd.read_csv(metric_path)
        metrics = cached[cached["phase"].isin(("du_setup", "du_test"))].copy()
    elif not metrics.empty and not force and metric_path.exists():
        # Partial recompute: replace only rows measured in this run.
        key_cols = ["covariate_set", "dataset_key", "resolution", "phase"]
        existing = pd.read_csv(metric_path)
        existing = existing[existing["phase"].isin(("du_setup", "du_test"))].copy()
        replace_keys = metrics[key_cols].drop_duplicates().assign(_replace=True)
        kept = existing.merge(replace_keys, on=key_cols, how="left")
        kept = kept[kept["_replace"].isna()].drop(columns="_replace")
        metrics = pd.concat([kept, metrics], ignore_index=True, sort=False)
    return results, metrics


def _sig_pair_set(df: pd.DataFrame, suffix: str = "") -> set[tuple[str, str]]:
    """Return significant `(target_gene, regulator)` pairs from a DU table."""
    padj_col = f"pvalue_adj{suffix}"
    if padj_col not in df.columns:
        padj_col = "pvalue_adj"
    sig = df[pd.to_numeric(df[padj_col], errors="coerce") <= FDR_THRESHOLD]
    return set(zip(sig["gene_key"], sig["covariate_key"], strict=False))


def _merge_du_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    shared_svp: set[str],
) -> pd.DataFrame:
    """Join two datasets' DU results on shared SVP targets and regulator keys."""
    left = _ensure_du_keys(left)
    right = _ensure_du_keys(right)
    left = left[left["gene_key"].isin(shared_svp)]
    right = right[right["gene_key"].isin(shared_svp)]
    return left[
        [
            "gene_key",
            "covariate_key",
            "gene",
            "covariate",
            "pvalue",
            "pvalue_adj",
            "statistic",
        ]
    ].merge(
        right[["gene_key", "covariate_key", "pvalue", "pvalue_adj", "statistic"]],
        on=["gene_key", "covariate_key"],
        suffixes=("_left", "_right"),
        how="inner",
    )


def du_pairwise_agreement(
    du_results: dict[str, pd.DataFrame],
    sv_results: dict[tuple[str, str], pd.DataFrame],
    *,
    covariate_set: str = "rbp",
) -> pd.DataFrame:
    """Summarize pairwise DU agreement on shared SVP targets and covariates."""
    rows = []
    for left_cfg, right_cfg in combinations(DATASETS.values(), 2):
        left_du = du_results.get(left_cfg.key)
        right_du = du_results.get(right_cfg.key)
        left_sv = sv_results.get((left_cfg.key, "16um"))
        right_sv = sv_results.get((right_cfg.key, "16um"))
        if left_du is None or right_du is None or left_sv is None or right_sv is None:
            continue

        shared_svp = _svp_gene_keys(left_sv) & _svp_gene_keys(right_sv)
        merged = _merge_du_pair(left_du, right_du, shared_svp)
        shared_covariates = set(merged["covariate_key"])
        rho = np.nan
        if merged.shape[0] >= 2:
            rho = spearmanr(
                minus_log10_p(merged["pvalue_left"]),
                minus_log10_p(merged["pvalue_right"]),
                nan_policy="omit",
            ).statistic

        left_sig = _sig_pair_set(merged, "_left")
        right_sig = _sig_pair_set(merged, "_right")
        sig_overlap = left_sig & right_sig
        sig_union = left_sig | right_sig
        rows.append(
            {
                "covariate_set": covariate_set,
                "left_key": left_cfg.key,
                "left": left_cfg.short_label,
                "right_key": right_cfg.key,
                "right": right_cfg.short_label,
                "n_shared_svp": len(shared_svp),
                "n_shared_covariates": len(shared_covariates),
                "n_shared_rbps": (len(shared_covariates) if covariate_set == "rbp" else np.nan),
                "n_shared_tfs": (len(shared_covariates) if covariate_set == "tf_chip" else np.nan),
                "shared_tested_pairs": int(merged.shape[0]),
                "spearman_logp": rho,
                "left_sig_pairs": len(left_sig),
                "right_sig_pairs": len(right_sig),
                "shared_sig_pairs": len(sig_overlap),
                "sig_pair_union": len(sig_union),
                "jaccard_sig_pairs": (len(sig_overlap) / len(sig_union) if sig_union else np.nan),
            }
        )
    return pd.DataFrame(rows)


def _dataset_du_for_shared_pairs(
    cfg_key: str,
    du_results: dict[str, pd.DataFrame],
    shared_svp: set[str],
) -> pd.DataFrame:
    """Prepare one dataset's DU rows for all-platform shared-pair merging."""
    df = _ensure_du_keys(du_results[cfg_key])
    keep = [
        "gene_key",
        "covariate_key",
        "gene",
        "covariate",
        "pvalue",
        "pvalue_adj",
        "statistic",
    ]
    df = df[df["gene_key"].isin(shared_svp)][keep].copy()
    suffix_map = {
        "gene": f"gene_{cfg_key}",
        "covariate": f"covariate_{cfg_key}",
        "pvalue": f"pvalue_{cfg_key}",
        "pvalue_adj": f"pvalue_adj_{cfg_key}",
        "statistic": f"statistic_{cfg_key}",
    }
    return df.rename(columns=suffix_map)


def du_shared_associations(
    du_results: dict[str, pd.DataFrame],
    sv_results: dict[tuple[str, str], pd.DataFrame],
    *,
    covariate_set: str = "rbp",
) -> pd.DataFrame:
    """Rank regulator-SVP pairs tested in all datasets and shared by all 16um SVP calls."""
    dataset_keys = [cfg.key for cfg in DATASETS.values() if cfg.key in du_results]
    svp_sets = [
        _svp_gene_keys(sv_results[(key, "16um")])
        for key in dataset_keys
        if (key, "16um") in sv_results
    ]
    if len(dataset_keys) < 2 or len(svp_sets) != len(dataset_keys):
        return pd.DataFrame()

    shared_svp = set.intersection(*svp_sets)
    merged: pd.DataFrame | None = None
    for key in dataset_keys:
        part = _dataset_du_for_shared_pairs(key, du_results, shared_svp)
        merged = (
            part
            if merged is None
            else merged.merge(part, on=["gene_key", "covariate_key"], how="inner")
        )
    if merged is None or merged.empty:
        return pd.DataFrame()

    padj_cols = [f"pvalue_adj_{key}" for key in dataset_keys]
    pvalue_cols = [f"pvalue_{key}" for key in dataset_keys]
    merged["max_padj"] = merged[padj_cols].max(axis=1)
    merged["mean_neglogp"] = np.vstack([minus_log10_p(merged[col]) for col in pvalue_cols]).mean(
        axis=0
    )
    sig_matrix = merged[padj_cols].le(FDR_THRESHOLD)
    merged["n_sig_datasets"] = sig_matrix.sum(axis=1)
    merged["sig_all"] = sig_matrix.all(axis=1)
    merged["covariate_set"] = covariate_set
    return merged.sort_values(
        ["sig_all", "max_padj", "mean_neglogp"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
