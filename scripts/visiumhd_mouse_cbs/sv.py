from __future__ import annotations

import gc
from functools import partial
from itertools import combinations, product
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import spearmanr
from spatialdata.models import TableModel

from .config import (
    DATASETS,
    FDR_THRESHOLD,
    RESULT_DIR,
    SV_GENE_SUBSET,
    SV_METHOD,
    SV_METRICS_PATH,
    DatasetConfig,
)
from .utils import (
    load_sdata,
    minus_log10_p,
    model_feature_rows,
    neighbor_degrees_by_res,
    normalize_gene_symbol,
    resolve_table_config,
    save_csv,
    setup_fft_model,
    time_call,
)


def _merge_metric_cache(new_metrics: pd.DataFrame, *, force: bool) -> pd.DataFrame:
    """Preserve cached metric rows unless their dataset/resolution/phase was recomputed.

    Cached result loads return no timing rows. This merge keeps the last real
    benchmark values instead of replacing them with placeholder load events.
    """
    if new_metrics.empty:
        return (
            pd.read_csv(SV_METRICS_PATH) if SV_METRICS_PATH.exists() else pd.DataFrame()
        )
    if force or not SV_METRICS_PATH.exists():
        return new_metrics

    key_cols = ["dataset_key", "resolution", "phase"]
    existing = pd.read_csv(SV_METRICS_PATH)
    replace_keys = new_metrics[key_cols].drop_duplicates().assign(_replace=True)
    kept = existing.merge(replace_keys, on=key_cols, how="left")
    kept = kept[kept["_replace"].isna()].drop(columns="_replace")
    return pd.concat([kept, new_metrics], ignore_index=True, sort=False)


def subset_by_genes(
    adata: AnnData, table_cfg: dict[str, Any], genes: tuple[str, ...] | list[str]
) -> AnnData:
    """Apply the explicit config-level SV gene subset, if any."""
    targets = {normalize_gene_symbol(g) for g in genes}
    var = adata.var
    gene_mask = (
        var[table_cfg["gene_names"]]
        .astype(str)
        .map(normalize_gene_symbol)
        .isin(targets)
        .to_numpy()
    )
    group_mask = (
        var[table_cfg["group_iso_by"]]
        .astype(str)
        .map(normalize_gene_symbol)
        .isin(targets)
        .to_numpy()
    )
    mask = gene_mask | group_mask
    if not mask.any():
        raise ValueError(
            f"None of the configured SV_GENE_SUBSET genes were present: {genes}"
        )
    return adata[:, mask].copy()


def subset_by_isoforms(adata: AnnData, isoforms: list[str]) -> AnnData:
    """Reuse exactly the 16um-retained features at finer resolutions."""
    present = [iso for iso in isoforms if iso in adata.var_names]
    if not present:
        raise ValueError("None of the requested 16um-filtered isoforms are present.")
    return adata[:, present].copy()


def run_sv_one(
    cfg: DatasetConfig,
    resolution: str,
    neighbor_degree: int,
    *,
    force: bool = False,
    n_jobs: int = 8,
    eligible_isoforms: list[str] | None = None,
    gene_subset: tuple[str, ...] | None = SV_GENE_SUBSET,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run or load one SV test table for one dataset/resolution.

    The returned metrics are empty on cache hits; `run_sv_suite()` merges any
    newly measured rows with the persisted benchmark cache.
    """
    result_path = RESULT_DIR / f"sv_{cfg.key}_{resolution}_hsic_ir.csv"
    eligible_path = RESULT_DIR / f"eligible_16um_features_{cfg.key}.csv"
    metric_rows: list[dict[str, Any]] = []

    if result_path.exists() and not force:
        # Cache hits should be transparent: load results but do not synthesize
        # runtime/memory rows that would overwrite the benchmark cache.
        res = pd.read_csv(result_path)
        if "gene_key" not in res.columns:
            res["gene_key"] = res["gene"].map(normalize_gene_symbol)
        features = (
            pd.read_csv(eligible_path) if eligible_path.exists() else pd.DataFrame()
        )
        return res, features, pd.DataFrame()

    sdata = load_sdata(cfg)
    table_cfg = resolve_table_config(cfg, sdata, resolution)
    table_name = table_cfg["table_name"]
    min_counts = table_cfg["min_counts"]
    min_bin_pct = table_cfg["min_bin_pct"]
    subset_config = None

    if eligible_isoforms is not None:
        # Finer-resolution runs inherit the 16um feature filter.
        subset = subset_by_isoforms(sdata.tables[table_name], eligible_isoforms)
        table_name = f"_sv_{cfg.key}_{resolution}_filtered16"
        sdata[table_name] = TableModel.parse(subset)
        min_counts = 0
        min_bin_pct = 0.0
    elif gene_subset is not None:
        # Explicit config subsets are for smoke tests or targeted reruns; a
        # genome-wide final run keeps SV_GENE_SUBSET=None.
        subset = subset_by_genes(sdata.tables[table_name], table_cfg, gene_subset)
        table_name = f"_sv_{cfg.key}_{resolution}_configured_subset"
        sdata[table_name] = TableModel.parse(subset)
        min_counts = 0
        min_bin_pct = 0.0
        subset_config = ";".join(gene_subset)

    model, setup_seconds, setup_rss = time_call(
        f"{cfg.key} {resolution} setup",
        partial(
            setup_fft_model,
            sdata,
            table_cfg,
            table_name=table_name,
            min_counts=min_counts,
            min_bin_pct=min_bin_pct,
            neighbor_degree=neighbor_degree,
        ),
    )
    metric_rows.append(
        {
            "dataset_key": cfg.key,
            "dataset": cfg.short_label,
            "resolution": resolution,
            "phase": "data_prep",
            "seconds": setup_seconds,
            "max_rss_gb": setup_rss,
            "neighbor_degree": neighbor_degree,
            "projected": False,
            "loaded_existing": False,
            "n_spots": int(model.n_spots),
            "n_grid": int(model.n_grid),
            "n_genes": int(model.n_genes),
            "gene_subset": subset_config,
        }
    )

    _, test_seconds, test_rss = time_call(
        f"{cfg.key} {resolution} SV {SV_METHOD}",
        partial(
            model.test_spatial_variability,
            method=SV_METHOD,
            ratio_transformation="none",
            n_jobs=n_jobs,
            chunk_size="auto",
            print_progress=True,
        ),
    )
    metric_rows.append(
        {
            "dataset_key": cfg.key,
            "dataset": cfg.short_label,
            "resolution": resolution,
            "phase": "sv_test",
            "seconds": test_seconds,
            "max_rss_gb": test_rss,
            "neighbor_degree": neighbor_degree,
            "projected": False,
            "loaded_existing": False,
            "n_spots": int(model.n_spots),
            "n_grid": int(model.n_grid),
            "n_genes": int(model.n_genes),
            "gene_subset": subset_config,
        }
    )

    res = model.get_formatted_test_results("sv", with_gene_summary=True).sort_values(
        "pvalue_adj"
    )
    res["dataset_key"] = cfg.key
    res["dataset"] = cfg.short_label
    res["resolution"] = resolution
    save_csv(res, result_path)
    res["gene_key"] = res["gene"].map(normalize_gene_symbol)

    features = model_feature_rows(model, table_cfg)
    if resolution == "16um":
        save_csv(features, eligible_path)

    del model, sdata
    gc.collect()
    return res, features, pd.DataFrame(metric_rows)


def run_sv_suite(
    *, force: bool = False, n_jobs: int = 8
) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    """Run/load SV tests for all datasets/resolutions and cache CSV outputs."""
    degrees = neighbor_degrees_by_res()
    results: dict[tuple[str, str], pd.DataFrame] = {}
    metrics = []
    for cfg in DATASETS.values():
        res16, features16, m16 = run_sv_one(
            cfg,
            "16um",
            degrees[(cfg.key, "16um")],
            force=force,
            n_jobs=n_jobs,
        )
        results[(cfg.key, "16um")] = res16
        metrics.append(m16)
        if features16.empty:
            features16 = pd.read_csv(
                RESULT_DIR / f"eligible_16um_features_{cfg.key}.csv"
            )
        # Finer grids reuse the 16um-filtered feature universe so resolution
        # comparisons are not driven by changing gene eligibility.
        isoforms = features16["isoform"].astype(str).tolist()
        for resolution in ["8um", "2um"]:
            res, _, m = run_sv_one(
                cfg,
                resolution,
                degrees[(cfg.key, resolution)],
                force=force,
                n_jobs=n_jobs,
                eligible_isoforms=isoforms,
            )
            results[(cfg.key, resolution)] = res
            metrics.append(m)
    metric_parts = [metric for metric in metrics if not metric.empty]
    new_metric_df = (
        pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    )
    metric_df = _merge_metric_cache(new_metric_df, force=force)
    if not new_metric_df.empty:
        save_csv(metric_df, SV_METRICS_PATH)
    return results, metric_df


def cross_resolution_comparison(
    sv_results: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Summarize within-platform SV agreement for notebook plotting.

    This is intentionally derived on demand from cached SV result tables rather
    than saved as a workflow artifact.
    """
    rows = []
    for cfg in DATASETS.values():
        for left, right in combinations(["16um", "8um", "2um"], 2):
            a = sv_results.get((cfg.key, left))
            b = sv_results.get((cfg.key, right))
            if a is None or b is None:
                continue
            merged = a[["gene_key", "gene", "pvalue", "pvalue_adj"]].merge(
                b[["gene_key", "pvalue", "pvalue_adj"]],
                on="gene_key",
                suffixes=(f"_{left}", f"_{right}"),
            )
            rho = np.nan
            if merged.shape[0] >= 2:
                rho, _ = spearmanr(
                    minus_log10_p(merged[f"pvalue_{left}"]),
                    minus_log10_p(merged[f"pvalue_{right}"]),
                )
            rows.append(
                {
                    "dataset_key": cfg.key,
                    "dataset": cfg.short_label,
                    "left_resolution": left,
                    "right_resolution": right,
                    "shared_genes": int(merged.shape[0]),
                    "spearman_logp": rho,
                    "left_svp": int((a["pvalue_adj"] < FDR_THRESHOLD).sum()),
                    "right_svp": int((b["pvalue_adj"] < FDR_THRESHOLD).sum()),
                }
            )
    return pd.DataFrame(rows)


def cross_platform_comparison(
    sv_results: dict[tuple[str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build 16um cross-platform summaries for notebook plotting."""
    sv16 = {cfg.key: sv_results.get((cfg.key, "16um")) for cfg in DATASETS.values()}
    sv16 = {key: df for key, df in sv16.items() if df is not None}
    sets = {
        key: set(df.loc[df["pvalue_adj"] < FDR_THRESHOLD, "gene_key"])
        for key, df in sv16.items()
    }
    rows = []
    keys = list(sv16)
    for mask in product([False, True], repeat=len(keys)):
        if not any(mask):
            continue
        in_keys = [keys[i] for i, flag in enumerate(mask) if flag]
        out_keys = [keys[i] for i, flag in enumerate(mask) if not flag]
        genes = set.intersection(*(sets[k] for k in in_keys)) if in_keys else set()
        for out in out_keys:
            genes -= sets[out]
        rows.append(
            {
                "intersection": " & ".join(DATASETS[k].short_label for k in in_keys),
                "n_genes": len(genes),
                "gene_keys": ";".join(sorted(genes)),
            }
        )
    intersections = pd.DataFrame(rows).sort_values("n_genes", ascending=False)

    corr_rows = []
    for left, right in combinations(keys, 2):
        a = sv16[left][
            ["gene_key", "gene", "pvalue", "pvalue_adj", "count_avg"]
        ].rename(
            columns={
                "gene": f"gene_{left}",
                "pvalue": f"pvalue_{left}",
                "pvalue_adj": f"padj_{left}",
                "count_avg": f"expr_{left}",
            }
        )
        b = sv16[right][
            ["gene_key", "gene", "pvalue", "pvalue_adj", "count_avg"]
        ].rename(
            columns={
                "gene": f"gene_{right}",
                "pvalue": f"pvalue_{right}",
                "pvalue_adj": f"padj_{right}",
                "count_avg": f"expr_{right}",
            }
        )
        merged = a.merge(b, on="gene_key", how="inner")
        merged["expr_mean"] = merged[[f"expr_{left}", f"expr_{right}"]].mean(axis=1)
        try:
            merged["expr_quantile"] = (
                pd.qcut(
                    merged["expr_mean"].rank(method="first"),
                    q=min(4, merged.shape[0]),
                    labels=False,
                )
                + 1
            )
        except ValueError:
            merged["expr_quantile"] = 1
        for quantile, group in merged.groupby("expr_quantile"):
            rho = np.nan
            if group.shape[0] >= 2:
                rho, _ = spearmanr(
                    minus_log10_p(group[f"pvalue_{left}"]),
                    minus_log10_p(group[f"pvalue_{right}"]),
                )
            corr_rows.append(
                {
                    "left": DATASETS[left].short_label,
                    "right": DATASETS[right].short_label,
                    "left_key": left,
                    "right_key": right,
                    "expression_quantile": int(quantile),
                    "shared_tested_genes": int(group.shape[0]),
                    "spearman_logp": rho,
                }
            )
    corr = pd.DataFrame(corr_rows)
    return intersections, corr, prioritize_shared_hits(sv16)


def prioritize_shared_hits(
    sv16: dict[str, pd.DataFrame], *, min_platforms: int = 2
) -> pd.DataFrame:
    """Rank shared 16um SVP calls across platforms.

    The saved table is one row per gene and keeps pairwise plus triple shared
    SVP hits. Ranking prioritizes broader platform support, then stronger
    adjusted p-values among the platforms where the gene is called.
    """
    dataset_keys = [cfg.key for cfg in DATASETS.values() if cfg.key in sv16]
    if not dataset_keys:
        return pd.DataFrame()

    rank_tables = []
    for key in dataset_keys:
        df = sv16[key].copy()
        if "gene_key" not in df:
            df["gene_key"] = df["gene"].map(normalize_gene_symbol)
        ranked = (
            df.sort_values("pvalue").drop_duplicates("gene_key").reset_index(drop=True)
        )
        ranked[f"rank_{key}"] = np.arange(1, ranked.shape[0] + 1)
        ranked[f"pvalue_{key}"] = ranked["pvalue"].to_numpy()
        ranked[f"padj_{key}"] = ranked["pvalue_adj"].to_numpy()
        ranked[f"called_{key}"] = ranked["pvalue_adj"].lt(FDR_THRESHOLD).to_numpy()
        ranked[f"gene_{key}"] = ranked["gene"].astype(str)
        rank_tables.append(
            ranked[
                [
                    "gene_key",
                    f"gene_{key}",
                    f"rank_{key}",
                    f"pvalue_{key}",
                    f"padj_{key}",
                    f"called_{key}",
                ]
            ]
        )
    score = rank_tables[0]
    for table in rank_tables[1:]:
        score = score.merge(table, on="gene_key", how="outer")

    called_cols = [f"called_{key}" for key in dataset_keys]
    rank_cols = [c for c in score.columns if c.startswith("rank_")]
    pvalue_cols = [f"pvalue_{key}" for key in dataset_keys]
    padj_cols = [f"padj_{key}" for key in dataset_keys]
    called = score[called_cols].eq(True)
    pvalues = score[pvalue_cols].where(called.to_numpy())
    padj = score[padj_cols].where(called.to_numpy())

    score["n_platforms_tested"] = score[padj_cols].notna().sum(axis=1)
    score["n_platforms_called"] = called.sum(axis=1)
    score["called_platforms"] = [
        ";".join(
            DATASETS[key].short_label for key in dataset_keys if row[f"called_{key}"]
        )
        for _, row in called.iterrows()
    ]
    score["sharing_class"] = score["n_platforms_called"].map(
        {3: "triple", 2: "double", 1: "single", 0: "not_called"}
    )
    score["min_pvalue"] = pvalues.min(axis=1)
    score["max_padj"] = padj.max(axis=1)
    neglogp = pd.DataFrame(index=score.index)
    for key in dataset_keys:
        neglogp[key] = minus_log10_p(score[f"pvalue_{key}"])
        neglogp.loc[~called[f"called_{key}"], key] = np.nan
    score["mean_neglogp"] = neglogp.mean(axis=1)
    score["mean_rank"] = score[rank_cols].mean(axis=1)
    score["called_all_three"] = score["n_platforms_called"].eq(len(dataset_keys))
    score = score[score["n_platforms_called"] >= min_platforms].copy()
    if score.empty:
        return score
    score = score.sort_values(
        ["n_platforms_called", "max_padj", "min_pvalue", "mean_neglogp"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    score.insert(0, "priority_rank", np.arange(1, score.shape[0] + 1))
    return score
