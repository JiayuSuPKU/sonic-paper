"""Graph-based pattern analysis using single-cell lineage tracing data."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats

from scripts.paper_utils import (
    cache_entry_is_current,
    cache_signature,
    path_fingerprint,
    update_cache_entry,
)

from .config import (
    LINEAGE_ANALYSIS_DATA_FILE,
    LINEAGE_BIVARIATE_DIAGNOSTICS_FILE,
    LINEAGE_BIVARIATE_SUMMARY_FILE,
    LINEAGE_CLUSTER_SCORES_FILE,
    LINEAGE_GENE_DIAGNOSTICS_FILE,
    LINEAGE_GENE_SCORES_FILE,
    LINEAGE_QC_FILE,
    LINEAGE_RESULT_DIR,
    LINEAGE_TARGET_METADATA_FILE,
    LINEAGE_TREE_DIR,
)
from .utils import cache_is_reusable, parse_newick, source_provenance

LOGGER = logging.getLogger(__name__)

EXAMPLE_TUMORS = ("3430_NT_T2", "3730_NT_T2")
GENE_PANEL = ("Rgs4", "Col4a2", "Tmem37", "Cd200")
PAIR_GENES = ("Gdpd2", "Spink5")
ANCESTRY_WEIGHTS = (1.0, 0.5, 0.25)
EXAMPLE_TREE_FILES = tuple(LINEAGE_TREE_DIR / f"{tumor}.newick" for tumor in EXAMPLE_TUMORS)
LINEAGE_INPUTS = (LINEAGE_ANALYSIS_DATA_FILE, *EXAMPLE_TREE_FILES)
MIN_TUMOR_CELLS = 300
MAX_GENE_ANALYSIS_CELLS = 3_000
FDR_CUTOFF = 0.05
CAR_RHO = 0.9
CLUSTER_MIN_CELLS = 10
GENE_MIN_CELLS_FRAC = 0.05
BIVARIATE_MIN_PAIRS = 10
BIVARIATE_SIGMA_THRESHOLDS = tuple(range(1, 11))


def _ancestral_connectivity(parent: dict[str, str], leaves: list[str]) -> sparse.csr_matrix:
    """Build sparse leaf connectivity from the nearest shared ancestor."""
    matrix = sparse.csr_matrix((len(leaves), len(leaves)), dtype=np.float64)
    seen = sparse.csr_matrix(matrix.shape, dtype=bool)
    for depth, weight in enumerate(ANCESTRY_WEIGHTS, start=1):
        groups: dict[str, list[int]] = defaultdict(list)
        for index, leaf in enumerate(leaves):
            ancestor = leaf
            for _ in range(depth):
                ancestor = parent.get(ancestor, "")
                if not ancestor:
                    break
            if ancestor:
                groups[ancestor].append(index)
        rows = [index for indices in groups.values() for index in indices]
        columns = [group for group, indices in enumerate(groups.values()) for _ in indices]
        membership = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, columns)),
            shape=(len(leaves), len(groups)),
        )
        same_ancestor = (membership @ membership.T).astype(bool)
        same_ancestor.setdiag(False)
        same_ancestor.eliminate_zeros()
        new_edges = same_ancestor - same_ancestor.multiply(seen)
        matrix += weight * new_edges
        seen = seen.maximum(same_ancestor)
    return matrix.tocsr()


def _moran_i(q: pd.Series, detector) -> pd.Series:
    """Conventional Moran I for the tested symmetric degree-normalized weights.

    SONIC reports Q=z'W_norm z using sample-standardized z, so
    I = n Q / ((n-1) sum(W_norm)) on the retained observations.
    """
    weights = detector.adata.obsp["lineage"]
    inverse = 1 / np.sqrt(np.asarray(weights.sum(axis=1)).ravel())
    weight_sum = float(inverse @ (weights @ inverse))
    return q * detector.n / ((detector.n - 1) * weight_sum)


def _run_lineage_q_tests(
    data: ad.AnnData,
    connectivity: sparse.csr_matrix,
    *,
    source: str,
    features: list[str] | None = None,
    min_cells: int = 1,
    min_cells_frac: float | None = None,
    detectors: dict[str, Any] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Reuse fitted operators while changing the observation/gene support filter."""
    from sonic import DetectorIrregular

    if detectors is None:
        data.obsp["lineage"] = connectivity
        detectors = {}
        for method, parameters in (
            ("moran", {}),
            ("car", {"rho": CAR_RHO, "standardize": False}),
        ):
            detector = DetectorIrregular(kernel_method=method, backend="matrix", **parameters)
            detector.setup_data(data, obsp_key="lineage", is_distance=False)
            detectors[method] = detector
    results = {}
    for method, detector in detectors.items():
        detector.min_cells = (
            max(1, int(min_cells_frac * detector.n))
            if min_cells_frac is not None
            else max(1, min(min_cells, detector.n))
        )
        results[method] = detector.compute_qstat(
            source=source,
            features=features,
            n_jobs=1,
            show_progress=False,
        )
    return results, detectors


def _merge_q_results(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align Moran and CAR outputs with explicit method suffixes."""
    return (
        results["moran"].add_suffix("_moran").join(results["car"].add_suffix("_car"), how="inner")
    )


def _bivariate_disagreement(
    tumor: str,
    detectors: dict[str, Any],
    gene_scores: pd.DataFrame,
) -> list[dict[str, object]]:
    """Summarize method disagreement among genes detected by both Q tests."""
    selected = sorted(gene_scores.index[gene_scores["Sig_moran"] & gene_scores["Sig_car"]])
    if len(selected) < 2:
        LOGGER.info("Lineage %s: skipping R statistics (<2 shared Q hits)", tumor)
        return []
    tables = []
    for method, column in (("moran", "Z_Moran"), ("car", "Z_CAR")):
        table = detectors[method].compute_rstat(
            features_x=selected,
            n_jobs=1,
            return_pval=True,
            show_progress=False,
        )
        tables.append(_canonical_pairs(table, column))
        del table
    pairs = tables[0].join(tables[1], how="inner")
    del tables
    if len(pairs) < BIVARIATE_MIN_PAIRS:
        LOGGER.info(
            "Lineage %s: skipping R statistics (<%d gene pairs)", tumor, BIVARIATE_MIN_PAIRS
        )
        return []

    regression = _pair_regression(pairs)
    deviation = pairs["Deviation_Z"]
    if tumor == "3730_NT_T2":
        _write_bivariate_diagnostics(detectors["moran"], selected, pairs, regression)
    records = []
    for threshold in BIVARIATE_SIGMA_THRESHOLDS:
        for group, mask in (
            ("Moran dominant", deviation < -threshold),
            ("CAR dominant", deviation > threshold),
        ):
            records.append(
                {
                    "Tumor": tumor,
                    "Sigma": threshold,
                    "Group": group,
                    "Proportion": float(mask.mean()),
                }
            )
    return records


def _cache_cross_tumor_statistics(*, force: bool) -> None:  # noqa: C901
    """Compute the cross-tumor Q/R tables used by the lineage figures."""
    outputs = (
        LINEAGE_CLUSTER_SCORES_FILE,
        LINEAGE_GENE_SCORES_FILE,
        LINEAGE_BIVARIATE_SUMMARY_FILE,
        LINEAGE_QC_FILE,
        LINEAGE_GENE_DIAGNOSTICS_FILE,
        LINEAGE_BIVARIATE_DIAGNOSTICS_FILE,
    )
    tree_files = tuple(sorted(LINEAGE_TREE_DIR.glob("*.newick")))
    if cache_is_reusable(
        *outputs,
        force=force,
        inputs=(LINEAGE_ANALYSIS_DATA_FILE, *tree_files),
        logger=LOGGER,
        label="Lineage cross-tumor statistics",
    ):
        return
    if not LINEAGE_ANALYSIS_DATA_FILE.exists():
        raise FileNotFoundError(f"Missing lineage input: {LINEAGE_ANALYSIS_DATA_FILE}")
    if not LINEAGE_TREE_DIR.exists():
        raise FileNotFoundError(f"Missing lineage input: {LINEAGE_TREE_DIR}")

    if not tree_files:
        raise FileNotFoundError(f"No Newick trees found in {LINEAGE_TREE_DIR}")

    source = ad.read_h5ad(LINEAGE_ANALYSIS_DATA_FILE, backed="r")
    cluster_scores = []
    gene_scores = []
    bivariate_summaries = []
    qc_rows = []
    diagnostic_tumors = set()
    LINEAGE_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    def record_qc(tumor, n_input, feature_type, results, detectors):
        for method, table in results.items():
            detector = detectors[method]
            qc_rows.append(
                {
                    "Tumor": tumor,
                    "feature_type": feature_type,
                    "method": method,
                    "n_input": n_input,
                    "n_retained": detector.n,
                    "min_cells": detector.min_cells,
                    "n_eligible": len(table),
                    "n_discoveries": int((table["P_adj"] <= FDR_CUTOFF).sum()),
                }
            )

    try:
        trees = []
        for path in tree_files:
            _, _, parent, leaves = parse_newick(path)
            if len(leaves) < MIN_TUMOR_CELLS:
                continue
            missing = pd.Index(leaves).difference(source.obs_names)
            if len(missing):
                raise ValueError(
                    f"{path.stem}: {len(missing)} tree leaves are absent from local data"
                )
            trees.append((path.stem, parent, leaves))
        if not trees:
            raise ValueError(f"No lineage trees with at least {MIN_TUMOR_CELLS} cells")

        for index, (tumor, parent, leaves) in enumerate(trees, start=1):
            LOGGER.info("Lineage %s: testing tumor %d/%d", tumor, index, len(trees))
            tumor_data = source[leaves, :].to_memory()
            connectivity = _ancestral_connectivity(parent, leaves)

            # Cluster indicators use a fixed ten-cell support threshold.
            cluster_results, detectors = _run_lineage_q_tests(
                tumor_data,
                connectivity,
                source="obs",
                features=["Cluster-Name"],
                min_cells=CLUSTER_MIN_CELLS,
            )
            record_qc(tumor, tumor_data.n_obs, "cluster", cluster_results, detectors)
            clusters = _merge_q_results(cluster_results)
            clusters["Tumor"] = tumor
            clusters["Cluster"] = clusters.index.str.replace("Cluster-Name_", "", regex=False)
            clusters["I_moran"] = _moran_i(clusters["Q_moran"], detectors["moran"])
            cluster_scores.append(clusters[["Tumor", "Cluster", "I_moran", "Q_car", "Z_score_car"]])

            if tumor_data.n_obs > MAX_GENE_ANALYSIS_CELLS:
                LOGGER.info(
                    "Lineage %s: skipping gene/R tests (%d cells exceed %d-cell cap)",
                    tumor,
                    tumor_data.n_obs,
                    MAX_GENE_ANALYSIS_CELLS,
                )
                del detectors, tumor_data
                continue

            # Gene tests use the original 5% expression-support filter.
            gene_results, gene_detectors = _run_lineage_q_tests(
                tumor_data,
                connectivity,
                source="var",
                min_cells_frac=GENE_MIN_CELLS_FRAC,
                detectors=detectors,
            )
            record_qc(tumor, tumor_data.n_obs, "gene", gene_results, gene_detectors)
            genes = _merge_q_results(gene_results)
            genes["Tumor"] = tumor
            genes["I_moran"] = _moran_i(genes["Q_moran"], gene_detectors["moran"])
            genes["Sig_moran"] = genes["P_adj_moran"] <= FDR_CUTOFF
            genes["Sig_car"] = genes["P_adj_car"] <= FDR_CUTOFF
            # SONIC removes zero-degree leaves before testing; measure support
            # on that same analyzed cell set, as in the original notebook.
            tested = gene_detectors["car"].adata
            columns = tested.var_names.get_indexer(genes.index)
            if np.any(columns < 0):
                raise ValueError("Q-statistic output contains genes absent from local data")
            values = tested.X[:, columns]
            genes["Sparsity"] = (
                1 - np.asarray((values != 0).sum(axis=0)).ravel() / tested.n_obs
                if sparse.issparse(values)
                else np.mean(np.asarray(values) == 0, axis=0)
            )
            gene_scores.append(
                genes.assign(Feature=genes.index)[
                    [
                        "Feature",
                        "Tumor",
                        "I_moran",
                        "Z_score_moran",
                        "Z_score_car",
                        "Sig_moran",
                        "Sig_car",
                        "Sparsity",
                    ]
                ]
            )
            if tumor == "3430_NT_T2":
                _write_gene_diagnostics(gene_detectors["moran"], genes)
                diagnostic_tumors.add(tumor)
            summaries = _bivariate_disagreement(tumor, gene_detectors, genes)
            bivariate_summaries.extend(summaries)
            if tumor == "3730_NT_T2" and summaries:
                diagnostic_tumors.add(tumor)
            # Release fitted operators and expression views before the next tumor.
            del detectors, gene_detectors, tumor_data, tested, values
    finally:
        source.file.close()

    if diagnostic_tumors != set(EXAMPLE_TUMORS):
        raise ValueError(
            f"Missing eligible lineage diagnostics for {set(EXAMPLE_TUMORS) - diagnostic_tumors}"
        )
    pd.DataFrame(qc_rows).to_csv(LINEAGE_QC_FILE, index=False)
    pd.concat(cluster_scores, ignore_index=True).to_csv(LINEAGE_CLUSTER_SCORES_FILE, index=False)
    pd.concat(gene_scores, ignore_index=True).to_csv(LINEAGE_GENE_SCORES_FILE, index=False)
    pd.DataFrame(bivariate_summaries, columns=["Tumor", "Sigma", "Group", "Proportion"]).to_csv(
        LINEAGE_BIVARIATE_SUMMARY_FILE, index=False
    )
    LOGGER.info(
        "Lineage cross-tumor statistics: cached %d cluster and %d gene tumors",
        len(cluster_scores),
        len(gene_scores),
    )


def _write_gene_diagnostics(detector, scores: pd.DataFrame) -> None:
    """Decompose the tested Moran operator on exactly its retained cells."""
    features = pd.Index(scores.index)
    target = detector.adata[:, features]
    normalized = detector.kernel_.realization()
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    expression = target.X
    expression = expression.toarray() if sparse.issparse(expression) else np.asarray(expression)
    marker_expression = expression.astype(np.float64, copy=False)
    expression = marker_expression.copy()
    expression -= expression.mean(axis=0)
    standard_deviation = expression.std(axis=0, ddof=1)
    if np.any(standard_deviation == 0):
        raise ValueError("Constant genes remain in the lineage score table")
    expression /= standard_deviation
    coefficients = eigenvectors.T @ expression
    contributions = eigenvalues[:, None] * coefficients**2
    positive = np.clip(contributions, 0, None).sum(axis=0)
    negative = -np.clip(contributions, None, 0).sum(axis=0)
    total = positive + negative
    cancellation = np.divide(
        2 * np.minimum(positive, negative),
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )
    diagnostics = scores.assign(Feature=scores.index)[
        ["Feature", "Sig_moran", "Sig_car", "Sparsity"]
    ].copy()
    diagnostics["Positive_contribution"] = positive
    diagnostics["Negative_contribution"] = negative
    diagnostics["Cancellation_fraction"] = cancellation
    diagnostics["Call"] = np.select(
        (
            diagnostics["Sig_moran"] & diagnostics["Sig_car"],
            ~diagnostics["Sig_moran"] & diagnostics["Sig_car"],
            diagnostics["Sig_moran"] & ~diagnostics["Sig_car"],
        ),
        ("Both", "CAR only", "Moran only"),
        default="Neither",
    )
    labels = target.obs["Cluster-Name"].astype(str).to_numpy()
    counts = pd.Series(labels).value_counts()
    small_states = counts[(counts >= 2) & (counts <= max(2, np.ceil(0.1 * len(labels))))].index
    markers = np.zeros(expression.shape[1], dtype=bool)
    for state in small_states:
        inside = labels == state
        test = stats.ttest_ind(
            marker_expression[inside],
            marker_expression[~inside],
            axis=0,
            equal_var=False,
            alternative="greater",
        )
        p_values = np.nan_to_num(test.pvalue, nan=1.0)
        markers |= stats.false_discovery_control(p_values, method="bh") <= FDR_CUTOFF
    diagnostics["Small_state_marker"] = markers

    example_gene = "Rgs4"
    example_index = features.get_loc(example_gene)
    np.savez_compressed(
        LINEAGE_GENE_DIAGNOSTICS_FILE,
        features=diagnostics["Feature"].to_numpy(str),
        sparsity=diagnostics["Sparsity"].to_numpy(),
        positive_contribution=positive,
        negative_contribution=negative,
        cancellation_fraction=cancellation,
        calls=diagnostics["Call"].to_numpy(str),
        small_state_markers=markers,
        example_gene=np.asarray(example_gene),
        eigenvalues=eigenvalues,
        moran_contributions=eigenvalues * coefficients[:, example_index] ** 2,
        effective_support=1 / (eigenvectors**4).sum(axis=0),
    )
    LOGGER.info("Lineage gene cancellation: cached %d genes", len(diagnostics))


def _cache_target_metadata(*, force: bool) -> None:
    if cache_is_reusable(
        LINEAGE_TARGET_METADATA_FILE,
        force=force,
        inputs=LINEAGE_INPUTS,
        logger=LOGGER,
        label="Lineage target metadata",
    ):
        return
    genes = GENE_PANEL + PAIR_GENES
    leaves = [leaf for path in EXAMPLE_TREE_FILES for leaf in parse_newick(path)[3]]
    source = ad.read_h5ad(LINEAGE_ANALYSIS_DATA_FILE, backed="r")
    try:
        missing = [gene for gene in genes if gene not in source.var_names]
        if missing:
            raise ValueError(f"Missing lineage example genes: {missing}")
        target = source[leaves, list(genes)].to_memory()
    finally:
        source.file.close()
    values = target[:, list(genes)].X
    values = values.toarray() if sparse.issparse(values) else np.asarray(values)
    metadata = target.obs[["Tumor", "Cluster-Name"]].copy()
    metadata.index.name = "Cell"
    for column, gene in enumerate(genes):
        metadata[gene] = values[:, column]
    metadata.reset_index().to_csv(LINEAGE_TARGET_METADATA_FILE, index=False)
    LOGGER.info("Lineage target metadata: cached %d cells", len(metadata))


def _canonical_pairs(table: pd.DataFrame, value_name: str) -> pd.DataFrame:
    first = table["Feature_1"].astype(str).to_numpy()
    second = table["Feature_2"].astype(str).to_numpy()
    result = pd.DataFrame(
        {
            "Gene_1": np.minimum(first, second),
            "Gene_2": np.maximum(first, second),
            value_name: table["Z_score"].to_numpy(),
        }
    )
    result = result[(first != second) & np.isfinite(result[value_name])]
    return result.drop_duplicates(("Gene_1", "Gene_2")).set_index(["Gene_1", "Gene_2"])


def _pair_regression(pairs: pd.DataFrame):
    """Fit finite cross-pair scores; an exact fit has zero residual outliers."""
    values = pairs[["Z_Moran", "Z_CAR"]].to_numpy()
    if len(pairs) < 2 or not np.isfinite(values).all() or np.ptp(values[:, 0]) == 0:
        raise ValueError("R-score regression needs finite, varying cross-pair scores")
    regression = stats.linregress(values[:, 0], values[:, 1])
    residual = values[:, 1] - (regression.intercept + regression.slope * values[:, 0])
    residual_sd = residual.std(ddof=1)
    if not np.isfinite(residual_sd):
        raise ValueError("R-score regression has non-finite residual variance")
    tolerance = 32 * np.finfo(float).eps * max(1.0, np.max(np.abs(values)))
    pairs["Deviation_Z"] = residual / residual_sd if residual_sd > tolerance else 0.0
    pairs["Group"] = np.select(
        (pairs["Deviation_Z"] < -3, pairs["Deviation_Z"] > 3),
        ("Moran dominant", "CAR dominant"),
        default="Consistent",
    )
    return regression


def _write_bivariate_diagnostics(detector, selected, pairs, regression) -> None:
    """Save the already-computed R results and a matching example decomposition."""
    missing = set(PAIR_GENES).difference(selected)
    if missing:
        raise ValueError(f"Configured example genes are not shared Q hits: {sorted(missing)}")
    upper = np.triu_indices(len(selected), k=1)
    names = np.asarray(selected)
    index = pd.MultiIndex.from_arrays([names[upper[0]], names[upper[1]]], names=pairs.index.names)
    ordered = pairs.reindex(index)
    if ordered[["Z_Moran", "Z_CAR"]].isna().any().any():
        raise ValueError("Missing cross-pair scores in the lineage diagnostics")
    codes = ordered["Group"].map({"Consistent": 0, "Moran dominant": 1, "CAR dominant": 2})
    # Decompose the selected example pair to expose modes with opposing signed
    # contributions to Moran's relational statistic.
    normalized = detector.kernel_.realization()
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    expression = detector.adata[:, list(PAIR_GENES)].X
    expression = expression.toarray() if sparse.issparse(expression) else np.asarray(expression)
    expression = stats.zscore(np.asarray(expression, dtype=float), axis=0, ddof=1)
    coefficients = eigenvectors.T @ expression
    contributions = coefficients[:, 0] * coefficients[:, 1] * eigenvalues
    positive_mode = int(np.argmax(contributions))
    negative_mode = int(np.argmin(contributions))
    np.savez_compressed(
        LINEAGE_BIVARIATE_DIAGNOSTICS_FILE,
        genes=np.asarray(selected),
        z_moran=ordered["Z_Moran"].to_numpy(np.float32),
        z_car=ordered["Z_CAR"].to_numpy(np.float32),
        group=codes.to_numpy(np.int8),
        regression=np.asarray((regression.slope, regression.intercept, regression.rvalue**2)),
        cells=detector.adata.obs_names.to_numpy(str),
        pair_genes=np.asarray(PAIR_GENES),
        pair_expression=expression,
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        contributions=contributions,
        selected_modes=np.asarray((positive_mode, negative_mode)),
        selected_vectors=eigenvectors[:, (positive_mode, negative_mode)],
    )
    LOGGER.info("Lineage bivariate diagnostics: cached %d gene pairs", len(pairs))


def cache_lineage_results(
    *,
    force: bool = False,
) -> None:
    """Compute every plot-ready cache from the repository's lineage data."""
    outputs = (
        LINEAGE_CLUSTER_SCORES_FILE,
        LINEAGE_GENE_SCORES_FILE,
        LINEAGE_BIVARIATE_SUMMARY_FILE,
        LINEAGE_GENE_DIAGNOSTICS_FILE,
        LINEAGE_TARGET_METADATA_FILE,
        LINEAGE_BIVARIATE_DIAGNOSTICS_FILE,
        LINEAGE_QC_FILE,
    )
    signature_data = {
        "source": source_provenance(__file__),
        "inputs": [
            path_fingerprint(LINEAGE_ANALYSIS_DATA_FILE),
            path_fingerprint(LINEAGE_TREE_DIR, recursive=True),
        ],
    }
    signature = cache_signature(signature_data)
    force = force or not cache_entry_is_current(
        LINEAGE_RESULT_DIR, "lineage", signature, outputs=outputs
    )
    _cache_cross_tumor_statistics(force=force)
    _cache_target_metadata(force=force)
    if force:
        update_cache_entry(
            LINEAGE_RESULT_DIR,
            "lineage",
            signature,
            outputs,
            signature_data=signature_data,
        )
    LOGGER.info("Lineage analysis caches are ready")
