"""Graph-based pattern analysis using single-cell lineage tracing data."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats

from .config import (
    LINEAGE_ANALYSIS_DATA_FILE,
    LINEAGE_BIVARIATE_DIAGNOSTICS_FILE,
    LINEAGE_BIVARIATE_SUMMARY_FILE,
    LINEAGE_CLUSTER_SCORES_FILE,
    LINEAGE_GENE_DIAGNOSTICS_FILE,
    LINEAGE_GENE_SCORES_FILE,
    LINEAGE_RESULT_DIR,
    LINEAGE_TARGET_METADATA_FILE,
    LINEAGE_TREE_DIR,
)
from .utils import cache_is_reusable, parse_newick

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


def _normalized_adjacency(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric degree-normalized graph operator."""
    symmetric = 0.5 * (matrix + matrix.T)
    degrees = symmetric.sum(axis=1)
    inverse = np.zeros_like(degrees)
    inverse[degrees > 0] = 1 / np.sqrt(degrees[degrees > 0])
    return inverse[:, None] * symmetric * inverse[None, :]


def _run_lineage_q_tests(
    data: ad.AnnData,
    connectivity: sparse.csr_matrix,
    *,
    source: str,
    features: list[str] | None = None,
    min_cells: int = 1,
    min_cells_frac: float | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Run Moran and CAR tests on the same lineage graph and observations."""
    from sonic import DetectorIrregular

    data.obsp["lineage"] = connectivity
    results: dict[str, pd.DataFrame] = {}
    detectors = {}
    for method, parameters in (
        ("moran", {}),
        ("car", {"rho": CAR_RHO, "standardize": False}),
    ):
        detector = DetectorIrregular(kernel_method=method, backend="matrix", **parameters)
        detector.setup_data(
            data,
            obsp_key="lineage",
            is_distance=False,
            min_cells=min_cells,
            min_cells_frac=min_cells_frac,
        )
        results[method] = detector.compute_qstat(
            source=source,
            features=features,
            n_jobs=1,
            show_progress=False,
        )
        detectors[method] = detector
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
    r_moran = detectors["moran"].compute_rstat(
        features_x=selected,
        n_jobs=1,
        return_pval=True,
        show_progress=False,
    )
    r_car = detectors["car"].compute_rstat(
        features_x=selected,
        n_jobs=1,
        return_pval=True,
        show_progress=False,
    )
    pairs = _canonical_pairs(r_moran, "Z_Moran").join(_canonical_pairs(r_car, "Z_CAR"), how="inner")
    if len(pairs) < BIVARIATE_MIN_PAIRS:
        LOGGER.info("Lineage %s: skipping R statistics (<10 gene pairs)", tumor)
        return []

    regression = stats.linregress(pairs["Z_Moran"], pairs["Z_CAR"])
    residual = pairs["Z_CAR"] - (regression.intercept + regression.slope * pairs["Z_Moran"])
    deviation = residual / residual.std()
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
            cluster_results, _ = _run_lineage_q_tests(
                tumor_data,
                connectivity,
                source="obs",
                features=["Cluster-Name"],
                min_cells=CLUSTER_MIN_CELLS,
            )
            clusters = _merge_q_results(cluster_results)
            clusters["Tumor"] = tumor
            clusters["Cluster"] = clusters.index.str.replace("Cluster-Name_", "", regex=False)
            clusters["I_moran"] = clusters["Q_moran"] / tumor_data.n_obs
            cluster_scores.append(clusters[["Tumor", "Cluster", "I_moran", "Q_car", "Z_score_car"]])

            if tumor_data.n_obs > MAX_GENE_ANALYSIS_CELLS:
                LOGGER.info(
                    "Lineage %s: skipping gene/R tests (%d cells exceed %d-cell cap)",
                    tumor,
                    tumor_data.n_obs,
                    MAX_GENE_ANALYSIS_CELLS,
                )
                continue

            # Gene tests use the original 5% expression-support filter.
            gene_results, gene_detectors = _run_lineage_q_tests(
                tumor_data,
                connectivity,
                source="var",
                min_cells_frac=GENE_MIN_CELLS_FRAC,
            )
            genes = _merge_q_results(gene_results)
            genes["Tumor"] = tumor
            genes["I_moran"] = genes["Q_moran"] / tumor_data.n_obs
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
                1 - values.getnnz(axis=0) / tested.n_obs
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
            bivariate_summaries.extend(_bivariate_disagreement(tumor, gene_detectors, genes))
    finally:
        source.file.close()

    LINEAGE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(cluster_scores, ignore_index=True).to_csv(LINEAGE_CLUSTER_SCORES_FILE, index=False)
    pd.concat(gene_scores, ignore_index=True).to_csv(LINEAGE_GENE_SCORES_FILE, index=False)
    pd.DataFrame(bivariate_summaries).to_csv(LINEAGE_BIVARIATE_SUMMARY_FILE, index=False)
    LOGGER.info(
        "Lineage cross-tumor statistics: cached %d cluster and %d gene tumors",
        len(cluster_scores),
        len(gene_scores),
    )


def _cache_gene_diagnostics(*, force: bool) -> None:
    if cache_is_reusable(
        LINEAGE_GENE_DIAGNOSTICS_FILE,
        force=force,
        inputs=(LINEAGE_GENE_SCORES_FILE, *LINEAGE_INPUTS),
        logger=LOGGER,
        label="Lineage gene cancellation",
    ):
        return

    tumor = "3430_NT_T2"
    scores = pd.read_csv(LINEAGE_GENE_SCORES_FILE)
    scores = scores[scores["Tumor"].eq(tumor)].copy()
    _, _, parent, leaves = parse_newick(LINEAGE_TREE_DIR / f"{tumor}.newick")
    connectivity = _ancestral_connectivity(parent, leaves)
    keep = np.asarray(connectivity.sum(axis=1)).ravel() > 0
    normalized = _normalized_adjacency(connectivity[keep][:, keep].toarray())
    # Project standardized gene expression onto the finite Moran operator's
    # eigenvectors so positive and negative modal contributions can be separated.
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    features = pd.Index(scores["Feature"])
    source = ad.read_h5ad(LINEAGE_ANALYSIS_DATA_FILE, backed="r")
    try:
        missing = features.difference(source.var_names)
        if len(missing):
            raise ValueError(f"{len(missing)} scored lineage genes are absent from local data")
        target = source[leaves, features].to_memory()
    finally:
        source.file.close()
    expression = target.X[keep]
    expression = expression.toarray() if sparse.issparse(expression) else np.asarray(expression)
    marker_expression = expression.astype(np.float64, copy=False)
    expression = marker_expression.copy()
    expression -= expression.mean(axis=0)
    standard_deviation = expression.std(axis=0)
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
    diagnostics = scores[["Feature", "Sig_moran", "Sig_car", "Sparsity"]].copy()
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
    labels = target.obs.iloc[np.flatnonzero(keep)]["Cluster-Name"].astype(str).to_numpy()
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
    return result.drop_duplicates(("Gene_1", "Gene_2")).set_index(["Gene_1", "Gene_2"])


def _cache_bivariate_diagnostics(*, force: bool) -> None:  # noqa: C901
    if cache_is_reusable(
        LINEAGE_BIVARIATE_DIAGNOSTICS_FILE,
        force=force,
        inputs=(
            LINEAGE_GENE_SCORES_FILE,
            LINEAGE_ANALYSIS_DATA_FILE,
            LINEAGE_TREE_DIR / "3730_NT_T2.newick",
        ),
        logger=LOGGER,
        label="Lineage bivariate diagnostics",
    ):
        return
    from sonic import DetectorIrregular

    tumor = "3730_NT_T2"
    scores = pd.read_csv(LINEAGE_GENE_SCORES_FILE)
    scores = scores[scores["Tumor"].eq(tumor)]
    selected = sorted(scores.loc[scores["Sig_moran"] & scores["Sig_car"], "Feature"].astype(str))
    if len(selected) < 2:
        raise ValueError("Fewer than two genes are significant under both lineage kernels")
    _, _, parent, leaves = parse_newick(LINEAGE_TREE_DIR / f"{tumor}.newick")
    source = ad.read_h5ad(LINEAGE_ANALYSIS_DATA_FILE, backed="r")
    try:
        tumor_data = source[leaves, selected].to_memory()
    finally:
        source.file.close()
    connectivity = _ancestral_connectivity(parent, leaves)
    tumor_data.obsp["lineage"] = connectivity

    # Reuse the cross-tumor Q calls and compute only the target R statistics.
    detectors = {}
    for method, params in (
        ("moran", {}),
        ("car", {"rho": CAR_RHO, "standardize": False}),
    ):
        detector = DetectorIrregular(kernel_method=method, backend="matrix", **params)
        detector.setup_data(
            tumor_data.copy(),
            obsp_key="lineage",
            is_distance=False,
            min_cells_frac=GENE_MIN_CELLS_FRAC,
        )
        detectors[method] = detector
    r_moran = detectors["moran"].compute_rstat(
        features_x=selected, n_jobs=1, return_pval=True, show_progress=False
    )
    r_car = detectors["car"].compute_rstat(
        features_x=selected, n_jobs=1, return_pval=True, show_progress=False
    )
    pairs = _canonical_pairs(r_moran, "Z_Moran").join(_canonical_pairs(r_car, "Z_CAR"), how="inner")
    regression = stats.linregress(pairs["Z_Moran"], pairs["Z_CAR"])
    pairs["Predicted"] = regression.intercept + regression.slope * pairs["Z_Moran"]
    pairs["Residual"] = pairs["Z_CAR"] - pairs["Predicted"]
    residual_sd = pairs["Residual"].std()
    pairs["Deviation_Z"] = pairs["Residual"] / residual_sd
    pairs["Group"] = np.select(
        (pairs["Deviation_Z"] < -3, pairs["Deviation_Z"] > 3),
        ("Moran dominant", "CAR dominant"),
        default="Consistent",
    )
    # Store symmetric pair results as compact upper triangles for plotting.
    gene_index = {gene: index for index, gene in enumerate(selected)}
    matrix_moran = np.zeros((len(selected), len(selected)), dtype=np.float32)
    matrix_car = np.zeros_like(matrix_moran)
    group_codes = np.zeros_like(matrix_moran, dtype=np.int8)
    code_for_group = {"Consistent": 0, "Moran dominant": 1, "CAR dominant": 2}
    for (gene_1, gene_2), row in pairs.iterrows():
        i, j = gene_index[gene_1], gene_index[gene_2]
        matrix_moran[i, j] = matrix_moran[j, i] = row["Z_Moran"]
        matrix_car[i, j] = matrix_car[j, i] = row["Z_CAR"]
        group_codes[i, j] = group_codes[j, i] = code_for_group[row["Group"]]
    upper = np.triu_indices(len(selected))
    # Decompose the selected example pair to expose modes with opposing signed
    # contributions to Moran's relational statistic.
    normalized = _normalized_adjacency(connectivity.toarray())
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    expression = tumor_data[:, list(PAIR_GENES)].X
    expression = expression.toarray() if sparse.issparse(expression) else np.asarray(expression)
    expression = stats.zscore(expression, axis=0)
    coefficients = eigenvectors.T @ expression
    contributions = coefficients[:, 0] * coefficients[:, 1] * eigenvalues
    positive_mode = int(np.argmax(contributions[1:]) + 1)
    negative_mode = int(np.argmin(contributions[1:]) + 1)
    np.savez_compressed(
        LINEAGE_BIVARIATE_DIAGNOSTICS_FILE,
        genes=np.asarray(selected),
        z_moran=matrix_moran[upper],
        z_car=matrix_car[upper],
        group=group_codes[upper],
        regression=np.asarray((regression.slope, regression.intercept, regression.rvalue**2)),
        cells=np.asarray(leaves),
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
    _cache_cross_tumor_statistics(force=force)
    _cache_gene_diagnostics(force=force)
    _cache_target_metadata(force=force)
    _cache_bivariate_diagnostics(force=force)
    LOGGER.info("Lineage analysis caches are ready")
