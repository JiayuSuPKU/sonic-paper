"""Compute and cache metrics for the joint embryo spectral embedding."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import hypergeom
from sklearn.cluster import MiniBatchKMeans

from scripts.paper_utils import (
    CLUSTERPROFILER_SCRIPT,
    cache_entry_is_current,
    cache_signature,
    run_clusterprofiler,
    update_cache_entry,
)

from . import config as C
from . import orthology, sv, utils

LOGGER = logging.getLogger("embryo_xspecies.metrics")

# Unified cache location and PCA coordinate contract.
RESULT_DIR = C.COMPARISON_RESULT_DIR / "metrics"
PC_COLUMNS = [f"PC{index}" for index in range(1, C.N_PC + 1)]
_CACHE_ARTIFACTS = (
    "cluster_go_enrichment.csv.gz",
    "cluster_go_queries.csv.gz",
    "cluster_go_queries_manifest.json",
    "cluster_summary.csv",
    "gene_stage_cluster_membership.csv.gz",
    "local_mixing_by_section.csv",
    "manifest.json",
)

GO_COLUMNS = [
    "query_group",
    "source",
    "native",
    "name",
    "display_name",
    "p_value",
    "p_raw",
    "p_adjust",
    "q_value",
    "significant",
    "description",
    "term_size",
    "query_size",
    "intersection_size",
    "effective_domain_size",
    "precision",
    "recall",
    "query",
    "parents",
    "geneID",
    "hit_entrez",
    "hit_genes",
    "semantic_simplified",
    "semantic_simplify_cutoff",
    "semantic_simplify_measure",
    "method",
]

# Concise figure labels curated from GO terms significantly enriched in both
# species. The supporting IDs are validated on every run to prevent stale labels.
CLUSTER_FUNCTIONAL_LABELS = {
    1: ("Muscle + tissue morphogenesis", ("GO:0060537", "GO:0048729")),
    2: ("Axon + tissue development", ("GO:0061564", "GO:0048729")),
    3: ("Neural junctions + bioenergetics", ("GO:0034329", "GO:0006119")),
    4: ("Chromosome segregation", ("GO:0098813",)),
    5: ("Lipid + vesicle metabolism", ("GO:0008610", "GO:0007034")),
}


# ---------------------------------------------------------------------------
# Shared GO enrichment
# ---------------------------------------------------------------------------
def cached_go_enrichment(  # noqa: C901
    queries: Mapping[str, tuple[Sequence[str], Sequence[str], str]],
    *,
    cache_name: str,
    panel: str,
    variant: str,
    result_dir: Path = RESULT_DIR,
    fdr_groups: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Cache GO tests, optionally pooling BH correction across named queries.

    ``fdr_groups`` maps every query to its correction family. Grouped results
    report only ``global_qvalue`` and use it for semantic simplification.
    Ungrouped temporal/transition queries retain their per-query BH correction.
    """
    wrapper = CLUSTERPROFILER_SCRIPT
    if not wrapper.exists():
        raise FileNotFoundError(f"Missing clusterProfiler wrapper: {wrapper}")
    normalized = {}
    for name, (genes, universe, orgdb) in queries.items():
        query_genes = sorted(set(map(str, genes)))
        universe_genes = sorted(set(map(str, universe)))
        if not set(query_genes).issubset(universe_genes):
            raise ValueError(f"GO query {name!r} contains genes outside its universe.")
        normalized[name] = (query_genes, universe_genes, orgdb)

    if fdr_groups is not None and (
        set(fdr_groups) != set(normalized)
        or any(not isinstance(group, str) or not group.strip() for group in fdr_groups.values())
    ):
        raise ValueError("fdr_groups must assign every GO query to a nonempty correction family.")

    cache_config = {
        "panel": panel,
        "variant": variant,
        "wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        "semantic_simplify": C.GO_SEMANTIC_SIMPLIFY,
        "semantic_simplify_fdr_cutoff": C.GO_FDR_THRESHOLD,
        "fdr_groups": dict(fdr_groups) if fdr_groups is not None else None,
        "queries": {
            name: {
                "genes_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
                "universe_sha256": hashlib.sha256("\n".join(universe).encode()).hexdigest(),
                "n_genes": len(genes),
                "n_universe": len(universe),
                "orgdb": orgdb,
            }
            for name, (genes, universe, orgdb) in normalized.items()
        },
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_path = result_dir / f"{cache_name}.csv.gz"
    manifest_path = result_dir / f"{cache_name}_manifest.json"
    signature = cache_signature(cache_config)
    outputs_owned = [cache_path, manifest_path]
    if cache_entry_is_current(result_dir, f"go:{cache_name}", signature, outputs=outputs_owned):
        return pd.read_csv(cache_path)

    with tempfile.TemporaryDirectory(prefix=f"embryo_{cache_name}_") as temporary:
        root = Path(temporary)
        jobs = []
        outputs = {}
        for index, (name, (genes, universe, orgdb)) in enumerate(normalized.items()):
            if not genes:
                continue
            universe_path = root / f"universe_{index}.csv"
            output = root / f"query_{index}.csv"
            pd.DataFrame({"gene": universe}).to_csv(universe_path, index=False)
            outputs[name] = output
            jobs.extend(
                {
                    "job_id": f"query_{index}",
                    "panel": panel,
                    "test": name,
                    "variant": variant,
                    "rank": rank,
                    "gene": gene,
                    "out_path": output,
                    "orgdb": orgdb,
                    "organism": "",
                    "universe_path": universe_path,
                    "fdr_group": fdr_groups[name] if fdr_groups is not None else "",
                    "semantic_simplify": C.GO_SEMANTIC_SIMPLIFY,
                    "semantic_simplify_fdr_cutoff": C.GO_FDR_THRESHOLD,
                }
                for rank, gene in enumerate(genes, start=1)
            )
        jobs_path = root / "jobs.csv"
        pd.DataFrame(jobs).to_csv(jobs_path, index=False)
        provenance = (
            run_clusterprofiler(
                jobs_path,
                cwd=C.PROJECT_ROOT,
                timeout_seconds=C.GO_TIMEOUT_SECONDS,
            )
            if jobs
            else {"status": "no query genes"}
        )
        frames = []
        for name, output in outputs.items():
            frame = pd.read_csv(output)
            frame.insert(0, "query_group", name)
            frames.append(frame)
    enrichment = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=GO_COLUMNS)
    )
    if fdr_groups is not None and enrichment.empty:
        enrichment = enrichment.drop(
            columns=["p_value", "p_adjust", "q_value", "significant"], errors="ignore"
        )
        enrichment["global_qvalue"] = pd.Series(dtype=float)
    temporary_csv = cache_path.with_suffix(".tmp")
    enrichment.to_csv(temporary_csv, index=False, compression="gzip")
    temporary_csv.replace(cache_path)
    query_results = {
        name: {
            "n_query_genes": len(genes),
            "n_universe_genes": len(universe),
            "n_terms": int(enrichment["query_group"].eq(name).sum()),
        }
        for name, (genes, universe, _) in normalized.items()
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(
            {
                **cache_config,
                "clusterprofiler_provenance": provenance,
                "query_results": query_results,
            },
            indent=2,
        )
        + "\n"
    )
    temporary_manifest.replace(manifest_path)
    update_cache_entry(
        result_dir,
        f"go:{cache_name}",
        signature,
        outputs_owned,
        signature_data=cache_config,
    )
    return enrichment


def select_go_terms(
    enrichment: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    fdr_column: str,
) -> pd.DataFrame:
    """Select specific significant terms retained by semantic simplification."""
    terms = enrichment.copy()
    terms["term_universe_fraction"] = terms["term_size"] / terms["effective_domain_size"]
    terms = terms[
        terms[fdr_column].le(C.GO_FDR_THRESHOLD)
        & terms["term_universe_fraction"].le(C.GO_MAX_TERM_UNIVERSE_FRACTION)
        & terms["semantic_simplified"].eq(True)
    ]
    columns = list(group_columns)
    return (
        terms.sort_values([*columns, fdr_column])
        .groupby(columns, sort=False)
        .head(C.GO_DISPLAY_TERMS_PER_GROUP)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Per-(gene, section) embedding annotations
# ---------------------------------------------------------------------------
def _mean_expression(points: pd.DataFrame) -> pd.Series:
    """Load mean counts per raw bin for every embedded gene-section pattern."""
    expression = pd.Series(index=points.index, dtype=np.float64)
    for species, cfg in C.SPECIES_CONFIGS.items():
        species_points = points[points["species"].eq(species)]
        for section_id, section in species_points.groupby("section_id", sort=False):
            path = Path(cfg.RESULT_DIR) / "gene_qc_cache" / f"{section_id}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}; rerun preprocessing first.")
            with np.load(path, allow_pickle=False) as cached:
                required = {"genes", "total_counts", "section_qc_json"}
                missing = required - set(cached.files)
                if missing:
                    raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
                genes = cached["genes"].astype(str)
                total_counts = cached["total_counts"].astype(np.float64, copy=False)
                section_qc = json.loads(str(cached["section_qc_json"].item()))
            if len(genes) != len(total_counts) or len(set(genes)) != len(genes):
                raise ValueError(f"{path} has invalid gene summaries.")
            n_bins = int(section_qc["n_bins_raw"])
            if n_bins < 1 or not np.isfinite(total_counts).all():
                raise ValueError(f"{path} has invalid expression summaries.")
            values = section["gene"].map(pd.Series(total_counts / n_bins, index=genes))
            if values.isna().any():
                missing_genes = section.loc[values.isna(), "gene"].astype(str).tolist()
                raise ValueError(f"{path} lacks embedded genes: {missing_genes[:10]}")
            expression.loc[section.index] = values.to_numpy()
    if expression.isna().any():
        raise RuntimeError("Some embedding points lack mean-expression values.")
    return expression


def embedding_points(embedding: dict[str, Any]) -> pd.DataFrame:
    """Build the per-(gene, section) table with orthology and SONIC metrics.

    The table is intentionally kept in memory rather than cached because its
    PCA coordinates are deterministically reconstructed from the spectra.
    Missing stage metadata or per-section SONIC results are errors.
    """
    frames: list[pd.DataFrame] = []
    for species, bundle in embedding["species"].items():
        gene_cols = np.asarray(bundle["gene_cols"], dtype=int)
        genes = np.asarray(bundle["gene_names"], dtype=str)
        coordinates = bundle["coords"][:, gene_cols, :]
        n_sections, n_genes, _ = coordinates.shape
        meta = bundle["meta"].reset_index(drop=True)
        stage_rank = {
            stage: index for index, stage in enumerate(C.SPECIES_CONFIGS[species].KEPT_TIMEPOINTS)
        }
        stage_index = meta["timepoint"].map(stage_rank)
        if stage_index.isna().any():
            raise RuntimeError(f"{species} embedding contains an unconfigured stage.")
        frame = pd.DataFrame(
            {
                "species": species,
                "section_id": np.repeat(meta["section_id"].astype(str), n_genes),
                "timepoint": np.repeat(meta["timepoint"].astype(str), n_genes),
                "stage_index": np.repeat(stage_index.astype(int), n_genes),
                "day": np.repeat(meta["day"].astype(float), n_genes),
                "gene": np.tile(genes, n_sections),
            }
        )
        for index, column in enumerate(PC_COLUMNS):
            frame[column] = coordinates[:, :, index].reshape(-1)
        frames.append(frame)
    points = pd.concat(frames, ignore_index=True)

    table = orthology.load_orthologs()
    mouse_to_human = dict(zip(table["mouse_symbol"], table["human_symbol"], strict=True))
    human_to_mouse = dict(zip(table["human_symbol"], table["mouse_symbol"], strict=True))
    mouse = points["species"].eq("mouse")
    points["mouse_symbol"] = np.where(mouse, points["gene"], points["gene"].map(human_to_mouse))
    points["human_symbol"] = np.where(mouse, points["gene"].map(mouse_to_human), points["gene"])

    panels = {
        species: group["gene"].drop_duplicates().astype(str).tolist()
        for species, group in points.groupby("species", sort=False)
    }
    section_ids = {
        species: group["section_id"].drop_duplicates().astype(str).tolist()
        for species, group in points.groupby("species", sort=False)
    }
    statistics = sv.load(panels, section_ids=section_ids).rename(columns={"feature": "gene"})
    points = points.merge(
        statistics[["species", "section_id", "gene", "Z_score"]],
        on=["species", "section_id", "gene"],
        how="left",
        validate="one_to_one",
    )
    if points["Z_score"].isna().any():
        raise RuntimeError("Some embedding points lack per-section SONIC results.")
    points["_mean_expression"] = _mean_expression(points)
    percentiles = points.groupby(["species", "section_id"], sort=False)[
        ["Z_score", "_mean_expression"]
    ].rank(pct=True)
    percentiles.columns = ["sv_percentile", "expression_percentile"]
    points[percentiles.columns] = percentiles
    points.drop(columns=["Z_score", "_mean_expression"], inplace=True)
    return points


# ---------------------------------------------------------------------------
# (Gene, stage) program clustering
# ---------------------------------------------------------------------------
def _cluster_states(points: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    coordinates = points[PC_COLUMNS].to_numpy(np.float32)
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=0,
        batch_size=4096,
        n_init=10,
    ).fit(coordinates)
    raw_labels = model.labels_
    order = np.lexsort((model.cluster_centers_[:, 1], model.cluster_centers_[:, 0]))
    remap = np.empty(n_clusters, dtype=int)
    remap[order] = np.arange(1, n_clusters + 1)
    points["cluster"] = remap[raw_labels]
    return pd.DataFrame(model.cluster_centers_[order], columns=PC_COLUMNS).assign(
        cluster=np.arange(1, n_clusters + 1)
    )


def _gene_stage_memberships(
    points: pd.DataFrame,
    centres: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each species-gene-stage median coordinate to its nearest cluster."""
    memberships = points.groupby(
        ["species", "gene", "timepoint", "stage_index"],
        as_index=False,
        sort=False,
    ).agg(
        day=("day", "median"),
        n_sections=("section_id", "nunique"),
        **{column: (column, "median") for column in PC_COLUMNS},
    )
    coordinates = memberships[PC_COLUMNS].to_numpy(np.float32)
    centre_coordinates = centres[PC_COLUMNS].to_numpy(np.float32)
    distances = np.square(coordinates[:, None, :] - centre_coordinates[None, :, :]).sum(axis=2)
    memberships["cluster"] = centres["cluster"].to_numpy(int)[distances.argmin(axis=1)]
    return memberships


def _cluster_functional_labels(enrichment: pd.DataFrame, clusters: Sequence[int]) -> pd.DataFrame:
    """Validate concise labels against significant terms in both species."""
    # Semantic simplification removes redundant display terms, not biological support.
    eligible = enrichment[
        enrichment["global_qvalue"].le(C.GO_FDR_THRESHOLD)
        & (enrichment["term_size"] / enrichment["effective_domain_size"]).le(
            C.GO_MAX_TERM_UNIVERSE_FRACTION
        )
    ]
    rows = []
    for cluster in clusters:
        label, go_ids = CLUSTER_FUNCTIONAL_LABELS.get(cluster, (f"Cluster {cluster}", ()))
        support = eligible[eligible["cluster"].eq(cluster) & eligible["native"].isin(go_ids)]
        observed = set(support[["species", "native"]].itertuples(index=False, name=None))
        expected = {(species, go_id) for species in C.GO_ORGDB for go_id in go_ids}
        supported = bool(go_ids) and observed == expected
        if not supported:
            LOGGER.warning(
                "Cluster %s: curated label lacks current GO support; using a generic label.",
                cluster,
            )
            label, go_ids = f"Cluster {cluster}", ()
        names = (
            support[["native", "display_name"]]
            .drop_duplicates("native")
            .set_index("native")["display_name"]
        )
        rows.append(
            {
                "cluster": cluster,
                "functional_label": label,
                "curated_label_supported": supported,
                "consensus_go_ids": ";".join(go_ids),
                "consensus_go_terms": "; ".join(names.loc[list(go_ids)]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Local mixing score for embedding quality
# ---------------------------------------------------------------------------
def local_mixing_scores(  # noqa: C901
    coordinates: np.ndarray,
    labels: Mapping[str, Sequence[str]],
    *,
    n_neighbors: int = C.LOCAL_MIXING_NEIGHBORS,
    batch_size: int = 25_000,
) -> pd.DataFrame:
    """Return finite-neighbor-corrected TV mixing scores in ``[0, 1]``.

    For point ``i`` and category ``c``, let ``q_ic`` be the category proportion
    among its ``k`` nearest neighbors and ``p_ic`` the global proportion after
    excluding point ``i``. The observed total-variation distance is

    ``D_i = 0.5 * sum_c |q_ic - p_ic|``.

    Its finite-neighbor null expectation is

    ``E_i = 0.5 * sum_c E[|X_ic / k - p_ic|]``,

    where ``X_ic ~ Hypergeometric(N - 1, K_ic, k)`` and ``K_ic`` is the number
    of category-``c`` points after excluding ``i``. With
    ``Dmax_i = 1 - min_c(p_ic)``, the returned score is

    ``M_i = 1 - clip((D_i - E_i) / (Dmax_i - E_i), 0, 1)``.

    Thus one denotes null-level or stronger mixing and zero denotes maximal
    label-composition divergence. Neighbors are selected deterministically;
    the hypergeometric distribution defines the correction rather than a draw.
    """
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("coordinates must have shape (points, dimensions).")
    if not np.isfinite(values).all():
        raise ValueError("coordinates contain non-finite values.")
    if not 1 <= n_neighbors < len(values):
        raise ValueError("n_neighbors must be between 1 and n_points - 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    # Excluding the query changes its own category count, so precompute one
    # population composition and finite-k null per possible query category.
    encoded: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    for factor, factor_labels in labels.items():
        series = pd.Series(factor_labels, copy=False)
        if len(series) != len(values) or series.isna().any():
            raise ValueError(f"{factor!r} must contain one nonmissing label per point.")
        codes, uniques = pd.factorize(series.astype(str), sort=True)
        if len(uniques) < 2:
            raise ValueError(f"{factor!r} must contain at least two labels.")
        counts = np.bincount(codes)
        population_size = len(values) - 1
        proportions = np.empty((len(counts), len(counts)), dtype=np.float64)
        null_distance = np.empty(len(counts), dtype=np.float64)
        maximum_distance = np.empty(len(counts), dtype=np.float64)
        for query_code in range(len(counts)):
            population_counts = counts.copy()
            population_counts[query_code] -= 1
            proportions[query_code] = population_counts / population_size
            expected_distance = 0.0
            for category_count in population_counts:
                # Each marginal neighbor count is hypergeometric; linearity of
                # expectation yields exact expected TV without Monte Carlo.
                lower = max(0, n_neighbors - (population_size - category_count))
                upper = min(n_neighbors, category_count)
                outcomes = np.arange(lower, upper + 1)
                expected_distance += 0.5 * np.sum(
                    np.abs(outcomes / n_neighbors - category_count / population_size)
                    * hypergeom.pmf(
                        outcomes,
                        population_size,
                        category_count,
                        n_neighbors,
                    )
                )
            null_distance[query_code] = expected_distance
            maximum_distance[query_code] = 1 - proportions[query_code].min()
        if np.any(maximum_distance <= null_distance):
            raise ValueError(f"{factor!r} has no TV range beyond finite-neighbor null variation.")
        encoded[factor] = (
            codes,
            proportions,
            null_distance,
            maximum_distance,
        )

    # Query one shared deterministic neighbor graph for every annotation factor.
    tree = cKDTree(values)
    scores = {factor: np.empty(len(values), dtype=np.float32) for factor in labels}
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        query_indices = np.arange(start, stop)
        raw_neighbors = tree.query(values[start:stop], k=n_neighbors + 1, workers=1)[1]
        keep = raw_neighbors != query_indices[:, None]
        keep &= np.cumsum(keep, axis=1) <= n_neighbors
        neighbors = raw_neighbors[keep].reshape(stop - start, n_neighbors)
        for factor, (
            codes,
            proportions,
            null_distance,
            maximum_distance,
        ) in encoded.items():
            query_codes = codes[query_indices]
            neighbor_codes = codes[neighbors]
            local_proportions = np.column_stack(
                [
                    np.mean(neighbor_codes == category, axis=1)
                    for category in range(proportions.shape[1])
                ]
            )
            observed_distance = 0.5 * np.abs(local_proportions - proportions[query_codes]).sum(
                axis=1
            )
            # Remove expected finite-k variation and bound excess divergence
            # relative to the most compositionally different neighborhood.
            excess_distance = (observed_distance - null_distance[query_codes]) / (
                maximum_distance[query_codes] - null_distance[query_codes]
            )
            scores[factor][start:stop] = 1 - np.clip(excess_distance, 0, 1)
    return pd.DataFrame(scores)


def _local_mixing_by_section(points: pd.DataFrame) -> pd.DataFrame:
    """Compute point-level mixing and aggregate it to independent sections."""
    labels = [f"Q{index}" for index in range(1, C.LOCAL_MIXING_QUANTILES + 1)]
    expression_labels = pd.cut(
        points["expression_percentile"],
        bins=np.linspace(0, 1, C.LOCAL_MIXING_QUANTILES + 1),
        labels=labels,
        include_lowest=True,
    )
    sv_labels = pd.cut(
        points["sv_percentile"],
        bins=np.linspace(0, 1, C.LOCAL_MIXING_QUANTILES + 1),
        labels=labels,
        include_lowest=True,
    )
    scores = local_mixing_scores(
        points[PC_COLUMNS].to_numpy(np.float32),
        {
            "species": points["species"],
            "time": points["species"] + ":" + points["timepoint"],
            "expression": expression_labels,
            "sv": sv_labels,
        },
    ).rename(columns=lambda factor: f"{factor}_mixing")
    scores[["query_species", "section_id", "timepoint"]] = points[
        ["species", "section_id", "timepoint"]
    ].rename(columns={"species": "query_species"})
    return (
        scores.groupby(["query_species", "section_id", "timepoint"], as_index=False)
        .agg(
            n_patterns=("expression_mixing", "size"),
            species_mixing=("species_mixing", "mean"),
            time_mixing=("time_mixing", "mean"),
            expression_mixing=("expression_mixing", "mean"),
            sv_mixing=("sv_mixing", "mean"),
        )
        .melt(
            id_vars=["query_species", "section_id", "timepoint", "n_patterns"],
            value_vars=[
                "species_mixing",
                "time_mixing",
                "expression_mixing",
                "sv_mixing",
            ],
            var_name="factor",
            value_name="mixing_score",
        )
        .assign(factor=lambda frame: frame["factor"].str.removesuffix("_mixing"))
        .rename(columns={"query_species": "species"})
        .sort_values(["factor", "species", "section_id"])
        .reset_index(drop=True)
    )


def summarize_local_mixing(
    by_section: pd.DataFrame,
    *,
    bootstrap_samples: int = 2_000,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Summarize section-level mixing with a section-bootstrap confidence interval."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")
    rng = np.random.default_rng(random_seed)
    summaries = []
    for factor, group in by_section.groupby("factor", sort=False):
        values = group["mixing_score"].to_numpy(float)
        bootstrap = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True).mean(
            axis=1
        )
        summaries.append(
            {
                "factor": factor,
                "n_sections": len(values),
                "mean": values.mean(),
                "median": np.median(values),
                "q25": np.quantile(values, 0.25),
                "q75": np.quantile(values, 0.75),
                "ci95_low": np.quantile(bootstrap, 0.025),
                "ci95_high": np.quantile(bootstrap, 0.975),
            }
        )
    return pd.DataFrame(summaries)


# ---------------------------------------------------------------------------
# Cache orchestration
# ---------------------------------------------------------------------------
def _input_signature_data(embedding: dict[str, Any]) -> dict[str, Any]:
    """Describe every data input and parameter consumed by embedding metrics."""
    paths = [orthology.ORTHO_TSV, CLUSTERPROFILER_SCRIPT]
    for cfg in C.SPECIES_CONFIGS.values():
        metadata = utils.filtered_metadata(cfg)
        paths.extend(
            [
                Path(cfg.RESULT_DIR) / "gene_qc.csv",
                cfg.GEOMETRY_CSV,
                cfg.SV_MANIFEST_JSON,
            ]
        )
        for section_id in metadata["section_id"].astype(str):
            paths.extend(
                [
                    cfg.SPECTRA_RESULT_DIR / f"{section_id}.npz",
                    cfg.SV_RESULT_DIR / f"{section_id}.csv.gz",
                    Path(cfg.RESULT_DIR) / "gene_qc_cache" / f"{section_id}.npz",
                ]
            )
    pca = embedding["pca"]
    arrays = [
        embedding["freq_edges"],
        pca.components_,
        pca.mean_,
        pca.explained_variance_ratio_,
    ]
    return utils.cache_signature_data(
        paths,
        arrays,
        cache_revision=C.CACHE_REVISIONS["metrics"],
        parameters=C.METRICS_CACHE_PARAMETERS,
    )


def require_current_cache(embedding: dict[str, Any]) -> dict[str, Any]:
    """Reject stale or incomplete plotting inputs without recomputing analysis."""
    manifest = utils.current_cache_manifest(
        RESULT_DIR, _CACHE_ARTIFACTS, _input_signature_data(embedding)
    )
    if manifest is None:
        raise RuntimeError(
            f"Stale or incomplete cache: {RESULT_DIR}. Run python -m scripts.embryo_xspecies.workflow --force."
        )
    return manifest


def run(embedding: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Compute or load all joint-embedding analysis metrics."""
    signature_data = _input_signature_data(embedding)
    signature = cache_signature(signature_data)
    if (
        not force
        and (
            manifest := utils.current_cache_manifest(
                RESULT_DIR,
                _CACHE_ARTIFACTS,
                signature_data,
            )
        )
        is not None
    ):
        LOGGER.info("Embedding metrics cache is current: %s", RESULT_DIR)
        return manifest
    if force and RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    points = embedding_points(embedding)

    # Cluster all section-level patterns, then collapse slide replicates to one
    # membership per species-gene-stage for enrichment and temporal tracking.
    centres = _cluster_states(points, C.N_PATTERN_CLUSTERS)
    memberships = _gene_stage_memberships(points, centres)
    gene_sets = memberships[["species", "gene", "cluster"]].drop_duplicates()
    queries = {}
    for species, species_memberships in memberships.groupby("species", sort=True):
        universe = species_memberships["gene"].unique()
        for cluster, group in gene_sets[gene_sets["species"].eq(species)].groupby(
            "cluster", sort=True
        ):
            queries[f"{species}:{int(cluster)}"] = (
                group["gene"],
                universe,
                C.GO_ORGDB[species],
            )
    enrichment = cached_go_enrichment(
        queries,
        cache_name="cluster_go_queries",
        panel="embryo_embedding",
        variant="stage_aware_membership",
        result_dir=RESULT_DIR,
        fdr_groups={name: name.split(":", 1)[0] for name in queries},
    )
    groups = enrichment["query_group"].str.split(":", expand=True).reindex(columns=[0, 1])
    enrichment["species"] = groups[0]
    enrichment["cluster"] = groups[1].astype(int)
    # Compute fold enrichment relative to the species-specific background genes (effective_domain_size).
    enrichment["fold_enrichment"] = (enrichment["intersection_size"] / enrichment["query_size"]) / (
        enrichment["term_size"] / enrichment["effective_domain_size"]
    )
    enrichment.to_csv(RESULT_DIR / "cluster_go_enrichment.csv.gz", index=False)
    functional_labels = _cluster_functional_labels(enrichment, centres["cluster"].tolist())
    centres = centres.merge(functional_labels, on="cluster", validate="one_to_one")
    centres.to_csv(RESULT_DIR / "cluster_summary.csv", index=False)
    memberships.to_csv(RESULT_DIR / "gene_stage_cluster_membership.csv.gz", index=False)

    # Compute and save local mixing metrics
    _local_mixing_by_section(points).to_csv(RESULT_DIR / "local_mixing_by_section.csv", index=False)

    clusters_per_gene = (
        memberships[["species", "gene", "cluster"]]
        .drop_duplicates()
        .groupby(["species", "gene"])
        .size()
    )

    embedding_manifest = {
        "n_gene_section_patterns": len(points),
        "n_sections": int(points[["species", "section_id"]].drop_duplicates().shape[0]),
        "n_clusters": C.N_PATTERN_CLUSTERS,
        "n_gene_stage_memberships": len(memberships),
        "genes_in_multiple_clusters": {
            species: int(group.gt(1).sum())
            for species, group in clusters_per_gene.groupby(level="species")
        },
        "pca_explained_variance_ratio": embedding["pca"].explained_variance_ratio_.tolist(),
        "cluster_go_enrichment": {
            "assignment_unit": "species, gene, stage; slide replicates use median PCA coordinates",
            "query": "unique genes observed in each species-cluster across stages",
            "background": "all QC-passing embedded genes for that species",
            "source": {
                species: f"{C.GO_SOURCE} via local {orgdb}" for species, orgdb in C.GO_ORGDB.items()
            },
            "global_fdr_scope": "all cluster terms within each species",
            "semantic_simplification_score": "global_qvalue",
            "within_cluster_fdr": "no separate BH correction or cutoff",
            "functional_labels": "manually shortened from GO IDs significant in both species",
        },
    }
    mixing_manifest = {
        "metric": "hypergeometric-corrected local total-variation mixing",
        "range": "0 = maximal label-composition divergence; 1 = null-level or stronger mixing",
        "n_neighbors": C.LOCAL_MIXING_NEIGHBORS,
        "unit": "section mean of per-pattern scores",
        "factors": [
            "species",
            "species-specific time",
            "expression quintile",
            "SV quintile",
        ],
    }
    manifest = {
        "cache_signature": signature,
        "signature_data": signature_data,
        "artifacts": list(_CACHE_ARTIFACTS),
        "embedding": embedding_manifest,
        "local_mixing": mixing_manifest,
        "go": {
            "source": C.GO_SOURCE,
            "orgdb": C.GO_ORGDB,
            "semantic_simplify": C.GO_SEMANTIC_SIMPLIFY,
            "display_filters": {
                "fdr_column": "global_qvalue",
                "fdr_threshold": C.GO_FDR_THRESHOLD,
                "maximum_term_universe_fraction": C.GO_MAX_TERM_UNIVERSE_FRACTION,
                "clusterprofiler_semantic_simplified_only": True,
                "terms_per_group": C.GO_DISPLAY_TERMS_PER_GROUP,
            },
        },
    }
    utils.write_cache_manifest(RESULT_DIR, _CACHE_ARTIFACTS, manifest)
    LOGGER.info("Wrote embedding metrics to %s", RESULT_DIR)
    return manifest
