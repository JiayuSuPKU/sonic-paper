"""Compute and cache metrics for the joint embryo spectral embedding."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import false_discovery_control, hypergeom, spearmanr
from sklearn.cluster import MiniBatchKMeans

from . import config as C
from . import embeddings, orthology, sv

LOGGER = logging.getLogger("embryo_xspecies.embedding_metrics")

# Unified cache location and PCA coordinate contract.
RESULT_DIR = C.COMPARISON_RESULT_DIR / "embedding_metrics"
PC_COLUMNS = [f"PC{index}" for index in range(1, C.N_PC + 1)]

# Technical families are retained in metrics but not chosen as figure examples.
EXAMPLE_GENE_EXCLUDED_PREFIXES = (
    "RPL",
    "RPS",
    "MRPL",
    "MRPS",
    "HBA",
    "HBB",
    "HIST",
)

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
def _cached_go_enrichment(  # noqa: C901
    queries: Mapping[str, tuple[Sequence[str], Sequence[str], str]],
    *,
    cache_name: str,
    panel: str,
    variant: str,
) -> pd.DataFrame:
    """Run and cache named clusterProfiler queries with explicit universes."""
    rscript = Path(sys.prefix) / "bin" / "Rscript"
    wrapper = C.PROJECT_ROOT / "scripts" / "run_clusterprofiler.R"
    if not rscript.exists():
        raise FileNotFoundError(f"Missing Rscript in active environment: {rscript}")

    normalized = {}
    for name, (genes, universe, orgdb) in queries.items():
        query_genes = sorted(set(map(str, genes)))
        universe_genes = sorted(set(map(str, universe)))
        if not query_genes:
            raise ValueError(f"GO query {name!r} has no genes.")
        if not set(query_genes).issubset(universe_genes):
            raise ValueError(f"GO query {name!r} contains genes outside its universe.")
        normalized[name] = (query_genes, universe_genes, orgdb)

    cache_config = {
        "panel": panel,
        "variant": variant,
        "wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        "semantic_simplify": C.GO_SEMANTIC_SIMPLIFY,
        "semantic_simplify_p_adjust_cutoff": C.GO_FDR_THRESHOLD,
        "queries": {
            name: {
                "genes_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
                "universe_sha256": hashlib.sha256(
                    "\n".join(universe).encode()
                ).hexdigest(),
                "n_genes": len(genes),
                "n_universe": len(universe),
                "orgdb": orgdb,
            }
            for name, (genes, universe, orgdb) in normalized.items()
        },
    }
    cache_path = RESULT_DIR / f"{cache_name}.csv.gz"
    manifest_path = RESULT_DIR / f"{cache_name}_manifest.json"
    if cache_path.exists() != manifest_path.exists():
        raise RuntimeError(
            f"Incomplete GO cache: expected both {cache_path} and {manifest_path}."
        )
    if cache_path.exists() and json.loads(manifest_path.read_text()) == cache_config:
        return pd.read_csv(cache_path)

    with tempfile.TemporaryDirectory(prefix=f"embryo_{cache_name}_") as temporary:
        root = Path(temporary)
        jobs = []
        outputs = {}
        for index, (name, (genes, universe, orgdb)) in enumerate(normalized.items()):
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
                    "semantic_simplify": C.GO_SEMANTIC_SIMPLIFY,
                    "semantic_simplify_p_adjust_cutoff": C.GO_FDR_THRESHOLD,
                }
                for rank, gene in enumerate(genes, start=1)
            )
        jobs_path = root / "jobs.csv"
        pd.DataFrame(jobs).to_csv(jobs_path, index=False)
        subprocess.run(
            [str(rscript), str(wrapper), str(jobs_path)],
            cwd=C.PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=C.GO_TIMEOUT_SECONDS,
        )
        frames = []
        for name, output in outputs.items():
            frame = pd.read_csv(output)
            if frame.empty:
                continue
            frame.insert(0, "query_group", name)
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"clusterProfiler returned no terms for {panel}.")
    enrichment = pd.concat(frames, ignore_index=True)
    enrichment.to_csv(cache_path, index=False)
    manifest_path.write_text(json.dumps(cache_config, indent=2) + "\n")
    return enrichment


def select_go_terms(
    enrichment: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    fdr_column: str,
) -> pd.DataFrame:
    """Select specific significant terms retained by semantic simplification."""
    terms = enrichment.copy()
    terms["term_universe_fraction"] = (
        terms["term_size"] / terms["effective_domain_size"]
    )
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
# Temporal variation
# ---------------------------------------------------------------------------
def quantify_temporal_variation(
    embedding: np.ndarray,
    stage_labels: Sequence[str],
    *,
    feature_names: Sequence[str],
    stage_order: Sequence[str],
) -> pd.DataFrame:
    """Quantify within- and between-stage variation for replicated embeddings.

    ``embedding`` must have shape ``(replicates, features, dimensions)``.
    Replicates sharing a stage label form one group. The returned mean squares
    sum variation over all embedding dimensions. For stage ``s``, replicate
    ``r``, ``N`` total replicates, and ``S`` stages, each feature uses

    ``within = sum_s sum_r ||x_sr - mean_s||^2 / (N - S)``

    ``between = sum_s n_s ||mean_s - grand_mean||^2 / (S - 1)``.
    """
    coordinates = np.asarray(embedding, dtype=np.float64)
    labels = np.asarray(stage_labels, dtype=str)
    features = np.asarray(feature_names, dtype=str)
    stages = list(map(str, stage_order))
    if coordinates.ndim != 3:
        raise ValueError(
            "embedding must have shape (replicates, features, dimensions)."
        )
    if labels.shape != (coordinates.shape[0],):
        raise ValueError("stage_labels must contain one label per replicate.")
    if features.shape != (coordinates.shape[1],):
        raise ValueError("feature_names must contain one name per feature.")
    if not stages or len(stages) != len(set(stages)):
        raise ValueError("stage_order must contain unique stages.")
    if set(labels) != set(stages):
        raise ValueError("stage_labels and stage_order contain different stages.")
    if not np.isfinite(coordinates).all():
        raise ValueError("embedding contains non-finite coordinates.")

    # Create per-stage indices and separate within- and between-stage variation.
    indices = [np.flatnonzero(labels == stage) for stage in stages]
    if any(len(index) < 2 for index in indices):
        raise ValueError("Each stage requires at least two replicates.")
    grand_mean = coordinates.mean(axis=0)
    total_ss = np.square(coordinates - grand_mean).sum(axis=(0, 2))
    within_ss = np.zeros(len(features), dtype=np.float64)
    for index in indices:
        values = coordinates[index]
        mean = values.mean(axis=0)
        within_ss += np.square(values - mean).sum(axis=(0, 2))

    n_replicates = len(coordinates)
    n_stages = len(stages)
    return pd.DataFrame(
        {
            "feature": features,
            "within_stage_variance": within_ss / (n_replicates - n_stages),
            "between_stage_variance": np.maximum(total_ss - within_ss, 0)
            / (n_stages - 1),
        }
    )


def _temporal_conservation(
    mouse: pd.DataFrame,
    human: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    mouse = mouse[mouse["passes_within_stage_filter"]]
    human = human[human["passes_within_stage_filter"]]
    orthologs = orthology.load_orthologs()
    table = (
        orthologs.merge(mouse, left_on="mouse_symbol", right_on="gene")
        .drop(columns="gene")
        .merge(
            human,
            left_on="human_symbol",
            right_on="gene",
            suffixes=("_mouse", "_human"),
        )
        .drop(columns="gene")
    )
    table["mouse_dynamic_percentile"] = table["between_stage_variance_mouse"].rank(
        pct=True
    )
    table["human_dynamic_percentile"] = table["between_stage_variance_human"].rank(
        pct=True
    )

    mouse_top = table["mouse_dynamic_percentile"].ge(
        C.TEMPORAL_DYNAMIC_PERCENTILE_THRESHOLD
    )
    human_top = table["human_dynamic_percentile"].ge(
        C.TEMPORAL_DYNAMIC_PERCENTILE_THRESHOLD
    )
    table["dynamic_group"] = "not top-decile"
    table.loc[mouse_top & ~human_top, "dynamic_group"] = "mouse-specific dynamic"
    table.loc[~mouse_top & human_top, "dynamic_group"] = "human-specific dynamic"
    table.loc[mouse_top & human_top, "dynamic_group"] = "conserved dynamic"
    overlap = int((mouse_top & human_top).sum())
    expected = int(mouse_top.sum()) * int(human_top.sum()) / len(table)
    rho, rho_p = spearmanr(
        table["between_stage_variance_mouse"],
        table["between_stage_variance_human"],
    )
    summary = {
        "n_orthologs": len(table),
        "between_stage_variance_spearman": float(rho),
        "between_stage_variance_spearman_p": float(rho_p),
        "top_decile_overlap": overlap,
        "mouse_specific_top_decile": int((mouse_top & ~human_top).sum()),
        "human_specific_top_decile": int((~mouse_top & human_top).sum()),
        "top_decile_expected": expected,
        "top_decile_fold_enrichment": overlap / expected,
        "top_decile_hypergeom_p": float(
            hypergeom.sf(
                overlap - 1,
                len(table),
                int(mouse_top.sum()),
                int(human_top.sum()),
            )
        ),
    }
    return table, summary


def _select_temporal_examples(
    conservation: pd.DataFrame,
    displayed_terms: pd.DataFrame,
) -> pd.DataFrame:
    conservation = conservation[
        conservation["mouse_protein_coding"] & conservation["human_protein_coding"]
    ]
    excluded_from_display = conservation["mouse_symbol"].str.upper().str.startswith(
        EXAMPLE_GENE_EXCLUDED_PREFIXES
    ) | conservation["human_symbol"].str.upper().str.startswith(
        EXAMPLE_GENE_EXCLUDED_PREFIXES
    )
    conservation = conservation[~excluded_from_display]
    selected = []
    ranking = {
        "conserved dynamic": (
            ["mouse_dynamic_percentile", "human_dynamic_percentile"],
            [False, False],
        ),
        "mouse-specific dynamic": (
            ["mouse_dynamic_percentile", "human_dynamic_percentile"],
            [False, True],
        ),
        "human-specific dynamic": (
            ["human_dynamic_percentile", "mouse_dynamic_percentile"],
            [False, True],
        ),
    }
    for category, (columns, ascending) in ranking.items():
        ranked = conservation[conservation["dynamic_group"].eq(category)].sort_values(
            columns, ascending=ascending
        )
        group_terms = displayed_terms[displayed_terms["query_group"].eq(category)]
        symbol_column = (
            "human_symbol" if category == "human-specific dynamic" else "mouse_symbol"
        )
        if group_terms.empty:
            selection_basis = "temporal variation percentile; no specific GO term"
            pathway_terms = ""
        else:
            hit_genes = set(
                "/".join(group_terms["hit_genes"].dropna().astype(str)).split("/")
            )
            ranked = ranked[ranked[symbol_column].isin(hit_genes)]
            selection_basis = "highest focal temporal variation percentile among displayed GO term genes"
        row = ranked.iloc[0]
        if not group_terms.empty:
            selected_symbol = str(row[symbol_column])
            pathway_terms = "; ".join(
                group_terms.loc[
                    group_terms["hit_genes"]
                    .fillna("")
                    .str.split("/")
                    .map(lambda genes, symbol=selected_symbol: symbol in genes),
                    "display_name",
                ].head(2)
            )
        selected.append(
            {
                "category": category,
                "mouse_symbol": row["mouse_symbol"],
                "human_symbol": row["human_symbol"],
                "mouse_dynamic_percentile": row["mouse_dynamic_percentile"],
                "human_dynamic_percentile": row["human_dynamic_percentile"],
                "selection_basis": selection_basis,
                "pathway_terms": pathway_terms,
            }
        )
    return pd.DataFrame(selected)


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
    """Build the per-(gene, section) table with orthology and QuadSV metrics.

    The table is intentionally kept in memory rather than cached because its
    PCA coordinates are deterministically reconstructed from the spectra.
    Missing stage metadata or per-section QuadSV results are errors.
    """
    frames: list[pd.DataFrame] = []
    for species, bundle in embedding["species"].items():
        gene_cols = np.asarray(bundle["gene_cols"], dtype=int)
        genes = np.asarray(bundle["gene_names"], dtype=str)
        coordinates = bundle["coords"][:, gene_cols, :]
        n_sections, n_genes, _ = coordinates.shape
        meta = bundle["meta"].reset_index(drop=True)
        stage_rank = {
            stage: index
            for index, stage in enumerate(C.SPECIES_CONFIGS[species].KEPT_TIMEPOINTS)
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
    mouse_to_human = dict(
        zip(table["mouse_symbol"], table["human_symbol"], strict=True)
    )
    human_to_mouse = dict(
        zip(table["human_symbol"], table["mouse_symbol"], strict=True)
    )
    mouse = points["species"].eq("mouse")
    points["mouse_symbol"] = np.where(
        mouse, points["gene"], points["gene"].map(human_to_mouse)
    )
    points["human_symbol"] = np.where(
        mouse, points["gene"].map(mouse_to_human), points["gene"]
    )

    panels = {
        species: group["gene"].drop_duplicates().astype(str).tolist()
        for species, group in points.groupby("species", sort=False)
    }
    section_ids = {
        species: group["section_id"].drop_duplicates().astype(str).tolist()
        for species, group in points.groupby("species", sort=False)
    }
    statistics = sv.load(panels, section_ids=section_ids).rename(
        columns={"feature": "gene"}
    )
    points = points.merge(
        statistics[["species", "section_id", "gene", "Z_score"]],
        on=["species", "section_id", "gene"],
        how="left",
        validate="one_to_one",
    )
    if points["Z_score"].isna().any():
        raise RuntimeError("Some embedding points lack per-section QuadSV results.")
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
    distances = np.square(coordinates[:, None, :] - centre_coordinates[None, :, :]).sum(
        axis=2
    )
    memberships["cluster"] = centres["cluster"].to_numpy(int)[distances.argmin(axis=1)]
    return memberships


def _cluster_functional_labels(enrichment: pd.DataFrame) -> pd.DataFrame:
    """Validate concise labels against significant terms in both species."""
    eligible = enrichment[
        enrichment["global_qvalue"].le(C.GO_FDR_THRESHOLD)
        & (enrichment["term_size"] / enrichment["effective_domain_size"]).le(
            C.GO_MAX_TERM_UNIVERSE_FRACTION
        )
        & enrichment["semantic_simplified"].eq(True)
    ]
    rows = []
    for cluster, (label, go_ids) in CLUSTER_FUNCTIONAL_LABELS.items():
        support = eligible[
            eligible["cluster"].eq(cluster) & eligible["native"].isin(go_ids)
        ]
        observed = set(
            support[["species", "native"]].itertuples(index=False, name=None)
        )
        expected = {(species, go_id) for species in C.GO_ORGDB for go_id in go_ids}
        if observed != expected:
            raise RuntimeError(
                f"Cluster {cluster} label {label!r} lacks cross-species GO support: "
                f"{sorted(expected - observed)}"
            )
        names = (
            support[["native", "display_name"]]
            .drop_duplicates("native")
            .set_index("native")["display_name"]
        )
        rows.append(
            {
                "cluster": cluster,
                "functional_label": label,
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
            raise ValueError(
                f"{factor!r} has no TV range beyond finite-neighbor null variation."
            )
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
            observed_distance = 0.5 * np.abs(
                local_proportions - proportions[query_codes]
            ).sum(axis=1)
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
        bootstrap = rng.choice(
            values, size=(bootstrap_samples, len(values)), replace=True
        ).mean(axis=1)
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
def _cache_temporal_metrics(
    embedding: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    statistics: dict[str, pd.DataFrame] = {}
    embedding_dimensions: int | None = None

    # For each species, quantify temporal variation and filter by within-stage variance.
    for species in ("mouse", "human"):
        bundle = embedding["species"][species]
        if bundle["label"] != species:
            raise ValueError(
                f"Expected {species!r} embedding, got {bundle['label']!r}."
            )
        gene_columns = np.asarray(bundle["gene_cols"], dtype=int)
        coordinates = np.asarray(bundle["coords"][:, gene_columns, :])
        dimensions = coordinates.shape[2]
        if embedding_dimensions is not None and dimensions != embedding_dimensions:
            raise ValueError("Mouse and human embeddings have different dimensions.")
        embedding_dimensions = dimensions
        metrics = quantify_temporal_variation(
            coordinates,
            bundle["meta"]["timepoint"].astype(str),
            feature_names=bundle["gene_names"],
            stage_order=C.SPECIES_CONFIGS[species].KEPT_TIMEPOINTS,
        ).rename(columns={"feature": "gene"})

        # Remove genes with excessive within-stage variance.
        cutoff = metrics["within_stage_variance"].quantile(
            C.TEMPORAL_MAX_WITHIN_STAGE_PERCENTILE
        )
        metrics["passes_within_stage_filter"] = metrics["within_stage_variance"].le(
            cutoff
        )
        statistics[species] = metrics

    if embedding_dimensions is None:
        raise RuntimeError("No species embeddings were quantified.")
    mouse = statistics["mouse"]
    human = statistics["human"]

    # Check human/mouse between-stage variance ranking agreement.
    conservation, summary = _temporal_conservation(mouse, human)
    coding = conservation[
        conservation["mouse_protein_coding"] & conservation["human_protein_coding"]
    ]
    mouse_universe = coding["mouse_symbol"].tolist()
    human_universe = coding["human_symbol"].tolist()
    enrichment = _cached_go_enrichment(
        {
            "mouse-specific dynamic": (
                coding.loc[
                    coding["dynamic_group"].eq("mouse-specific dynamic"),
                    "mouse_symbol",
                ],
                mouse_universe,
                C.GO_ORGDB["mouse"],
            ),
            "human-specific dynamic": (
                coding.loc[
                    coding["dynamic_group"].eq("human-specific dynamic"),
                    "human_symbol",
                ],
                human_universe,
                C.GO_ORGDB["human"],
            ),
            "conserved dynamic": (
                coding.loc[
                    coding["dynamic_group"].eq("conserved dynamic"),
                    "mouse_symbol",
                ],
                mouse_universe,
                C.GO_ORGDB["mouse"],
            ),
        },
        cache_name="temporal_go_enrichment",
        panel="embryo_temporal",
        variant="disjoint_top_decile",
    )
    displayed_terms = select_go_terms(
        enrichment,
        group_columns=["query_group"],
        fdr_column="p_adjust",
    )
    selected = _select_temporal_examples(conservation, displayed_terms)

    metric_columns = [
        "gene",
        "within_stage_variance",
        "between_stage_variance",
        "passes_within_stage_filter",
    ]
    pd.concat(
        [
            mouse[metric_columns].assign(species="mouse"),
            human[metric_columns].assign(species="human"),
        ],
        ignore_index=True,
    )[["species", *metric_columns]].to_csv(
        RESULT_DIR / "temporal_gene_statistics.csv.gz", index=False
    )
    conservation[
        [
            "human_symbol",
            "mouse_symbol",
            "mouse_dynamic_percentile",
            "human_dynamic_percentile",
            "dynamic_group",
        ]
    ].to_csv(RESULT_DIR / "ortholog_temporal_conservation.csv.gz", index=False)
    selected.to_csv(RESULT_DIR / "selected_genes.csv", index=False)

    manifest = {
        "method": (
            "one-way stage mean squares summed over joint-embedding dimensions; "
            "genes above the within-stage variance 95th percentile are excluded, "
            "then ranked by between-stage variance"
        ),
        "embedding_dimensions": embedding_dimensions,
        "dynamic_score": "between_stage_variance",
        "maximum_within_stage_percentile": C.TEMPORAL_MAX_WITHIN_STAGE_PERCENTILE,
        "within_stage_variance_cutoffs": {
            species: float(
                metrics.loc[
                    metrics["passes_within_stage_filter"], "within_stage_variance"
                ].max()
            )
            for species, metrics in statistics.items()
        },
        "interpreted_genes": {
            species: int(metrics["passes_within_stage_filter"].sum())
            for species, metrics in statistics.items()
        },
        "same_stage_slides_are_replicates": True,
        "stage_replicates": {
            species: {
                stage: int(bundle["meta"]["timepoint"].astype(str).eq(stage).sum())
                for stage in C.SPECIES_CONFIGS[species].KEPT_TIMEPOINTS
            }
            for species, bundle in embedding["species"].items()
        },
        "within_stage_variation_interpretation": (
            "combined section-plane, embryo, and measurement nuisance baseline"
        ),
        "genes": {species: len(metrics) for species, metrics in statistics.items()},
        "orthology_cache": str(orthology.ORTHO_TSV.relative_to(C.PROJECT_ROOT)),
        "go_and_example_gene_filter": (
            "protein-coding in both species according to cached orthology flags"
        ),
        **summary,
        "technical_families_excluded_from_display_only": list(
            EXAMPLE_GENE_EXCLUDED_PREFIXES
        ),
        "go_gene_sets": (
            "mutually exclusive conserved, mouse-specific, and human-specific "
            "top-decile dynamic orthologs"
        ),
        "dynamic_percentile_threshold": C.TEMPORAL_DYNAMIC_PERCENTILE_THRESHOLD,
    }
    return selected, manifest


def run(embedding: dict[str, Any]) -> dict[str, Any]:
    """Compute and cache all joint-embedding analysis metrics."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    selected, temporal_manifest = _cache_temporal_metrics(embedding)
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
    enrichment = _cached_go_enrichment(
        queries,
        cache_name="cluster_go_enrichment",
        panel="embryo_embedding",
        variant="stage_aware_membership",
    )
    groups = enrichment["query_group"].str.split(":", expand=True)
    enrichment["species"] = groups[0]
    enrichment["cluster"] = groups[1].astype(int)
    enrichment["global_qvalue"] = enrichment.groupby("species")["p_raw"].transform(
        lambda values: false_discovery_control(values.fillna(1).to_numpy(float))
    )
    enrichment["fold_enrichment"] = (
        enrichment["intersection_size"] / enrichment["query_size"]
    ) / (enrichment["term_size"] / enrichment["effective_domain_size"])
    enrichment.to_csv(RESULT_DIR / "cluster_go_enrichment.csv.gz", index=False)
    functional_labels = _cluster_functional_labels(enrichment)
    centres = centres.merge(functional_labels, on="cluster", validate="one_to_one")
    centres.to_csv(RESULT_DIR / "cluster_summary.csv", index=False)
    memberships.to_csv(RESULT_DIR / "gene_stage_cluster_membership.csv.gz", index=False)

    # Compute and save local mixing metrics
    _local_mixing_by_section(points).to_csv(
        RESULT_DIR / "local_mixing_by_section.csv", index=False
    )

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
        "n_strict_orthologs": temporal_manifest["n_orthologs"],
        "temporal_example_genes": selected["mouse_symbol"].astype(str).tolist(),
        "pca_explained_variance_ratio": embedding[
            "pca"
        ].explained_variance_ratio_.tolist(),
        "cluster_go_enrichment": {
            "assignment_unit": "species, gene, stage; slide replicates use median PCA coordinates",
            "query": "unique genes observed in each species-cluster across stages",
            "background": "all QC-passing embedded genes for that species",
            "source": {
                species: f"{C.GO_SOURCE} via local {orgdb}"
                for species, orgdb in C.GO_ORGDB.items()
            },
            "global_fdr_scope": "all cluster terms within each species",
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
        "embedding": embedding_manifest,
        "temporal": temporal_manifest,
        "local_mixing": mixing_manifest,
        "go": {
            "source": C.GO_SOURCE,
            "orgdb": C.GO_ORGDB,
            "semantic_simplify": C.GO_SEMANTIC_SIMPLIFY,
            "display_filters": {
                "fdr_threshold": C.GO_FDR_THRESHOLD,
                "maximum_term_universe_fraction": C.GO_MAX_TERM_UNIVERSE_FRACTION,
                "clusterprofiler_semantic_simplified_only": True,
                "terms_per_group": C.GO_DISPLAY_TERMS_PER_GROUP,
            },
        },
    }
    (RESULT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info("Wrote embedding metrics to %s", RESULT_DIR)
    return manifest


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(embeddings.load_spectra_and_fit_joint_embedding())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
