"""Cache global temporal variation and consecutive-stage transitions."""

from __future__ import annotations

import gc
import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr

from . import config as C
from . import embeddings, metrics, orthology, utils

LOGGER = logging.getLogger("embryo_xspecies.temporal")
RESULT_DIR = C.COMPARISON_RESULT_DIR / "temporal"
_EPS = 1e-12
_CACHE_ARTIFACTS = (
    "go_enrichment.csv.gz",
    "go_enrichment_manifest.json",
    "manifest.json",
    "metrics.csv.gz",
    "ortholog_temporal_conservation.csv.gz",
    "selected_genes.csv",
    "summary.csv",
    "temporal_gene_statistics.csv.gz",
    "temporal_go_enrichment.csv.gz",
    "temporal_go_enrichment_manifest.json",
)

# Technical families remain quantified but are not selected as figure examples.
_EXAMPLE_GENE_EXCLUDED_PREFIXES = (
    "MALAT1",
    "RPL",
    "RPS",
    "MRPL",
    "MRPS",
    "HBA",
    "HBB",
    "HIST",
)


def _input_signature(embedding: dict[str, Any]) -> str:
    """Return the signature of inputs consumed by temporal metrics."""
    paths = [
        Path(__file__),
        Path(C.__file__),
        Path(metrics.__file__),
        orthology.ORTHO_TSV,
        C.PROJECT_ROOT / "scripts" / "run_clusterprofiler.R",
    ]
    for cfg in C.SPECIES_CONFIGS.values():
        metadata = utils.filtered_metadata(cfg)
        paths.extend(
            cfg.SPECTRA_RESULT_DIR / f"{section_id}.npz"
            for section_id in metadata["section_id"].astype(str)
        )
    pca = embedding["pca"]
    return utils.cache_signature(
        paths,
        [
            embedding["freq_edges"],
            pca.components_,
            pca.mean_,
            pca.explained_variance_ratio_,
        ],
    )


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
    """Compare temporal-variation ranks across strict human-mouse orthologs."""
    mouse = mouse[mouse["passes_within_stage_filter"]]
    human = human[human["passes_within_stage_filter"]]
    table = (
        orthology.load_orthologs()
        .merge(mouse, left_on="mouse_symbol", right_on="gene")
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
    """Select one protein-coding example for each temporal-dynamics group."""
    conservation = conservation[
        conservation["mouse_protein_coding"] & conservation["human_protein_coding"]
    ]
    excluded = conservation["mouse_symbol"].str.upper().str.startswith(
        _EXAMPLE_GENE_EXCLUDED_PREFIXES
    ) | conservation["human_symbol"].str.upper().str.startswith(
        _EXAMPLE_GENE_EXCLUDED_PREFIXES
    )
    conservation = conservation[~excluded]
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
            selection_basis = (
                "highest focal temporal variation percentile among displayed "
                "GO term genes"
            )
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


def quantify_stage_movements(
    coordinates: np.ndarray,
    stage_labels: Sequence[str],
    *,
    feature_names: Sequence[str],
    stage_pairs: Sequence[tuple[str, str]],
    stage_values: Mapping[str, float],
) -> pd.DataFrame:
    """Estimate stage-centroid movement after slide and replicate correction.

    ``coordinates`` has shape ``(sections, features, dimensions)``. Section-wide
    displacement is removed by median-centering over features. The squared
    distance between consecutive-stage centroids is then reduced by the summed
    sampling variance of those centroids. The stage interval is retained as
    metadata for downstream visualization.
    """
    values = np.asarray(coordinates, dtype=np.float64)
    labels = np.asarray(stage_labels, dtype=str)
    features = np.asarray(feature_names, dtype=str)
    pairs = [(str(previous), str(current)) for previous, current in stage_pairs]
    if values.ndim != 3:
        raise ValueError(
            "coordinates must have shape (sections, features, dimensions)."
        )
    if labels.shape != (values.shape[0],):
        raise ValueError("stage_labels must contain one label per section.")
    if features.shape != (values.shape[1],):
        raise ValueError("feature_names must contain one name per feature.")
    required_stages = {stage for pair in pairs for stage in pair}
    if missing := required_stages - set(labels):
        raise ValueError(f"Missing stages: {sorted(missing)}")
    if missing := required_stages - set(stage_values):
        raise ValueError(f"Missing stage-axis values: {sorted(missing)}")
    if not np.isfinite(values).all():
        raise ValueError("coordinates contain non-finite values.")

    centered = values - np.median(values, axis=1, keepdims=True)
    frames = []
    for previous, current in pairs:
        previous_values = centered[labels == previous]
        current_values = centered[labels == current]
        if len(previous_values) < 2 or len(current_values) < 2:
            raise ValueError(
                f"{previous}->{current} requires at least two sections per stage."
            )
        interval = float(stage_values[current]) - float(stage_values[previous])
        if interval <= 0:
            raise ValueError(
                f"{previous}->{current} has non-positive stage interval {interval}."
            )

        displacement = current_values.mean(axis=0) - previous_values.mean(axis=0)
        squared_movement = np.square(displacement).sum(axis=1)  # len(features)
        # Adjust each feature for movement expected from finite replication.
        # Each feature is handled independently.
        noise_variance = previous_values.var(axis=0, ddof=1).sum(axis=1) / len(
            previous_values
        ) + current_values.var(axis=0, ddof=1).sum(axis=1) / len(current_values)
        adjusted = np.sqrt(np.maximum(squared_movement - noise_variance, 0))
        frames.append(
            pd.DataFrame(
                {
                    "feature": features,
                    "previous_stage": previous,
                    "current_stage": current,
                    "stage_interval": interval,
                    "background_centered_movement": np.sqrt(squared_movement),
                    "replicate_noise_variance": noise_variance,
                    "adjusted_movement": adjusted,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _test_table(result: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Standardize one QuadSV differential-test result table."""
    required = {"Feature", "Statistic", "P_value", "P_adj"}
    if missing := required - set(result):
        raise ValueError(
            f"QuadSV {prefix} result is missing columns: {sorted(missing)}"
        )
    return result[["Feature", "Statistic", "P_value", "P_adj"]].rename(
        columns={
            "Feature": "feature",
            "Statistic": f"{prefix}_statistic",
            "P_value": f"{prefix}_pvalue",
            "P_adj": f"{prefix}_fdr",
        }
    )


def _transition_tests(
    bundle: dict[str, Any],
    section_indices: np.ndarray,
    labels: np.ndarray,
    genes: list[str],
) -> pd.DataFrame:
    """Run DE, amplitude-sensitive DF, and shape-only DF through QuadSV."""
    comparator = embeddings.comparator_from_cached_spectra(
        bundle, section_indices, genes
    )
    design = labels[section_indices]
    tables = [
        _test_table(
            comparator.test_diff_expr(design, log_expression=False),
            "de",
        ),
        _test_table(
            comparator.test_diff_freq(
                design,
                statistic="log_l2",
                null="analytic",
                normalize_shape=False,
            ),
            "df",
        ),
        _test_table(
            comparator.test_diff_freq(
                design,
                statistic="log_l2",
                null="analytic",
                normalize_shape=True,
            ),
            "df_norm",
        ),
    ]
    merged = tables[0]
    for table in tables[1:]:
        merged = merged.merge(table, on="feature", validate="one_to_one")
    del comparator
    gc.collect()
    return merged


def _cache_global_variation(
    embedding: dict[str, Any],
) -> dict[str, Any]:
    """Cache all-stage variation, ortholog conservation, GO, and examples."""
    statistics: dict[str, pd.DataFrame] = {}
    embedding_dimensions: int | None = None

    # Quantify each species independently, treating same-stage slides as replicates.
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
        variation = quantify_temporal_variation(
            coordinates,
            bundle["meta"]["timepoint"].astype(str),
            feature_names=bundle["gene_names"],
            stage_order=C.SPECIES_CONFIGS[species].KEPT_TIMEPOINTS,
        ).rename(columns={"feature": "gene"})
        cutoff = variation["within_stage_variance"].quantile(
            C.TEMPORAL_MAX_WITHIN_STAGE_PERCENTILE
        )
        variation["passes_within_stage_filter"] = variation["within_stage_variance"].le(
            cutoff
        )
        statistics[species] = variation

    if embedding_dimensions is None:
        raise RuntimeError("No species embeddings were quantified.")
    mouse = statistics["mouse"]
    human = statistics["human"]
    conservation, summary = _temporal_conservation(mouse, human)
    coding = conservation[
        conservation["mouse_protein_coding"] & conservation["human_protein_coding"]
    ]
    mouse_universe = coding["mouse_symbol"].tolist()
    human_universe = coding["human_symbol"].tolist()
    enrichment = metrics.cached_go_enrichment(
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
        result_dir=RESULT_DIR,
    )
    displayed_terms = metrics.select_go_terms(
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
                variation.loc[
                    variation["passes_within_stage_filter"], "within_stage_variance"
                ].max()
            )
            for species, variation in statistics.items()
        },
        "interpreted_genes": {
            species: int(variation["passes_within_stage_filter"].sum())
            for species, variation in statistics.items()
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
        "genes": {species: len(variation) for species, variation in statistics.items()},
        "orthology_cache": str(orthology.ORTHO_TSV.relative_to(C.PROJECT_ROOT)),
        "go_and_example_gene_filter": (
            "protein-coding in both species according to cached orthology flags"
        ),
        **summary,
        "technical_families_excluded_from_display_only": list(
            _EXAMPLE_GENE_EXCLUDED_PREFIXES
        ),
        "go_gene_sets": (
            "mutually exclusive conserved, mouse-specific, and human-specific "
            "top-decile dynamic orthologs"
        ),
        "dynamic_percentile_threshold": C.TEMPORAL_DYNAMIC_PERCENTILE_THRESHOLD,
        "example_genes": selected["mouse_symbol"].astype(str).tolist(),
    }
    return manifest


def _summarize(
    transitions: pd.DataFrame,
    section_counts: Mapping[str, Mapping[str, int]],
) -> pd.DataFrame:
    """Summarize effect sizes and FDR event counts per species-transition."""
    rows = []
    for (species, previous, current), group in transitions.groupby(
        ["species", "previous_stage", "current_stage"], sort=False
    ):
        de = group["de_significant"]
        df = group["df_significant"]
        df_norm = group["df_norm_significant"]
        n_genes = len(group)
        rows.append(
            {
                "species": species,
                "previous_stage": previous,
                "current_stage": current,
                "previous_cs": group["previous_cs"].iat[0],
                "current_cs": group["current_cs"].iat[0],
                "n_genes": n_genes,
                "n_previous_sections": section_counts[species][previous],
                "n_current_sections": section_counts[species][current],
                "median_absolute_normalized_expression_change": group[
                    "absolute_normalized_expression_change"
                ].median(),
                "median_adjusted_movement": group["adjusted_movement"].median(),
                "n_de_significant": int(de.sum()),
                "n_df_significant": int(df.sum()),
                "n_df_norm_significant": int(df_norm.sum()),
                "fraction_de_significant": float(de.mean()),
                "fraction_df_significant": float(df.mean()),
                "fraction_df_norm_significant": float(df_norm.mean()),
            }
        )
    return pd.DataFrame(rows)


def run(embedding: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Compute or load global variation and consecutive-stage statistics."""
    signature = _input_signature(embedding)
    if (
        not force
        and (
            manifest := utils.current_cache_manifest(
                RESULT_DIR, _CACHE_ARTIFACTS, signature
            )
        )
        is not None
    ):
        LOGGER.info("Temporal cache is current: %s", RESULT_DIR)
        return manifest
    if force and RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    global_variation = _cache_global_variation(embedding)
    stages_by_species = {
        species: list(C.SPECIES_CONFIGS[species].KEPT_TIMEPOINTS)
        for species in ("mouse", "human")
    }
    stage_values = {
        "mouse": {
            stage: float(utils.mouse_e_to_carnegie(C.MOUSE.STAGE_TO_DAY[stage]))
            for stage in stages_by_species["mouse"]
        },
        "human": {
            stage: utils.human_label_to_carnegie(stage)
            for stage in stages_by_species["human"]
        },
    }

    frames = []
    section_counts: dict[str, dict[str, int]] = {}
    for species in ("mouse", "human"):
        bundle = embedding["species"][species]
        genes = list(map(str, bundle["gene_names"]))
        columns = np.asarray([bundle["name_to_col"][gene] for gene in genes])
        labels = bundle["meta"]["timepoint"].astype(str).to_numpy()
        section_counts[species] = {
            str(stage): int(count)
            for stage, count in zip(*np.unique(labels, return_counts=True), strict=True)
        }

        # Set up stage pairs for consecutive-stage comparisons.
        stages = stages_by_species[species]
        pairs = list(zip(stages[:-1], stages[1:], strict=True))

        # Compute section-background-centered PCA distance and replicate-noise variance.
        movements = quantify_stage_movements(
            bundle["coords"][:, columns, :],
            labels,
            feature_names=genes,
            stage_pairs=pairs,
            stage_values=stage_values[species],
        )
        if not bundle["presence"][:, columns].all():
            raise RuntimeError(
                f"{species} transition panel contains absent genes despite species-wide QC."
            )

        # Compute DE, DF, and DF-norm tests for each consecutive-stage transition.
        transition_frames = []
        for previous, current in pairs:
            indices = np.flatnonzero(np.isin(labels, [previous, current]))
            tests = _transition_tests(bundle, indices, labels, genes)
            # Compute mean log1p normalized expression change for each gene between the two stages.
            previous_dc = bundle["dc"][labels == previous][:, columns].mean(axis=0)
            current_dc = bundle["dc"][labels == current][:, columns].mean(axis=0)
            change = current_dc - previous_dc
            expression = pd.DataFrame(
                {
                    "feature": genes,
                    "previous_stage": previous,
                    "current_stage": current,
                    "previous_normalized_expression": previous_dc,
                    "current_normalized_expression": current_dc,
                    "normalized_expression_change": change,
                    "relative_normalized_expression_change": change
                    / (np.abs(previous_dc) + np.abs(current_dc) + _EPS),
                }
            )
            expression["absolute_normalized_expression_change"] = expression[
                "relative_normalized_expression_change"
            ].abs()
            transition_frames.append(
                expression.merge(tests, on="feature", validate="one_to_one")
            )

        transition_table = movements.merge(
            pd.concat(transition_frames, ignore_index=True),
            on=["feature", "previous_stage", "current_stage"],
            validate="one_to_one",
        )
        transition_table = transition_table.rename(columns={"feature": "gene"})
        transition_table.insert(0, "species", species)
        transition_table.insert(
            3,
            "previous_cs",
            transition_table["previous_stage"].map(stage_values[species]),
        )
        transition_table.insert(
            4,
            "current_cs",
            transition_table["current_stage"].map(stage_values[species]),
        )
        frames.append(transition_table)

    transitions = pd.concat(frames, ignore_index=True)
    for test in ("de", "df", "df_norm"):
        transitions[f"{test}_significant"] = transitions[f"{test}_fdr"].le(
            C.TRANSITION_FDR_THRESHOLD
        )
    transitions = transitions.sort_values(
        ["species", "current_cs", "adjusted_movement"],
        ascending=[True, True, False],
    )
    transitions.to_csv(RESULT_DIR / "metrics.csv.gz", index=False)
    summary = _summarize(transitions, section_counts)
    summary.to_csv(RESULT_DIR / "summary.csv", index=False)

    # Run GO enrichment for each transition's top hits that are DF-norm significant.
    queries = {}
    for (species, stage), group in transitions.groupby(
        ["species", "current_stage"], sort=True
    ):
        active = (
            group["adjusted_movement"]
            .rank(pct=True)
            .ge(C.TEMPORAL_DYNAMIC_PERCENTILE_THRESHOLD)
        )
        queries[f"{species}|{stage}"] = (
            group.loc[active & group["df_norm_significant"], "gene"],
            group["gene"],
            C.GO_ORGDB[species],
        )
    metrics.cached_go_enrichment(
        queries,
        cache_name="go_enrichment",
        panel="embryo_species_stage_transitions",
        variant="top_decile_adjusted_movement_df_norm_fdr",
        result_dir=RESULT_DIR,
    )

    manifest = {
        "cache_signature": signature,
        "artifacts": list(_CACHE_ARTIFACTS),
        "global_variation": global_variation,
        "consecutive_stage": {
            "stage_pairs": {
                species: [
                    list(pair) for pair in zip(stages[:-1], stages[1:], strict=True)
                ]
                for species, stages in stages_by_species.items()
            },
            "stage_axis": (
                "human labels parsed directly as CS; mouse E9-E16 mapped linearly "
                "to CS9-CS23 with utils.mouse_e_to_carnegie"
            ),
            "expression_change": (
                "post-minus-pre comparator DC (mean normalized log1p expression), "
                "plus a signed symmetric relative change bounded to [-1, 1]"
            ),
            "movement": (
                "section-background-centered PCA distance minus stage-centroid "
                "sampling variance"
            ),
            "tests": {
                "DE": "ComparatorGrid.test_diff_expr on cached DC values",
                "DF": (
                    "ComparatorGrid.test_diff_freq(log_l2, analytic, "
                    "normalize_shape=False)"
                ),
                "DF-norm": (
                    "ComparatorGrid.test_diff_freq(log_l2, analytic, "
                    "normalize_shape=True)"
                ),
                "fdr_scope": (
                    "all QC-passing genes within each species, transition, and test"
                ),
                "fdr_threshold": C.TRANSITION_FDR_THRESHOLD,
            },
            "replicates": "individual sections at each stage; no embryo averaging",
            "active_definition": "species-transition top decile of adjusted movement",
            "go_foreground_definition": (
                "active genes with significant shape-only DF-norm test "
                f"(FDR <= {C.TRANSITION_FDR_THRESHOLD})"
            ),
        },
    }
    (RESULT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info("Wrote temporal metrics to %s", RESULT_DIR)
    return manifest
