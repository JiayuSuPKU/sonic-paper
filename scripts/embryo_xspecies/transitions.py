"""Quantify species-specific expression and spatial-pattern transitions."""

from __future__ import annotations

import gc
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from . import config as C
from . import embedding_metrics, embeddings, utils

LOGGER = logging.getLogger("embryo_xspecies.transitions")
RESULT_DIR = C.COMPARISON_RESULT_DIR / "stage_transitions"
_EPS = 1e-12


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


def run(embedding: dict[str, Any]) -> dict[str, Any]:
    """Compute each species' consecutive-stage transition statistics."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
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

        metrics = movements.merge(
            pd.concat(transition_frames, ignore_index=True),
            on=["feature", "previous_stage", "current_stage"],
            validate="one_to_one",
        )
        metrics = metrics.rename(columns={"feature": "gene"})
        metrics.insert(0, "species", species)
        metrics.insert(
            3, "previous_cs", metrics["previous_stage"].map(stage_values[species])
        )
        metrics.insert(
            4, "current_cs", metrics["current_stage"].map(stage_values[species])
        )
        frames.append(metrics)

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
    embedding_metrics.cached_go_enrichment(
        queries,
        cache_name="go_enrichment",
        panel="embryo_species_stage_transitions",
        variant="top_decile_adjusted_movement_df_norm_fdr",
        result_dir=RESULT_DIR,
    )

    manifest = {
        "stage_pairs": {
            species: [list(pair) for pair in zip(stages[:-1], stages[1:], strict=True)]
            for species, stages in stages_by_species.items()
        },
        "stage_axis": (
            "human labels parsed directly as CS; mouse E9-E16 mapped linearly "
            "to CS9-CS23 with utils.mouse_e_to_carnegie"
        ),
        "expression_change": (
            "post-minus-pre comparator DC (mean normalized log1p expression), plus "
            "a signed symmetric relative change bounded to [-1, 1]"
        ),
        "movement": (
            "section-background-centered PCA distance minus stage-centroid "
            "sampling variance"
        ),
        "tests": {
            "DE": "ComparatorGrid.test_diff_expr on cached DC values",
            "DF": "ComparatorGrid.test_diff_freq(log_l2, analytic, normalize_shape=False)",
            "DF-norm": "ComparatorGrid.test_diff_freq(log_l2, analytic, normalize_shape=True)",
            "fdr_scope": "all QC-passing genes within each species, transition, and test",
            "fdr_threshold": C.TRANSITION_FDR_THRESHOLD,
        },
        "replicates": "individual sections at each stage; no embryo averaging",
        "active_definition": "species-transition top decile of adjusted movement",
        "go_foreground_definition": (
            "active genes with significant shape-only DF-norm test "
            f"(FDR <= {C.TRANSITION_FDR_THRESHOLD})"
        ),
        "artifacts": [
            "metrics.csv.gz",
            "summary.csv",
            "go_enrichment.csv.gz",
            "go_enrichment_manifest.json",
            "manifest.json",
        ],
    }
    (RESULT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info("Wrote stage-transition metrics to %s", RESULT_DIR)
    return manifest


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(embeddings.load_spectra_and_fit_joint_embedding())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
