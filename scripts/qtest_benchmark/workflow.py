"""Compute the cached inputs used by the Q-test visualization notebook."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from scripts.paper_utils import (
    PROJECT_ROOT,
    _load_cache_manifest,
    cache_entry_is_current,
    cache_signature,
    installed_version,
    path_fingerprint,
    project_relative,
    update_cache_entry,
    write_workflow_manifest,
)

from .benchmark import (
    BENCHMARK_SAMPLE_SIZES,
    POWER_SIGNAL_TO_NOISE_RATIOS,
    RUNTIME_IMPLEMENTATIONS,
    VALIDATION_CACHE_FILES,
    VALIDATION_KERNEL_PARAMETERS,
    cache_runtime_scaling_results,
    cache_validation_benchmark_results,
    runtime_provenance,
)
from .config import (
    ARTIFACT_GENES,
    BENCHMARK_PROJECTION_FILE,
    BENCHMARK_RUNS_FILE,
    BENCHMARK_SCALING_FILE,
    BENCHMARK_TARGETS,
    CAR_K_NEIGHBORS,
    CAR_RHOS,
    FAST_DECAY_RHO,
    IMMUNE_GENES,
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
    PIXEL_SIZE_UM,
    RAW_SPATIALDATA,
    RESULT_DIR,
    SLOW_DECAY_RHO,
    SMALL_INTESTINE_PATTERN_RANKS_FILE,
    SMALL_INTESTINE_SCORES_FILE,
    VALIDATION_RESULT_DIR,
)
from .lineage import (
    ANCESTRY_WEIGHTS,
    BIVARIATE_MIN_PAIRS,
    BIVARIATE_SIGMA_THRESHOLDS,
    CAR_RHO,
    CLUSTER_MIN_CELLS,
    EXAMPLE_TUMORS,
    FDR_CUTOFF,
    GENE_MIN_CELLS_FRAC,
    GENE_PANEL,
    MAX_GENE_ANALYSIS_CELLS,
    MIN_TUMOR_CELLS,
    PAIR_GENES,
    cache_lineage_results,
)
from .small_intestine import (
    DEFAULT_CHUNK_SIZE,
    MIN_TOTAL_COUNTS,
    PATTERN_SMOOTHING_SIGMA,
    cache_small_intestine_kernel_results,
)
from .utils import source_provenance

LOGGER = logging.getLogger(__name__)

# Bump a section revision only when computation changes outside its explicit parameters.
CACHE_REVISIONS = {"validation": 2, "small-intestine": 1, "lineage": 1}

SYNTHETIC_CACHE_PARAMETERS = {
    "runtime": {
        "implementations": [
            {
                "name": implementation.name,
                "max_n": implementation.max_n,
                "null_method": implementation.null_method,
                "kind": implementation.kind,
            }
            for implementation in RUNTIME_IMPLEMENTATIONS
        ],
        "sample_sizes": BENCHMARK_SAMPLE_SIZES,
        "targets": BENCHMARK_TARGETS,
    },
    "validation": {
        "kernel_parameters": VALIDATION_KERNEL_PARAMETERS,
        "power_signal_to_noise_ratios": POWER_SIGNAL_TO_NOISE_RATIOS,
    },
}

SMALL_INTESTINE_CACHE_PARAMETERS = {
    "artifact_genes": ARTIFACT_GENES,
    "car_k_neighbors": CAR_K_NEIGHBORS,
    "car_rhos": CAR_RHOS,
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "fast_decay_rho": FAST_DECAY_RHO,
    "fft_solver": "fft2",
    "immune_genes": IMMUNE_GENES,
    "min_total_counts": MIN_TOTAL_COUNTS,
    "pattern_smoothing_sigma": PATTERN_SMOOTHING_SIGMA,
    "pixel_size_um": PIXEL_SIZE_UM,
    "slow_decay_rho": SLOW_DECAY_RHO,
    "workers": 1,
}

LINEAGE_CACHE_PARAMETERS = {
    "ancestry_weights": ANCESTRY_WEIGHTS,
    "backend": "matrix",
    "bivariate_min_pairs": BIVARIATE_MIN_PAIRS,
    "bivariate_sigma_thresholds": BIVARIATE_SIGMA_THRESHOLDS,
    "car_rho": CAR_RHO,
    "cluster_min_cells": CLUSTER_MIN_CELLS,
    "example_tumors": EXAMPLE_TUMORS,
    "fdr_cutoff": FDR_CUTOFF,
    "gene_min_cells_frac": GENE_MIN_CELLS_FRAC,
    "gene_panel": GENE_PANEL,
    "max_gene_analysis_cells": MAX_GENE_ANALYSIS_CELLS,
    "min_tumor_cells": MIN_TUMOR_CELLS,
    "n_jobs": 1,
    "pair_genes": PAIR_GENES,
}


# Canonical cache parameters are kept here so every invalidating knob is auditable.
CACHE_PARAMETERS = {
    "runtime": SYNTHETIC_CACHE_PARAMETERS["runtime"],
    "validation": SYNTHETIC_CACHE_PARAMETERS["validation"],
    "small-intestine": SMALL_INTESTINE_CACHE_PARAMETERS,
    "lineage": LINEAGE_CACHE_PARAMETERS,
}


def _group_signature_data(
    group: str,
    parameters: dict[str, Any],
    *inputs: Path,
    recursive: tuple[Path, ...] = (),
) -> dict[str, Any]:
    return {
        "source": runtime_provenance() if group == "validation" else source_provenance(__file__),
        "cache_revision": CACHE_REVISIONS[group],
        "dependencies": {"sonic-spatial": installed_version("sonic-spatial")},
        "inputs": [path_fingerprint(path, recursive=path.is_dir()) for path in inputs]
        + [path_fingerprint(path, recursive=True) for path in recursive],
        "parameters": parameters,
        "section": group,
    }


def generate_analysis_caches(
    *,
    analysis: str = "all",
    source: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Generate one analysis cache group or all default cache groups."""
    started_at = time.perf_counter()
    valid_analyses = {"all", "runtime", "validation", "small-intestine", "lineage"}
    if analysis not in valid_analyses:
        raise ValueError(f"Unknown analysis {analysis!r}; choose from {sorted(valid_analyses)}")
    if source is not None and analysis not in {"all", "small-intestine"}:
        raise ValueError("source applies only to the small-intestine analysis")

    outputs: dict[str, str] = {}
    cache_groups: list[str] = []
    if analysis in {"all", "runtime"}:
        LOGGER.info("Running runtime benchmarks (8 cores / 16 GiB projection)")
        cache_runtime_scaling_results(force=force)
        cache_groups.append("runtime")
        outputs.update(
            {
                "benchmark_scaling": project_relative(BENCHMARK_SCALING_FILE),
                "benchmark_projection": project_relative(BENCHMARK_PROJECTION_FILE),
                "benchmark_runs": project_relative(BENCHMARK_RUNS_FILE),
            }
        )
    if analysis in {"all", "validation"}:
        LOGGER.info("Running implementation-agreement, null-calibration, and power validation")
        validation_outputs = tuple(
            VALIDATION_RESULT_DIR / name for name in VALIDATION_CACHE_FILES.values()
        )
        signature_data = _group_signature_data("validation", CACHE_PARAMETERS["validation"])
        signature = cache_signature(signature_data)
        refresh = force or not cache_entry_is_current(
            RESULT_DIR,
            "validation",
            signature,
            outputs=validation_outputs,
        )
        cache_validation_benchmark_results(force=refresh)
        if refresh:
            update_cache_entry(
                RESULT_DIR,
                "validation",
                signature,
                validation_outputs,
                signature_data=signature_data,
            )
        cache_groups.append("validation")
        outputs["validation"] = project_relative(VALIDATION_RESULT_DIR)
    if analysis in {"all", "small-intestine"}:
        LOGGER.info("Running small-intestine analysis")
        effective_source = source if source is not None else RAW_SPATIALDATA
        small_intestine_outputs = (
            SMALL_INTESTINE_SCORES_FILE,
            SMALL_INTESTINE_PATTERN_RANKS_FILE,
        )
        signature_data = _group_signature_data(
            "small-intestine",
            CACHE_PARAMETERS["small-intestine"],
            effective_source,
        )
        signature = cache_signature(signature_data)
        refresh = force or not cache_entry_is_current(
            RESULT_DIR,
            "small-intestine",
            signature,
            outputs=small_intestine_outputs,
        )
        cache_small_intestine_kernel_results(effective_source, force=refresh)
        if refresh:
            update_cache_entry(
                RESULT_DIR,
                "small-intestine",
                signature,
                small_intestine_outputs,
                signature_data=signature_data,
            )
        cache_groups.append("small-intestine")
        outputs.update(
            {
                "small_intestine_scores": project_relative(SMALL_INTESTINE_SCORES_FILE),
                "small_intestine_pattern_ranks": project_relative(
                    SMALL_INTESTINE_PATTERN_RANKS_FILE
                ),
            }
        )
    if analysis in {"all", "lineage"}:
        LOGGER.info("Running lineage analysis")
        lineage_outputs = (
            LINEAGE_CLUSTER_SCORES_FILE,
            LINEAGE_GENE_SCORES_FILE,
            LINEAGE_BIVARIATE_SUMMARY_FILE,
            LINEAGE_GENE_DIAGNOSTICS_FILE,
            LINEAGE_BIVARIATE_DIAGNOSTICS_FILE,
            LINEAGE_TARGET_METADATA_FILE,
            LINEAGE_QC_FILE,
        )
        signature_data = _group_signature_data(
            "lineage",
            CACHE_PARAMETERS["lineage"],
            LINEAGE_ANALYSIS_DATA_FILE,
            recursive=(LINEAGE_TREE_DIR,),
        )
        signature = cache_signature(signature_data)
        refresh = force or not cache_entry_is_current(
            RESULT_DIR,
            "lineage",
            signature,
            outputs=lineage_outputs,
        )
        cache_lineage_results(force=refresh)
        if refresh:
            update_cache_entry(
                RESULT_DIR,
                "lineage",
                signature,
                lineage_outputs,
                signature_data=signature_data,
            )
        cache_groups.append("lineage")
        outputs.update(
            {
                "lineage_qc": project_relative(LINEAGE_QC_FILE),
                "lineage_cluster_scores": project_relative(LINEAGE_CLUSTER_SCORES_FILE),
                "lineage_gene_scores": project_relative(LINEAGE_GENE_SCORES_FILE),
                "lineage_bivariate_summary": project_relative(LINEAGE_BIVARIATE_SUMMARY_FILE),
                "lineage_gene_diagnostics": project_relative(LINEAGE_GENE_DIAGNOSTICS_FILE),
                "lineage_bivariate_diagnostics": project_relative(
                    LINEAGE_BIVARIATE_DIAGNOSTICS_FILE
                ),
                "lineage_target_metadata": project_relative(LINEAGE_TARGET_METADATA_FILE),
            }
        )

    outputs["cache_manifest"] = project_relative(RESULT_DIR / "cache_manifest.json")
    manifest_path = write_workflow_manifest(
        "qtest_benchmark",
        RESULT_DIR,
        started_at=started_at,
        parameters={
            "analysis": analysis,
            "cache_parameters": CACHE_PARAMETERS,
            "cache_revisions": CACHE_REVISIONS,
            "force": force,
            "source": project_relative(source) if source is not None else None,
        },
        outputs=outputs,
        summary={"cache_groups": cache_groups},
    )
    outputs["workflow_manifest"] = project_relative(manifest_path)
    return {"outputs": outputs, "cache_groups": cache_groups}


def require_current_figure_cache(group: str) -> Path | None:
    """Reject stale plot inputs and return the scoring run's SpatialData source.

    Validate the direct cache manifests so interactive cache generation is also
    supported. Recorded settings describe the run; source, inputs and outputs
    must still match. This function never computes an analysis.
    """
    locations = {
        "runtime": (
            RESULT_DIR,
            "runtime",
            (BENCHMARK_SCALING_FILE, BENCHMARK_PROJECTION_FILE, BENCHMARK_RUNS_FILE),
        ),
        "validation": (
            VALIDATION_RESULT_DIR,
            "validation",
            tuple(VALIDATION_RESULT_DIR / name for name in VALIDATION_CACHE_FILES.values()),
        ),
        "small-intestine": (
            RESULT_DIR,
            "small-intestine-scores",
            (SMALL_INTESTINE_SCORES_FILE, SMALL_INTESTINE_PATTERN_RANKS_FILE),
        ),
        "lineage": (
            LINEAGE_RESULT_DIR,
            "lineage",
            (
                LINEAGE_CLUSTER_SCORES_FILE,
                LINEAGE_GENE_SCORES_FILE,
                LINEAGE_BIVARIATE_SUMMARY_FILE,
                LINEAGE_GENE_DIAGNOSTICS_FILE,
                LINEAGE_TARGET_METADATA_FILE,
                LINEAGE_BIVARIATE_DIAGNOSTICS_FILE,
                LINEAGE_QC_FILE,
            ),
        ),
    }
    result_dir, section, outputs = locations[group]
    entry = _load_cache_manifest(result_dir / "cache_manifest.json")["entries"].get(section, {})
    recorded = entry.get("signature_data") if isinstance(entry, dict) else None
    message = (
        f"Stale or missing {group} cache; rerun workflow.py --analysis {group} before plotting."
    )
    if not isinstance(recorded, dict):
        raise RuntimeError(message)
    current = dict(recorded)
    if group == "runtime":
        current["provenance"] = runtime_provenance()
    else:
        current["source"] = source_provenance(__file__)
    paths = []
    try:
        for fingerprint in recorded.get("inputs", []):
            path = Path(fingerprint["path"])
            paths.append(path if path.is_absolute() else PROJECT_ROOT / path)
        if "inputs" in recorded:
            current["inputs"] = [path_fingerprint(path, recursive=path.is_dir()) for path in paths]
    except (KeyError, TypeError, OSError) as exc:
        raise RuntimeError(message) from exc
    if not cache_entry_is_current(result_dir, section, cache_signature(current), outputs=outputs):
        raise RuntimeError(message)
    if group == "small-intestine":
        if len(paths) != 1:
            raise RuntimeError(message)
        return paths[0].resolve()
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        choices=("all", "runtime", "validation", "small-intestine", "lineage"),
        default="all",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Optional path to the raw Visium HD SpatialData zarr.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute existing caches.")
    args = parser.parse_args()
    generate_analysis_caches(
        analysis=args.analysis,
        source=args.source,
        force=args.force,
    )


if __name__ == "__main__":
    main()
