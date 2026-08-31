"""Compute the cached inputs used by the Q-test visualization notebook."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from scripts.paper_utils import (
    cache_entry_is_current,
    cache_signature,
    installed_version,
    path_fingerprint,
    project_relative,
    update_cache_entry,
    write_workflow_manifest,
)

from .benchmark import (
    VALIDATION_CACHE_FILES,
    cache_runtime_scaling_results,
    cache_validation_benchmark_results,
)
from .config import (
    BENCHMARK_PROJECTION_FILE,
    BENCHMARK_SCALING_FILE,
    LINEAGE_ANALYSIS_DATA_FILE,
    LINEAGE_BIVARIATE_DIAGNOSTICS_FILE,
    LINEAGE_BIVARIATE_SUMMARY_FILE,
    LINEAGE_CLUSTER_SCORES_FILE,
    LINEAGE_GENE_DIAGNOSTICS_FILE,
    LINEAGE_GENE_SCORES_FILE,
    LINEAGE_TARGET_METADATA_FILE,
    LINEAGE_TREE_DIR,
    RAW_SPATIALDATA,
    RESULT_DIR,
    SMALL_INTESTINE_PATTERN_RANKS_FILE,
    SMALL_INTESTINE_SCORES_FILE,
    VALIDATION_RESULT_DIR,
)
from .lineage import cache_lineage_results
from .small_intestine import cache_small_intestine_kernel_results

LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent


def _group_signature(group: str, *inputs: Path, recursive: tuple[Path, ...] = ()) -> str:
    return cache_signature(
        {
            "group": group,
            "inputs": [path_fingerprint(path) for path in inputs]
            + [path_fingerprint(path, recursive=True) for path in recursive],
            "sonic_version": installed_version("sonic-spatial"),
        }
    )


def generate_analysis_caches(
    *,
    analysis: str = "all",
    source: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Generate one analysis cache group or all default cache groups."""
    started_at = time.perf_counter()
    valid_analyses = {"all", "synthetic", "small-intestine", "lineage"}
    if analysis not in valid_analyses:
        raise ValueError(f"Unknown analysis {analysis!r}; choose from {sorted(valid_analyses)}")
    if source is not None and analysis not in {"all", "small-intestine"}:
        raise ValueError("source applies only to the small-intestine analysis")

    outputs: dict[str, str] = {}
    cache_groups: list[str] = []
    if analysis in {"all", "synthetic"}:
        LOGGER.info("Running synthetic-data analyses")
        synthetic_outputs = (
            BENCHMARK_SCALING_FILE,
            BENCHMARK_PROJECTION_FILE,
            *(VALIDATION_RESULT_DIR / name for name in VALIDATION_CACHE_FILES.values()),
        )
        signature = _group_signature(
            "synthetic",
            SCRIPT_DIR / "benchmark.py",
            SCRIPT_DIR / "config.py",
            SCRIPT_DIR / "utils.py",
        )
        refresh = force or not cache_entry_is_current(
            RESULT_DIR,
            "synthetic",
            signature,
            outputs=synthetic_outputs,
        )
        cache_runtime_scaling_results(force=refresh)
        cache_validation_benchmark_results(force=refresh)
        if refresh:
            update_cache_entry(RESULT_DIR, "synthetic", signature, synthetic_outputs)
        cache_groups.append("synthetic")
        outputs.update(
            {
                "benchmark_scaling": project_relative(BENCHMARK_SCALING_FILE),
                "benchmark_projection": project_relative(BENCHMARK_PROJECTION_FILE),
                "validation": project_relative(VALIDATION_RESULT_DIR),
            }
        )
    if analysis in {"all", "small-intestine"}:
        LOGGER.info("Running small-intestine analysis")
        effective_source = source if source is not None else RAW_SPATIALDATA
        small_intestine_outputs = (
            SMALL_INTESTINE_SCORES_FILE,
            SMALL_INTESTINE_PATTERN_RANKS_FILE,
        )
        signature = _group_signature(
            "small-intestine",
            SCRIPT_DIR / "small_intestine.py",
            SCRIPT_DIR / "config.py",
            SCRIPT_DIR / "utils.py",
            effective_source,
        )
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
        )
        signature = _group_signature(
            "lineage",
            SCRIPT_DIR / "lineage.py",
            SCRIPT_DIR / "config.py",
            SCRIPT_DIR / "utils.py",
            LINEAGE_ANALYSIS_DATA_FILE,
            recursive=(LINEAGE_TREE_DIR,),
        )
        refresh = force or not cache_entry_is_current(
            RESULT_DIR,
            "lineage",
            signature,
            outputs=lineage_outputs,
        )
        cache_lineage_results(force=refresh)
        if refresh:
            update_cache_entry(RESULT_DIR, "lineage", signature, lineage_outputs)
        cache_groups.append("lineage")
        outputs.update(
            {
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
            "force": force,
            "source": project_relative(source) if source is not None else None,
        },
        outputs=outputs,
        summary={"cache_groups": cache_groups},
    )
    outputs["workflow_manifest"] = project_relative(manifest_path)
    return {"outputs": outputs, "cache_groups": cache_groups}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        choices=("all", "synthetic", "small-intestine", "lineage"),
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
