"""Compute the cached inputs used by the Q-test visualization notebook."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .benchmark import cache_runtime_scaling_results, cache_validation_benchmark_results
from .config import RAW_SPATIALDATA
from .lineage import cache_lineage_results
from .small_intestine import cache_small_intestine_kernel_results

LOGGER = logging.getLogger(__name__)


def generate_analysis_caches(
    *,
    analysis: str = "all",
    source: Path | None = None,
    force: bool = False,
) -> None:
    """Generate one analysis cache group or all default cache groups."""
    valid_analyses = {"all", "synthetic", "small-intestine", "lineage"}
    if analysis not in valid_analyses:
        raise ValueError(
            f"Unknown analysis {analysis!r}; choose from {sorted(valid_analyses)}"
        )
    if source is not None and analysis not in {"all", "small-intestine"}:
        raise ValueError("source applies only to the small-intestine analysis")

    if analysis in {"all", "synthetic"}:
        LOGGER.info("Running synthetic-data analyses")
        cache_runtime_scaling_results(force=force)
        cache_validation_benchmark_results(force=force)
    if analysis in {"all", "small-intestine"}:
        LOGGER.info("Running small-intestine analysis")
        cache_small_intestine_kernel_results(
            source if source is not None else RAW_SPATIALDATA, force=force
        )
    if analysis in {"all", "lineage"}:
        LOGGER.info("Running lineage analysis")
        cache_lineage_results(force=force)


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
    parser.add_argument(
        "--force", action="store_true", help="Recompute existing caches."
    )
    args = parser.parse_args()
    generate_analysis_caches(
        analysis=args.analysis,
        source=args.source,
        force=args.force,
    )


if __name__ == "__main__":
    main()
