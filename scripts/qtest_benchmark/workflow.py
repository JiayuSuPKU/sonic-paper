"""Compute the cached inputs used by the Q-test visualization notebook."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.qtest_benchmark"

from .benchmark import cache_runtime_scaling_results, cache_validation_benchmark_results
from .config import RAW_SPATIALDATA
from .small_intestine import cache_small_intestine_kernel_results

LOGGER = logging.getLogger(__name__)


def generate_analysis_caches(
    *,
    analysis: str = "all",
    source: Path | None = None,
    force: bool = False,
) -> None:
    """Generate one analysis cache group or all default cache groups."""
    if analysis in {"all", "small-intestine"}:
        LOGGER.info("Running small-intestine analysis")
        cache_small_intestine_kernel_results(source or RAW_SPATIALDATA, force=force)
    if analysis in {"all", "synthetic"}:
        LOGGER.info("Running synthetic-data analyses")
        cache_runtime_scaling_results(force=force)
        cache_validation_benchmark_results(force=force)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        choices=("all", "synthetic", "small-intestine"),
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
