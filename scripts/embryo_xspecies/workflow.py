"""Build every cached input used by the embryo visualization notebook.

The workflow downloads configured MOSTA/HESTA sections, prepares SpatialData,
builds strict orthology, computes per-section SV and radial spectra, fits the
joint PCA, and caches embedding and temporal metrics. Current stages are
skipped; ``--force`` rebuilds derived caches and ``--force-download`` also
refreshes raw data. Figures are rendered separately by ``visualization.ipynb``.
"""

from __future__ import annotations

import argparse
import logging

from . import config as C
from . import (
    download_hesta_data,
    download_mosta_data,
    embeddings,
    metrics,
    orthology,
    preprocessing,
    sv,
    temporal,
)

LOGGER = logging.getLogger("embryo_xspecies.workflow")


def run(
    *,
    force: bool = False,
    force_download: bool = False,
    progress: bool = False,
    download_workers: int = 6,
) -> dict:
    """Build or validate the complete analysis cache in dependency order."""
    C.ensure_output_dirs()

    LOGGER.info("Step 1/7: configured MOSTA and HESTA downloads")
    downloads = (
        ("MOSTA", download_mosta_data),
        ("HESTA", download_hesta_data),
    )
    for label, module in downloads:
        if not force_download and module.download_is_current():
            LOGGER.info("%s download cache is current; skipping.", label)
            continue
        status = module.run(force=force_download, workers=download_workers)
        if status or not module.download_is_current():
            raise RuntimeError(f"{label} download did not produce a complete cache.")

    rebuild_derived = force or force_download
    LOGGER.info("Step 2/7: species preprocessing")
    for cfg in C.SPECIES_CONFIGS.values():
        if not rebuild_derived and preprocessing.preprocessing_is_current(cfg):
            LOGGER.info("%s preprocessing cache is current; skipping.", cfg.PANEL_KEY)
            continue
        status = preprocessing.run_preprocessing(cfg, force=rebuild_derived)
        if status or not preprocessing.preprocessing_is_current(cfg):
            raise RuntimeError(
                f"{cfg.PANEL_KEY} preprocessing did not produce a complete cache."
            )

    LOGGER.info("Step 3/7: strict human-mouse orthology")
    orthology.prepare_orthologs(force=force)

    LOGGER.info("Step 4/7: per-section spatial variability")
    sv.run(force=rebuild_derived)

    LOGGER.info("Step 5/7: per-section radial spectra and joint PCA")
    embedding = embeddings.load_spectra_and_fit_joint_embedding(
        force=rebuild_derived,
        progress=progress,
    )

    LOGGER.info("Step 6/7: embedding metrics")
    metrics_manifest = metrics.run(embedding, force=rebuild_derived)

    LOGGER.info("Step 7/7: global temporal variation and stage transitions")
    temporal_manifest = temporal.run(embedding, force=rebuild_derived)
    return {
        "metrics": metrics_manifest,
        "temporal": temporal_manifest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild derived caches without re-downloading raw data.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download all raw sections and rebuild dependent caches.",
    )
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifests = run(
        force=args.force,
        force_download=args.force_download,
        progress=args.progress,
        download_workers=max(1, args.download_workers),
    )
    embedding = manifests["metrics"]["embedding"]
    temporal = manifests["temporal"]["global_variation"]
    LOGGER.info(
        "Complete: %d patterns, %d gene-stage memberships, temporal examples %s",
        embedding["n_gene_section_patterns"],
        embedding["n_gene_stage_memberships"],
        ", ".join(temporal["example_genes"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
