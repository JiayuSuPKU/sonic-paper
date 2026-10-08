"""Run ANKRD30A-specific long-read, Atera DU, and TCGA validation caches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.splisosm_atera_brca"

import pandas as pd

from scripts.paper_utils import (
    cache_entry_is_current,
    cache_signature,
    path_fingerprint,
    update_cache_entry,
)
from scripts.splisosm_utils import source_provenance

from .ankrd30a.atera_du import (
    ankrd30a_regulator_priority_table,
    select_ankrd30a_focus_regulators,
)
from .ankrd30a.download_tcga_brca_clinical_expression import (
    ensure_tcga_clinical_expression_data,
)
from .ankrd30a.long_read import (
    _write_ankrd30a_visualization_cache,
    prepare_ankrd30a_long_read_annotations,
)
from .ankrd30a.run_trps1_proseq import run_trps1_proseq
from .ankrd30a.tcga import (
    compute_regulator_correlations,
    load_tcga_metrics,
    prepare_tcga_regulator_expression,
    quantify_tcga_ankrd30a,
)
from .config import (
    ANKRD30A_REGULATOR_PRIORITY_CSV,
    DU_CSV,
    RESULT_DIR,
    TCGA_CBIO_RSEM_TXT,
    TCGA_ER_POSITIVE_CORRELATION_CSV,
    TCGA_METRIC_TABLE_TSV,
)
from .utils import log, save_csv


def _correlation_signature_data() -> dict:
    return {
        "source": source_provenance(__file__),
        "inputs": [
            path_fingerprint(path)
            for path in (
                DU_CSV,
                ANKRD30A_REGULATOR_PRIORITY_CSV,
                TCGA_METRIC_TABLE_TSV,
                TCGA_CBIO_RSEM_TXT,
            )
        ],
    }


def run_ankrd30a_analysis(
    du: pd.DataFrame,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    """Run ANKRD30A long-read, regulator-priority, and TCGA summary caches."""
    pacbio_tx, pacbio_features, _sqanti = prepare_ankrd30a_long_read_annotations(force=force)
    # Source downloads remain idempotent here; redownload/debug controls belong
    # in the dedicated download scripts rather than this analysis wrapper.
    download_status = ensure_tcga_clinical_expression_data(force=False)
    quant_status = quantify_tcga_ankrd30a(force=force)
    # metrics is one row per TCGA-BRCA sample with clinical ER status plus
    # ANKRD30A PB transcript counts/usages and probe-region counts/usages.
    metrics = load_tcga_metrics(force=force or quant_status.get("status") == "computed")
    priority = ankrd30a_regulator_priority_table(du, force=force)
    visualization_cache_status = _write_ankrd30a_visualization_cache(
        pacbio_tx,
        pacbio_features,
        priority,
        force=force,
    )
    proseq_status = run_trps1_proseq(force=force)
    regulators = select_ankrd30a_focus_regulators(priority)
    # TCGA expression is loaded only for the prioritized regulators to avoid
    # caching a large derived expression table; expr is a long table of raw
    # cBioPortal RSEM values, log1p-transformed inside the correlation test.
    signature_data = _correlation_signature_data()
    if not force and cache_entry_is_current(
        RESULT_DIR,
        "tcga_correlations",
        cache_signature(signature_data),
        outputs=(TCGA_ER_POSITIVE_CORRELATION_CSV,),
    ):
        correlations = pd.read_csv(TCGA_ER_POSITIVE_CORRELATION_CSV)
    else:
        expr = prepare_tcga_regulator_expression(regulators)
        correlations = compute_regulator_correlations(metrics, regulators, expr)
        save_csv(correlations, TCGA_ER_POSITIVE_CORRELATION_CSV)
        update_cache_entry(
            RESULT_DIR,
            "tcga_correlations",
            cache_signature(signature_data),
            (TCGA_ER_POSITIVE_CORRELATION_CSV,),
            signature_data=signature_data,
        )
    status = {
        "visualization_cache_status": visualization_cache_status,
        "download_status": download_status,
        "quantification_status": quant_status,
        "trps1_proseq_status": proseq_status,
        "n_pacbio_ankrd30a_transcripts": int(pacbio_tx.shape[0]),
        "n_prioritized_regulators": len(regulators),
        "n_correlation_rows": int(correlations.shape[0]),
    }
    return correlations, regulators, status


def parse_args() -> argparse.Namespace:
    """Parse the standalone ANKRD30A analysis CLI."""
    parser = argparse.ArgumentParser(
        description="Run ANKRD30A long-read, regulator, and TCGA validation summaries."
    )
    parser.add_argument(
        "--du-csv", type=Path, default=DU_CSV, help="Canonical Atera DU result CSV."
    )
    parser.add_argument("--force", action="store_true", help="Rebuild derived TCGA summary caches.")
    return parser.parse_args()


def main() -> None:
    """Run the ANKRD30A-specific analysis workflow."""
    args = parse_args()
    if not args.du_csv.exists():
        raise FileNotFoundError(f"Missing DU result CSV: {args.du_csv}")
    du = pd.read_csv(args.du_csv)
    correlations, regulators, status = run_ankrd30a_analysis(
        du,
        force=args.force,
    )
    log(
        "ANKRD30A analysis complete: "
        f"{correlations.shape[0]:,} correlation rows, "
        f"{len(regulators):,} prioritized regulators, "
        f"TCGA quantification={status.get('quantification_status', {}).get('status', 'unknown')}, "
        f"TRPS1 PRO-seq={status.get('trps1_proseq_status', {}).get('status', 'unknown')}"
    )


if __name__ == "__main__":
    main()
