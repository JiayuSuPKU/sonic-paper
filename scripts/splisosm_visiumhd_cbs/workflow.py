from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import splisosm

from scripts.paper_utils import write_workflow_manifest
from scripts.splisosm_visiumhd_cbs.config import (
    CACHE_REVISIONS,
    DATASETS,
    DU_CACHE_PARAMETERS,
    DU_COVARIATE_MIN_BIN_PCT,
    DU_COVARIATE_MIN_COUNTS,
    DU_RBP_SUBSET,
    DU_TARGET_GENE_SUBSET,
    DU_TF_SUBSET,
    FDR_THRESHOLD,
    FIGURE_DIR,
    NEIGHBOR_DEGREE,
    PROJECT_ROOT,
    RESULT_DIR,
    RHO,
    SV_CACHE_PARAMETERS,
    SV_GENE_SUBSET,
)
from scripts.splisosm_visiumhd_cbs.du import (
    du_shared_associations,
    run_du_suite,
    subset_du_results_by_class,
)
from scripts.splisosm_visiumhd_cbs.sv import (
    prioritize_shared_hits,
    run_sv_suite,
)
from scripts.splisosm_visiumhd_cbs.utils import (
    ensure_dirs,
    log,
    neighbor_degrees_by_res,
    save_csv,
    save_json,
)

# Cache-defining knobs are documented here and written to workflow_manifest.json.
CACHE_PARAMETERS = {"sv": SV_CACHE_PARAMETERS, "du": DU_CACHE_PARAMETERS}


def run_workflow(*, force: bool = False, n_jobs: int = 8) -> dict[str, Any]:
    """Compute/cache all result tables used by the final visualization notebook.

    This intentionally does not render figures. The notebook owns all plotting so
    final panels and custom exploratory visualizations stay inspectable together.
    """
    started_at = time.perf_counter()
    ensure_dirs()
    log(f"splisosm: {getattr(splisosm, '__version__', 'unknown')} ({splisosm.__file__})")
    log(f"force={force}, n_jobs={n_jobs}")
    log(f"fixed neighbor_degree={NEIGHBOR_DEGREE}, rho={RHO}")
    if SV_GENE_SUBSET is not None:
        log(f"SV_GENE_SUBSET={SV_GENE_SUBSET}")
    if DU_TARGET_GENE_SUBSET is not None:
        log(f"DU_TARGET_GENE_SUBSET={DU_TARGET_GENE_SUBSET}")
    if DU_RBP_SUBSET is not None:
        log(f"DU_RBP_SUBSET={DU_RBP_SUBSET}")
    if DU_TF_SUBSET is not None:
        log(f"DU_TF_SUBSET={DU_TF_SUBSET}")

    # 1) SV calls plus one ranked shared-hit table for downstream prioritization.
    sv_results, sv_metrics = run_sv_suite(force=force, n_jobs=n_jobs)
    sv16 = {
        cfg.key: sv_results[(cfg.key, "16um")]
        for cfg in DATASETS.values()
        if (cfg.key, "16um") in sv_results
    }
    shared_hits = prioritize_shared_hits(sv16)
    save_csv(shared_hits, RESULT_DIR / "shared_svp_hits.csv")

    # 2) One combined RBP+TF DU run per dataset; class-specific outputs are
    # derived by filtering the combined table to avoid duplicate DU work.
    du_results, du_metrics = run_du_suite(
        sv_results,
        force=force,
        n_jobs=n_jobs,
    )

    # 3) Downstream DU summaries used directly by the Mbp/Gabbr1 figure cells.
    du_rbp_results = subset_du_results_by_class(du_results, "rbp")
    du_tf_results = subset_du_results_by_class(du_results, "tf_chip")
    du_rbp_shared = du_shared_associations(du_rbp_results, sv_results, covariate_set="rbp")
    du_tf_shared = du_shared_associations(du_tf_results, sv_results, covariate_set="tf_chip")
    save_csv(du_rbp_shared, RESULT_DIR / "du_rbp_shared_associations.csv")
    save_csv(du_tf_shared, RESULT_DIR / "du_tf_chip_shared_associations.csv")
    if not du_metrics.empty:
        save_csv(du_metrics, RESULT_DIR / "du_runtime_memory_metrics.csv")

    metric_parts = [df for df in [sv_metrics, du_metrics] if df is not None and not df.empty]
    metrics = (
        pd.concat(metric_parts, ignore_index=True, sort=False) if metric_parts else pd.DataFrame()
    )
    if not metrics.empty:
        save_csv(metrics, RESULT_DIR / "runtime_memory_metrics.csv")

    # 4) Lightweight provenance manifest for auditing cached outputs.
    summary = {
        "fdr": FDR_THRESHOLD,
        "rho": RHO,
        "splisosm_version": getattr(splisosm, "__version__", "unknown"),
        "neighbor_degrees": {
            f"{dataset}:{resolution}": degree
            for (dataset, resolution), degree in neighbor_degrees_by_res().items()
        },
        "sv_gene_subset": None if SV_GENE_SUBSET is None else list(SV_GENE_SUBSET),
        "du_target_gene_subset": (
            None if DU_TARGET_GENE_SUBSET is None else list(DU_TARGET_GENE_SUBSET)
        ),
        "du_covariate_min_counts": DU_COVARIATE_MIN_COUNTS,
        "du_covariate_min_bin_pct": DU_COVARIATE_MIN_BIN_PCT,
        "du_rbp_subset": None if DU_RBP_SUBSET is None else list(DU_RBP_SUBSET),
        "du_tf_subset": None if DU_TF_SUBSET is None else list(DU_TF_SUBSET),
        "n_sv_results": len(sv_results),
        "n_shared_hits": int(shared_hits.shape[0]) if shared_hits is not None else 0,
        "n_du_regulator_tables": len(du_results),
        "n_du_rbp_shared_associations": int(du_rbp_shared.shape[0]),
        "n_du_tf_chip_shared_associations": int(du_tf_shared.shape[0]),
        "outputs": {
            "results": str(RESULT_DIR.relative_to(PROJECT_ROOT)),
            "figures": str(FIGURE_DIR.relative_to(PROJECT_ROOT)),
            "cache_manifest": str((RESULT_DIR / "cache_manifest.json").relative_to(PROJECT_ROOT)),
        },
    }
    save_json(summary, RESULT_DIR / "summary.json")
    manifest_path = write_workflow_manifest(
        "splisosm_visiumhd_cbs",
        RESULT_DIR,
        started_at=started_at,
        parameters={
            "cache_parameters": CACHE_PARAMETERS,
            "cache_revisions": CACHE_REVISIONS,
            "force": force,
            "n_jobs": n_jobs,
        },
        outputs=summary["outputs"],
        summary={key: value for key, value in summary.items() if key != "outputs"},
    )
    summary["outputs"]["workflow_manifest"] = str(manifest_path.relative_to(PROJECT_ROOT))
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI options for cached workflow execution."""
    parser = argparse.ArgumentParser(
        description="Run and cache Visium HD mouse CBS SV/DU test results."
    )
    parser.add_argument("--force", action="store_true", help="Recompute cached outputs.")
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SPLISOSM_N_JOBS", "8")))
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_workflow(force=args.force, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
