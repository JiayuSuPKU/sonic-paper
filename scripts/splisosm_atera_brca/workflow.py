from __future__ import annotations

import argparse
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.splisosm_atera_brca"

import pandas as pd

from scripts.paper_utils import write_workflow_manifest

from .config import (
    CACHE_REVISIONS,
    DU_CACHE_PARAMETERS,
    DU_GPR_BACKEND,
    DU_METHOD,
    FDR_THRESHOLD,
    FIGURE_DIR,
    MIN_BIN_PCT,
    MIN_COMPONENT_SIZE,
    MIN_COUNTS,
    RESULT_DIR,
    RHO,
    SV_CACHE_PARAMETERS,
    SV_METHOD,
)
from .du import run_du
from .plot_top20_svp_probe_transcript_panels import render_top20_figures
from .probe_mapping import prepare_svp_probe_mapping_cache
from .run_ankrd30a_analysis import run_ankrd30a_analysis
from .sv import run_sv, svp_target_genes
from .utils import ensure_dirs, project_path

# Cache-defining knobs are documented here and written to workflow_manifest.json.
CACHE_PARAMETERS = {"sv": SV_CACHE_PARAMETERS, "du": DU_CACHE_PARAMETERS}


def run_workflow(
    *,
    force: bool = False,
    n_jobs: int = 8,
    run_top20: bool = True,
    run_ankrd30a: bool = True,
) -> dict[str, Any]:
    """Build canonical caches and the default downstream inspection outputs."""
    started_at = time.perf_counter()
    ensure_dirs()

    # SPLISOSM is the only statistical engine in this workflow; everything
    # downstream annotates or summarizes its canonical segmented SV/DU outputs.
    svp, sv_metrics = run_sv(force=force, n_jobs=n_jobs)
    du, regulator_status, du_metrics = run_du(
        svp,
        force=force,
        n_jobs=n_jobs,
    )

    # Cache probe/genome/transcript mapping for every significant SVP gene, but
    # leave top-SVP figure rendering to the standalone plotting script/notebook.
    svp_genes = svp_target_genes(svp)
    svp_probe_cache = prepare_svp_probe_mapping_cache(svp_genes, force=force)

    if run_top20:
        # Top-20 PNGs are inspection artifacts; the two PDFs are notebook-only.
        top20_inventory = render_top20_figures(force=force)
    else:
        top20_inventory = pd.DataFrame()

    # ANKRD30A interpretation is explicit and optional; it remains separate
    # from the all-gene SV/DU/probe-mapping cache.
    if run_ankrd30a:
        tcga_correlations, tcga_regulators, ankrd30a_status = run_ankrd30a_analysis(
            du,
            force=force,
        )
    else:
        tcga_correlations = pd.DataFrame()
        tcga_regulators = []
        ankrd30a_status = {
            "download_status": {},
            "quantification_status": {"status": "skipped"},
            "n_prioritized_regulators": 0,
            "n_correlation_rows": 0,
        }

    metric_parts = [df for df in [sv_metrics, du_metrics] if not df.empty]
    metrics = (
        pd.concat(metric_parts, ignore_index=True, sort=False) if metric_parts else pd.DataFrame()
    )
    if not metrics.empty:
        # Combined runtime table lets the workflow summary stay compact.
        metrics.to_csv(RESULT_DIR / "runtime_memory_metrics.csv", index=False)

    summary = {
        "analysis": "splisosm_atera_brca",
        "splisosm_version": version("splisosm"),
        "sv_method": SV_METHOD,
        "du_method": DU_METHOD,
        "du_gpr_backend": DU_GPR_BACKEND,
        "fdr_threshold": FDR_THRESHOLD,
        "rho": RHO,
        "min_counts": MIN_COUNTS,
        "min_bin_pct": MIN_BIN_PCT,
        "min_component_size": MIN_COMPONENT_SIZE,
        "n_sv_genes": int(svp.shape[0]),
        "n_svp_genes_fdr_0_01": int((svp["pvalue_adj"] <= FDR_THRESHOLD).sum()),
        "n_du_pairs": int(du.shape[0]),
        "n_regulators_tested": int(regulator_status["tested_in_splisosmnp_du"].sum()),
        "n_svp_probe_mapping_genes_cached": len(svp_genes),
        "top20_rendered": bool(run_top20),
        "n_top20_svp_targets": int(top20_inventory.shape[0]),
        "ankrd30a_analysis_run": bool(run_ankrd30a),
        "ankrd30a_visualization_cache_status": ankrd30a_status.get(
            "visualization_cache_status", {}
        ),
        "tcga_quantification_run": ankrd30a_status.get("quantification_status", {}).get("status")
        in {"cached", "computed"},
        "tcga_quantification_status": ankrd30a_status.get("quantification_status", {}),
        "trps1_proseq_status": ankrd30a_status.get("trps1_proseq_status", {}),
        "tcga_clinical_expression_download_checked": bool(ankrd30a_status.get("download_status")),
        "tcga_clinical_expression_download_status": ankrd30a_status.get("download_status", {}),
        "n_tcga_regulator_correlations": int(tcga_correlations.shape[0]),
        "tcga_regulators": tcga_regulators,
        "outputs": {
            "results": project_path(RESULT_DIR),
            "figures": project_path(FIGURE_DIR),
            "cache_manifest": project_path(RESULT_DIR / "cache_manifest.json"),
            "svp_probe_mapping_cache": project_path(svp_probe_cache["mapping"]),
            "ankrd30a_visualization_cache": ankrd30a_status.get(
                "visualization_cache_status", {}
            ).get("outputs", []),
            "ankrd30a_trps1_proseq": ankrd30a_status.get("trps1_proseq_status", {}).get(
                "outputs", []
            ),
        },
    }
    manifest_path = write_workflow_manifest(
        "splisosm_atera_brca",
        RESULT_DIR,
        started_at=started_at,
        parameters={
            "cache_parameters": CACHE_PARAMETERS,
            "cache_revisions": CACHE_REVISIONS,
            "force": force,
            "n_jobs": n_jobs,
            "run_top20": run_top20,
            "run_ankrd30a": run_ankrd30a,
        },
        outputs=summary["outputs"],
        summary={key: value for key, value in summary.items() if key != "outputs"},
    )
    summary["outputs"]["workflow_manifest"] = project_path(manifest_path)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse the small orchestration CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical Atera FFPE human BRCA cache workflow: SV, DU, "
            "all-SVP probe mapping, top-20 PNGs, and ANKRD30A summaries."
        )
    )
    parser.add_argument("--force", action="store_true", help="Recompute cached outputs.")
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SPLISOSM_N_JOBS", "8")))
    parser.add_argument("--skip-top20", action="store_true", help="Skip top-20 SVP PNG rendering.")
    parser.add_argument(
        "--skip-ankrd30a",
        action="store_true",
        help="Skip ANKRD30A-specific long-read/TCGA summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_workflow(
        force=args.force,
        n_jobs=args.n_jobs,
        run_top20=not args.skip_top20,
        run_ankrd30a=not args.skip_ankrd30a,
    )


if __name__ == "__main__":
    main()
