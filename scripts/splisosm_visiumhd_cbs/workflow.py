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

from scripts.paper_utils import (
    cache_entry_is_current,
    cache_signature,
    path_fingerprint,
    update_cache_entry,
    write_workflow_manifest,
)
from scripts.splisosm_utils import source_provenance
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
from scripts.splisosm_visiumhd_cbs.reference_annotations import remap_tf_symbols
from scripts.splisosm_visiumhd_cbs.sv import (
    prioritize_shared_hits,
    run_sv_suite,
)
from scripts.splisosm_visiumhd_cbs.utils import (
    ensure_dirs,
    log,
    neighbor_degrees_by_res,
    save_csv,
)

# Cache-defining knobs are documented here and written to workflow_manifest.json.
CACHE_PARAMETERS = {"sv": SV_CACHE_PARAMETERS, "du": DU_CACHE_PARAMETERS}


def _summary_signature_data() -> dict:
    return {
        "source": source_provenance(__file__),
        "inputs": [
            path_fingerprint(path)
            for cfg in DATASETS.values()
            for path in (
                *(RESULT_DIR / f"sv_{cfg.key}_{res}_hsic_ir.csv" for res in ("16um", "8um", "2um")),
                RESULT_DIR / f"du_regulator_{cfg.key}_16um_hsic_gp.csv",
            )
        ],
    }


def require_current_figure_cache() -> None:
    """Reject stale or mixed SV, DU and derived summaries before plotting."""
    from scripts.paper_utils import require_files

    from .config import REMAP_TF_TARGETS
    from .du import _du_cache_signature_data, target_genes_for_du
    from .sv import _sv_cache_signature_data
    from .utils import normalize_gene_symbol

    require_files((REMAP_TF_TARGETS,), label="cached ReMap regulator list; run the workflow first")

    remap_tf_symbols(read_only=True)
    degrees = neighbor_degrees_by_res()
    for cfg in DATASETS.values():
        isoforms = None
        for res in ("16um", "8um", "2um"):
            data = _sv_cache_signature_data(
                cfg, res, degrees[(cfg.key, res)], isoforms, SV_GENE_SUBSET
            )
            if not cache_entry_is_current(RESULT_DIR, f"sv:{cfg.key}:{res}", cache_signature(data)):
                raise RuntimeError(
                    f"Stale {cfg.key} {res} SV cache; rerun splisosm_visiumhd_cbs.workflow"
                )
            if res == "16um":
                isoforms = (
                    pd.read_csv(RESULT_DIR / f"eligible_16um_features_{cfg.key}.csv")["isoform"]
                    .astype(str)
                    .tolist()
                )
        sv = pd.read_csv(RESULT_DIR / f"sv_{cfg.key}_16um_hsic_ir.csv")
        keys = {normalize_gene_symbol(g) for g in target_genes_for_du(cfg, sv)}
        data = _du_cache_signature_data(cfg, keys, degrees[(cfg.key, "16um")])
        if not cache_entry_is_current(RESULT_DIR, f"du:{cfg.key}:16um", cache_signature(data)):
            raise RuntimeError(f"Stale {cfg.key} DU cache; rerun splisosm_visiumhd_cbs.workflow")
    if not cache_entry_is_current(
        RESULT_DIR, "summaries", cache_signature(_summary_signature_data())
    ):
        raise RuntimeError("Stale Visium summaries/metrics; rerun splisosm_visiumhd_cbs.workflow")


def run_workflow(*, force: bool = False, n_jobs: int = 8) -> dict[str, Any]:
    """Compute/cache all result tables used by the final visualization notebook.

    This intentionally does not render figures. The notebook owns all plotting so
    final panels and custom exploratory visualizations stay inspectable together.
    """
    started_at = time.perf_counter()
    ensure_dirs()
    remap_tf_symbols(force=force)
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
    # 2) One combined RBP+TF DU run per dataset; class-specific outputs are
    # derived by filtering the combined table to avoid duplicate DU work.
    du_results, du_metrics = run_du_suite(
        sv_results,
        force=force,
        n_jobs=n_jobs,
    )

    # 3) Reuse derived summaries without rewriting valid output fingerprints.
    summary_data = _summary_signature_data()
    summary_paths = tuple(
        RESULT_DIR / name
        for name in (
            "shared_svp_hits.csv",
            "du_rbp_shared_associations.csv",
            "du_tf_shared_associations.csv",
            "runtime_memory_metrics.csv",
        )
    )
    if not force and cache_entry_is_current(
        RESULT_DIR, "summaries", cache_signature(summary_data), outputs=summary_paths
    ):
        shared_hits, du_rbp_shared, du_tf_shared = (pd.read_csv(path) for path in summary_paths[:3])
    else:
        sv16 = {cfg.key: sv_results[(cfg.key, "16um")] for cfg in DATASETS.values()}
        shared_hits = prioritize_shared_hits(sv16)
        du_rbp_shared = du_shared_associations(
            subset_du_results_by_class(du_results, "rbp"), sv_results, covariate_set="rbp"
        )
        du_tf_shared = du_shared_associations(
            subset_du_results_by_class(du_results, "tf_chip"), sv_results, covariate_set="tf_chip"
        )
        parts = [df for df in (sv_metrics, du_metrics) if df is not None and not df.empty]
        metrics = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
        for frame, path in zip(
            (shared_hits, du_rbp_shared, du_tf_shared, metrics), summary_paths, strict=True
        ):
            save_csv(frame, path)
        update_cache_entry(
            RESULT_DIR,
            "summaries",
            cache_signature(summary_data),
            summary_paths,
            signature_data=summary_data,
        )

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
