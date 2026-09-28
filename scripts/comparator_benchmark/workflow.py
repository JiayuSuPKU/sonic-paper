"""Run and cache comparator-benchmark analysis inputs.

AD/PIG caches the full binary benchmark grid, radial-bin robustness diagnostics,
and reference extras used by the finalized AD/PIG figures. Brain-dev caches a
continuous-time benchmark grid plus marker controls, anisotropy, and enrichment
contrasts. Figure layout stays in ``visualization.ipynb``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.comparator_benchmark"

from sonic.statistics import apply_bh_correction

from scripts.paper_utils import (
    cache_entry_is_current,
    cache_signature,
    installed_version,
    path_fingerprint,
    project_relative,
    run_clusterprofiler,
    update_cache_entry,
    write_workflow_manifest,
)

from .ad_pig_helpers import (
    ad_pig_anisotropy_cache_ready,
    ad_pig_cache_has_current_qc,
    ad_pig_spectral_replicate_distance_cache_ready,
)
from .brain_dev_helpers import (
    brain_dev_anisotropy_cache_ready,
    brain_dev_forced_controls,
    brain_dev_positive_control_table,
)
from .config import (
    AD_ENRICH_CONTRAST_FILE,
    AD_PIG_ANISOTROPY_FILE,
    AD_PIG_DROP_PROBES,
    AD_PIG_ENRICHMENT_TOP_N,
    AD_PIG_FDR_ALPHA,
    AD_PIG_FULL_BINNINGS,
    AD_PIG_FULL_FEATURE_MODES,
    AD_PIG_FULL_INPUTS,
    AD_PIG_FULL_NULL_ECDF_FILE,
    AD_PIG_FULL_TESTS_FILE,
    AD_PIG_POS_CONTROLS,
    AD_PIG_RADIAL_BIN_ROBUSTNESS_FILE,
    AD_PIG_REFERENCE_BINNING,
    AD_PIG_REFERENCE_FEATURE_MODE,
    AD_PIG_REFERENCE_INPUT,
    AD_PIG_REFERENCE_N_RADIAL_BINS,
    AD_PIG_RENAME_PROBES,
    AD_PIG_SPECTRAL_REPLICATE_DISTANCE_FILE,
    BRAIN_DEV_ANISOTROPY_FILE,
    BRAIN_DEV_CELL_TYPE_CONTROL_GROUPS,
    BRAIN_DEV_CELL_TYPE_CONTROL_TOP_N,
    BRAIN_DEV_CELL_TYPE_LABELS,
    BRAIN_DEV_CELL_TYPE_MANUAL_GENES,
    BRAIN_DEV_CELL_TYPE_PROGRAM_TOP_N,
    BRAIN_DEV_ENRICH_CONTRAST_FILE,
    BRAIN_DEV_ENRICHMENT_TOP_N,
    BRAIN_DEV_FULL_FEATURE_MODES,
    BRAIN_DEV_FULL_INPUTS,
    BRAIN_DEV_NULL_ECDF_FILE,
    BRAIN_DEV_NULL_PERMUTATIONS,
    BRAIN_DEV_POSITIVE_CONTROLS_FILE,
    BRAIN_DEV_REFERENCE_BINNING,
    BRAIN_DEV_REFERENCE_FEATURE_MODE,
    BRAIN_DEV_REFERENCE_INPUT,
    BRAIN_DEV_REFERENCE_N_RADIAL_BINS,
    BRAIN_DEV_TABLE_S4_CNMFS,
    BRAIN_DEV_TESTS_FILE,
    MAX_GENES,
    N_RADIAL_BINS,
    PANELS,
    PRESENCE_THRESHOLD,
    PROJECT_ROOT,
    RADIAL_BIN_DIAGNOSTIC_BINS,
    RESULT_DIR,
    SPATIAL_TESTS,
    TEST_LABELS,
)
from .utils import (
    cache_has_columns,
    de_test_binary,
    de_test_continuous,
    ensure_dirs,
    fit_all_samples,
    geometric_frequency_edges,
    load_panel,
    log,
    log1p_normalized_samples,
    resolve_metadata_path,
    save_csv,
    select_gene_panel,
)

# Key statistical knobs are explicit; bump a revision for algorithm-only changes.
CACHE_REVISIONS = {"ad_pig": 1, "brain_dev": 1}
COMMON_CACHE_PARAMETERS = {
    "n_radial_bins": N_RADIAL_BINS,
    "spatial_tests": SPATIAL_TESTS,
    "test_labels": TEST_LABELS,
    "enrichment": {
        "sources": ["GO:BP", "KEGG"],
        "orgdb": "org.Mm.eg.db",
        "organism": "mmu",
    },
}
CACHE_PARAMETERS = {
    "ad_pig": {
        **COMMON_CACHE_PARAMETERS,
        "full_inputs": AD_PIG_FULL_INPUTS,
        "full_binnings": AD_PIG_FULL_BINNINGS,
        "full_feature_modes": AD_PIG_FULL_FEATURE_MODES,
        "reference_input": AD_PIG_REFERENCE_INPUT,
        "reference_binning": AD_PIG_REFERENCE_BINNING,
        "reference_feature_mode": AD_PIG_REFERENCE_FEATURE_MODE,
        "reference_n_radial_bins": AD_PIG_REFERENCE_N_RADIAL_BINS,
        "radial_bin_diagnostics": RADIAL_BIN_DIAGNOSTIC_BINS,
        "fdr_alpha": AD_PIG_FDR_ALPHA,
        "enrichment_top_n": AD_PIG_ENRICHMENT_TOP_N,
        "positive_controls": AD_PIG_POS_CONTROLS,
        "drop_probes": AD_PIG_DROP_PROBES,
        "rename_probes": AD_PIG_RENAME_PROBES,
    },
    "brain_dev": {
        **COMMON_CACHE_PARAMETERS,
        "full_inputs": BRAIN_DEV_FULL_INPUTS,
        "full_feature_modes": BRAIN_DEV_FULL_FEATURE_MODES,
        "reference_input": BRAIN_DEV_REFERENCE_INPUT,
        "reference_binning": BRAIN_DEV_REFERENCE_BINNING,
        "reference_feature_mode": BRAIN_DEV_REFERENCE_FEATURE_MODE,
        "reference_n_radial_bins": BRAIN_DEV_REFERENCE_N_RADIAL_BINS,
        "null_permutations": BRAIN_DEV_NULL_PERMUTATIONS,
        "enrichment_top_n": BRAIN_DEV_ENRICHMENT_TOP_N,
        "cell_type_program_top_n": BRAIN_DEV_CELL_TYPE_PROGRAM_TOP_N,
        "cell_type_control_top_n": BRAIN_DEV_CELL_TYPE_CONTROL_TOP_N,
        "cell_type_control_groups": BRAIN_DEV_CELL_TYPE_CONTROL_GROUPS,
        "cell_type_manual_genes": BRAIN_DEV_CELL_TYPE_MANUAL_GENES,
        "cell_type_labels": BRAIN_DEV_CELL_TYPE_LABELS,
    },
}


def _panel_cache_signature_data(
    panel: str,
    *,
    presence_threshold: float,
    max_genes: int,
) -> dict[str, Any]:
    panel_config = PANELS[panel]
    inputs = [panel_config.metadata_csv]
    if panel == "brain_dev":
        inputs.append(BRAIN_DEV_TABLE_S4_CNMFS)
    if panel_config.metadata_csv.exists():
        try:
            metadata = pd.read_csv(panel_config.metadata_csv, usecols=["h5ad_path"])
        except (OSError, ValueError, pd.errors.EmptyDataError):
            pass
        else:
            inputs.extend(
                resolve_metadata_path(path, panel_config.metadata_csv)
                for path in metadata["h5ad_path"].dropna().astype(str)
            )
    return {
        "cache_revision": CACHE_REVISIONS[panel],
        "dependencies": {"sonic-spatial": installed_version("sonic-spatial")},
        "inputs": [path_fingerprint(path) for path in inputs],
        "parameters": {
            **CACHE_PARAMETERS[panel],
            "panel": panel,
            "presence_threshold": presence_threshold,
            "max_genes": max_genes,
        },
    }


def _run_guarded_panel(
    panel: str,
    runner: Callable[..., dict[str, Any]],
    *,
    force: bool,
    presence_threshold: float,
    max_genes: int,
) -> dict[str, Any]:
    signature_data = _panel_cache_signature_data(
        panel,
        presence_threshold=presence_threshold,
        max_genes=max_genes,
    )
    signature = cache_signature(signature_data)
    refresh = force or not cache_entry_is_current(RESULT_DIR, panel, signature)
    result = runner(
        force=refresh,
        presence_threshold=presence_threshold,
        max_genes=max_genes,
    )
    owned_outputs = [
        output for output in result["outputs"].values() if isinstance(output, (str, Path))
    ]
    if refresh or result.get("provenance") is not None:
        update_cache_entry(
            RESULT_DIR,
            panel,
            signature,
            owned_outputs,
            signature_data=signature_data,
            provenance=result.get("provenance"),
        )
    return result


def _ad_pig_full_cache_ready(test_path: Path, null_path: Path) -> bool:
    if not cache_has_columns(
        test_path,
        {
            "panel",
            "scenario_id",
            "input",
            "binning",
            "feature_mode",
            "requested_n_radial_bins",
            "n_radial_bins",
            "test",
            "Feature",
            "P_value",
            "P_adj",
        },
    ):
        return False
    if not cache_has_columns(
        null_path,
        {
            "panel",
            "scenario_id",
            "input",
            "binning",
            "feature_mode",
            "requested_n_radial_bins",
            "n_radial_bins",
            "perm_idx",
            "test",
            "P_value",
        },
    ):
        return False
    if not ad_pig_cache_has_current_qc(test_path):
        return False
    try:
        tests = pd.read_csv(
            test_path,
            usecols=[
                "panel",
                "input",
                "binning",
                "feature_mode",
                "requested_n_radial_bins",
                "n_radial_bins",
                "test",
            ],
        )
        null = pd.read_csv(
            null_path,
            usecols=[
                "panel",
                "input",
                "binning",
                "feature_mode",
                "requested_n_radial_bins",
                "n_radial_bins",
                "test",
            ],
        )
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False
    expected = {
        (input_transform, binning, feature_mode, N_RADIAL_BINS)
        for input_transform in AD_PIG_FULL_INPUTS
        for binning in AD_PIG_FULL_BINNINGS
        for feature_mode in AD_PIG_FULL_FEATURE_MODES
    }
    # Validate against the requested count; collapsed log grids can have fewer
    # actual bins while still representing the configured scenario.
    observed_scenarios = set(
        tests.loc[
            tests["panel"].eq("ad_pig"),
            ["input", "binning", "feature_mode", "requested_n_radial_bins"],
        ]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    null_scenarios = set(
        null.loc[
            null["panel"].eq("ad_pig"),
            ["input", "binning", "feature_mode", "requested_n_radial_bins"],
        ]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_scenarios != expected or null_scenarios != expected:
        return False
    observed_tests = set(tests.loc[tests["panel"].eq("ad_pig"), "test"].dropna().astype(str))
    null_tests = set(null.loc[null["panel"].eq("ad_pig"), "test"].dropna().astype(str))
    return observed_tests == set(TEST_LABELS) and null_tests == {test for test, _ in SPATIAL_TESTS}


def _brain_dev_cache_ready(test_path: Path, null_path: Path) -> bool:
    """Return whether brain-dev observed and permutation caches are current."""
    if not cache_has_columns(
        test_path,
        {
            "panel",
            "scenario_id",
            "input",
            "binning",
            "feature_mode",
            "requested_n_radial_bins",
            "n_radial_bins",
            "test",
            "Feature",
            "P_value",
            "P_adj",
        },
    ):
        return False
    if not cache_has_columns(
        null_path,
        {
            "panel",
            "scenario_id",
            "input",
            "binning",
            "feature_mode",
            "requested_n_radial_bins",
            "n_radial_bins",
            "perm_idx",
            "test",
            "P_value",
        },
    ):
        return False
    try:
        tests = pd.read_csv(
            test_path,
            usecols=[
                "panel",
                "input",
                "binning",
                "feature_mode",
                "requested_n_radial_bins",
                "test",
            ],
        )
        null = pd.read_csv(
            null_path,
            usecols=[
                "panel",
                "input",
                "binning",
                "feature_mode",
                "requested_n_radial_bins",
                "perm_idx",
                "test",
            ],
        )
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False

    expected = {
        (
            input_transform,
            BRAIN_DEV_REFERENCE_BINNING,
            feature_mode,
            BRAIN_DEV_REFERENCE_N_RADIAL_BINS,
        )
        for input_transform in BRAIN_DEV_FULL_INPUTS
        for feature_mode in BRAIN_DEV_FULL_FEATURE_MODES
    }
    observed_scenarios = set(
        tests.loc[
            tests["panel"].eq("brain_dev"),
            ["input", "binning", "feature_mode", "requested_n_radial_bins"],
        ]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    null_scenarios = set(
        null.loc[
            null["panel"].eq("brain_dev"),
            ["input", "binning", "feature_mode", "requested_n_radial_bins"],
        ]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_scenarios != expected or null_scenarios != expected:
        return False
    observed_tests = set(tests.loc[tests["panel"].eq("brain_dev"), "test"].dropna().astype(str))
    null_tests = set(null.loc[null["panel"].eq("brain_dev"), "test"].dropna().astype(str))
    if observed_tests != set(TEST_LABELS) or null_tests != {test for test, _ in SPATIAL_TESTS}:
        return False

    null_perm_counts = (
        null.loc[null["panel"].eq("brain_dev")]
        .groupby(["input", "binning", "feature_mode", "requested_n_radial_bins", "test"])[
            "perm_idx"
        ]
        .nunique()
    )
    expected_null_groups = len(expected) * len(SPATIAL_TESTS)
    return (
        null_perm_counts.size == expected_null_groups
        and int(null_perm_counts.min()) == BRAIN_DEV_NULL_PERMUTATIONS
        and int(null_perm_counts.max()) == BRAIN_DEV_NULL_PERMUTATIONS
    )


def _select_comparison_genes(
    panel: str,
    samples: dict[Path, Any],
    presence_threshold: float = PRESENCE_THRESHOLD,
    max_genes: int = MAX_GENES,
) -> list[str]:
    """Select benchmark genes, forcing present curated controls into each panel."""
    genes = select_gene_panel(samples, presence_threshold, max_genes)
    if panel not in PANELS:
        return genes

    present = set.intersection(*(set(adata.var_names) for adata in samples.values()))
    controls = [gene for gene in PANELS[panel].pos_controls if gene in present]
    if panel == "brain_dev":
        controls.extend(gene for gene in brain_dev_forced_controls() if gene in present)
    return list(dict.fromkeys(genes + controls))


def _subset_design(sample_set: Any, paths_order: list[Path]) -> dict[str, Any]:
    # Convert stored sample-set metadata into the compact design object expected
    # by the comparator API. Binary and continuous contrasts share this shape.
    path_to_idx = {path: i for i, path in enumerate(paths_order)}
    configured_paths = [*sample_set.a_paths, *sample_set.b_paths]
    missing = [path for path in configured_paths if path not in path_to_idx]
    if missing:
        raise KeyError(f"Configured samples are absent from the fitted comparator: {missing}")

    if sample_set.design_values is not None:
        idx_list = [path_to_idx[path] for path in sample_set.a_paths]
        if not idx_list:
            raise ValueError("Configured continuous comparison has no samples")
        if len(sample_set.design_values) != len(idx_list):
            raise ValueError("Configured design values do not align with its samples")
        return {
            "is_design": True,
            "idx": np.asarray(idx_list, dtype=int),
            "design_values": sample_set.design_values,
            "design_name": sample_set.design_name,
        }

    a_idx = [path_to_idx[path] for path in sample_set.a_paths]
    b_idx = [path_to_idx[path] for path in sample_set.b_paths]
    if not a_idx or not b_idx:
        raise ValueError("Configured binary comparison has an empty sample group")
    return {
        "is_design": False,
        "idx": np.asarray(a_idx + b_idx, dtype=int),
        "groups": np.asarray(["A"] * len(a_idx) + ["B"] * len(b_idx)),
    }


def _spatial_test(
    comparator: Any,
    subset: dict[str, Any],
    *,
    normalize_shape: bool = False,
) -> pd.DataFrame:
    # ``comparator`` already owns precomputed spectra. ``subset`` only selects
    # sample rows and supplies the current real or permuted sample-level design.
    sub = comparator.subset(subset["idx"])
    if subset["is_design"]:
        design_name = subset["design_name"]
        return sub.test_diff_freq(
            pd.DataFrame({design_name: subset["design_values"]}),
            contrast=design_name,
            statistic="log_l2",
            null="analytic",
            normalize_shape=normalize_shape,
        )
    return sub.test_diff_freq(
        subset["groups"],
        statistic="log_l2",
        null="analytic",
        normalize_shape=normalize_shape,
    )


def _de_test(comparator: Any, subset: dict[str, Any], genes: list[str]) -> pd.DataFrame:
    if subset["is_design"]:
        return de_test_continuous(
            comparator.dc_[subset["idx"]],
            subset["design_values"],
            genes,
            design_name=subset["design_name"],
        )
    return de_test_binary(comparator.dc_[subset["idx"]], subset["groups"], genes)


def _observed_test_rows(
    *,
    comparator: Any,
    subset: dict[str, Any],
    genes: list[str],
    meta: dict[str, Any],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    # DE is the scalar/DC companion test; DF and DF_norm use the same spectra
    # with and without L1 shape normalization.
    for test, frame in (
        ("DE", _de_test(comparator, subset, genes)),
        ("DF", _spatial_test(comparator, subset, normalize_shape=False)),
        ("DF_norm", _spatial_test(comparator, subset, normalize_shape=True)),
    ):
        out = frame.copy()
        if "P_value" not in out.columns:
            raise ValueError("Expected a P_value column")
        if "P_adj" not in out.columns:
            out["P_adj"] = apply_bh_correction(out["P_value"].to_numpy(float))
        keep = [col for col in ("Feature", "Statistic", "P_value", "P_adj") if col in out.columns]
        out = out[keep].copy()
        out.insert(0, "test", test)
        for key, value in reversed(meta.items()):
            out.insert(0, key, value)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def _null_ecdf_rows(
    *,
    comparator: Any,
    subset: dict[str, Any],
    perm_idx: int,
    meta: dict[str, Any],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    # The permutation cache is only used for ECDF calibration plots, so it keeps
    # spatial p-values and scenario metadata rather than full per-gene records.
    # Spectra are not recomputed here; permutations only reshuffle groups/design
    # values before the analytic DF/DF_norm test is run on the fitted spectra.
    for test, normalize_shape in SPATIAL_TESTS:
        # Run the spatial test for the current scenario.
        frame = _spatial_test(comparator, subset, normalize_shape=normalize_shape)
        # Store only the necessary information for the ECDF.
        rows.append(
            pd.DataFrame(
                {
                    **meta,
                    "perm_idx": int(perm_idx),
                    "test": test,
                    "P_value": frame["P_value"].to_numpy(float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _fit_scenario_samples(
    samples: dict[Path, Any],
    genes: list[str],
    *,
    binning: str,
    feature_mode: str,
    n_radial_bins: int,
) -> tuple[Any, list[Path], int]:
    def _actual_bin_count(comparator: Any, fallback: int) -> int:
        # Due to adaptive binning, the actual bin number can be fewer than requested
        intervals = getattr(comparator, "frequency_bin_intervals_", None)
        if intervals is not None:
            intervals = np.asarray(intervals)
            if intervals.ndim == 2 and intervals.shape[0] > 0:
                return int(intervals.shape[0])
        fitted_n = getattr(comparator, "_n_radial_bins", None)
        if fitted_n is not None:
            return int(fitted_n)
        return int(fallback)

    if binning == "linear":
        comparator, paths_order = fit_all_samples(
            samples, genes, feature_mode, n_radial_bins=n_radial_bins
        )
        actual_n_bins = _actual_bin_count(comparator, n_radial_bins)
        return comparator, paths_order, int(actual_n_bins)
    if binning != "log":
        raise ValueError(f"Unknown binning: {binning!r}")

    # Build log edges from the radial grid first, then reuse those frequency
    # edges for either radial or 2D feature extraction in the scenario grid.
    # This temporary fit is only used to recover the requested physical
    # frequency range; the actual scenario fit below lets SONIC adapt bins.
    linear_cmp, _paths = fit_all_samples(
        samples,
        genes,
        "radial",
        n_radial_bins=n_radial_bins,
        adaptive_freq_edges=False,
    )
    log_edges = geometric_frequency_edges(np.asarray(linear_cmp.freq_edges))
    comparator, paths_order = fit_all_samples(
        samples,
        genes,
        feature_mode,
        n_radial_bins=n_radial_bins,
        freq_edges=log_edges,
    )
    actual_n_bins = _actual_bin_count(comparator, log_edges.size - 1)
    return comparator, paths_order, int(actual_n_bins)


def run_ad_pig_full_scenarios(  # noqa: C901
    *,
    force: bool = False,
    presence_threshold: float = PRESENCE_THRESHOLD,
    max_genes: int = MAX_GENES,
) -> dict[str, Any]:
    """Cache observed tests and permutation-null ECDF p-values for all AD/PIG scenarios."""
    test_path = RESULT_DIR / AD_PIG_FULL_TESTS_FILE
    null_path = RESULT_DIR / AD_PIG_FULL_NULL_ECDF_FILE
    if _ad_pig_full_cache_ready(test_path, null_path) and not force:
        log("using cached AD/PIG full scenario tests; pass --force to recompute")
        return {
            "ad_pig_full_tests": str(test_path.relative_to(PROJECT_ROOT)),
            "ad_pig_full_null_ecdf": str(null_path.relative_to(PROJECT_ROOT)),
        }

    ensure_dirs()

    # Load the AD/PIG panel data and select genes for comparison.
    samples, _meta, sample_sets = load_panel("ad_pig")
    genes = _select_comparison_genes("ad_pig", samples, presence_threshold, max_genes)
    real_set = next(s for s in sample_sets if s.null_type == "real")
    perm_sets = [s for s in sample_sets if s.null_type == "perm"]
    # Normalize each input representation once, then reuse it across the
    # binning x feature-mode grid.
    samples_by_input: dict[str, dict[Path, Any]] = {}
    for input_transform in AD_PIG_FULL_INPUTS:
        if input_transform == "raw":
            samples_by_input[input_transform] = samples
        elif input_transform == "log1p":
            samples_by_input[input_transform] = log1p_normalized_samples(samples)
        else:
            raise ValueError(f"Unknown input transform: {input_transform!r}")

    # Run comparison for the full AD/PIG scenarios.
    observed_parts: list[pd.DataFrame] = []
    null_parts: list[pd.DataFrame] = []
    for input_transform in AD_PIG_FULL_INPUTS:
        for binning in AD_PIG_FULL_BINNINGS:
            for feature_mode in AD_PIG_FULL_FEATURE_MODES:
                log(
                    "AD/PIG full scenario "
                    f"{input_transform}_{binning}_{feature_mode} "
                    f"requested_bins={N_RADIAL_BINS}"
                )
                # Compute per-sample spectra once for the current scenario.
                # Real and permuted tests below reuse this fitted comparator.
                comparator, paths_order, actual_n_bins = _fit_scenario_samples(
                    samples_by_input[input_transform],
                    genes,
                    binning=binning,
                    feature_mode=feature_mode,
                    n_radial_bins=N_RADIAL_BINS,
                )
                if actual_n_bins != N_RADIAL_BINS:
                    log(f"  adapted requested {N_RADIAL_BINS} bins to actual {actual_n_bins} bins")
                suffix = f"bins{actual_n_bins}"
                if actual_n_bins != N_RADIAL_BINS:
                    suffix = f"{suffix}_requested{N_RADIAL_BINS}"
                meta = {
                    "panel": "ad_pig",
                    "scenario_id": (f"{input_transform}_{binning}_{feature_mode}_{suffix}"),
                    "input": input_transform,
                    "binning": binning,
                    "feature_mode": feature_mode,
                    "requested_n_radial_bins": N_RADIAL_BINS,
                    "n_radial_bins": int(actual_n_bins),
                }

                # The observed cache keeps full per-gene rows because it feeds
                # ranking, enrichment, and scatter-plot panels.
                real_subset = _subset_design(real_set, paths_order)
                observed_parts.append(
                    _observed_test_rows(
                        comparator=comparator,
                        subset=real_subset,
                        genes=genes,
                        meta=meta,
                    )
                )

                # Permuted sample labels are cached only as p-values for null
                # ECDF curves; DE permutation is not needed for those panels.
                for perm_idx, perm_set in enumerate(perm_sets):
                    subset = _subset_design(perm_set, paths_order)
                    null_parts.append(
                        _null_ecdf_rows(
                            comparator=comparator,
                            subset=subset,
                            perm_idx=perm_idx,
                            meta=meta,
                        )
                    )

    save_csv(pd.concat(observed_parts, ignore_index=True), test_path)
    save_csv(pd.concat(null_parts, ignore_index=True), null_path)
    return {
        "ad_pig_full_tests": str(test_path.relative_to(PROJECT_ROOT)),
        "ad_pig_full_null_ecdf": str(null_path.relative_to(PROJECT_ROOT)),
    }


def _effective_rank_summary(
    comparator: Any,
    paths_order: list[Path],
    real_set: Any,
    *,
    normalize_shape: bool,
) -> dict[str, Any]:
    subset = _subset_design(real_set, paths_order)
    if subset["is_design"]:
        raise ValueError("AD/PIG effective-rank summary requires a binary real contrast")
    real_cmp = comparator.subset(subset["idx"])
    n_bins = int(real_cmp.spectra_.shape[-1])
    # Compute per-sample rank of cross-gene spectra.
    sample_rank = np.asarray(real_cmp.effective_rank(), dtype=float)
    # Compute the pooled residual covariance.
    null_state = real_cmp.estimate_null_covariance(
        subset["groups"],
        normalize_shape=normalize_shape,
    )
    sample_names = [paths_order[int(sample_idx)].stem for sample_idx in subset["idx"]]
    return {
        "n_frequency_bins": n_bins,
        "sample_effective_rank_samples": json.dumps(sample_names, separators=(",", ":")),
        "sample_effective_rank_values": json.dumps(
            [float(rank) for rank in sample_rank], separators=(",", ":")
        ),
        "residual_effective_rank": float(null_state["effective_rank"]),
    }


def run_ad_pig_radial_bin_robustness(  # noqa: C901
    *,
    force: bool = False,
    presence_threshold: float = PRESENCE_THRESHOLD,
    max_genes: int = MAX_GENES,
    bin_counts: tuple[int, ...] = RADIAL_BIN_DIAGNOSTIC_BINS,
) -> dict[str, Any]:
    """Cache raw/radial bin-count robustness metrics for AD/PIG."""
    path = RESULT_DIR / AD_PIG_RADIAL_BIN_ROBUSTNESS_FILE
    robustness_columns = {
        "panel",
        "input",
        "binning",
        "feature_mode",
        "test",
        "requested_n_radial_bins",
        "n_radial_bins",
        "n_frequency_bins",
        "sample_effective_rank_samples",
        "sample_effective_rank_values",
        "residual_effective_rank",
        "perm_fpr_0p05",
        "n_fdr_hits",
        "pvalue_spearman_vs_ref",
        "ref_fdr_recovered_fraction",
    }
    cache_ready = cache_has_columns(path, robustness_columns)
    if cache_ready:
        try:
            cached = pd.read_csv(path, usecols=list(robustness_columns))
            expected = {
                ("raw", binning, "radial", test, int(n_bins))
                for n_bins in bin_counts
                for binning in AD_PIG_FULL_BINNINGS
                for test, _ in SPATIAL_TESTS
            }
            found = set(
                cached.loc[
                    cached["panel"].eq("ad_pig"),
                    [
                        "input",
                        "binning",
                        "feature_mode",
                        "test",
                        "requested_n_radial_bins",
                    ],
                ]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            cache_ready = found == expected
        except (OSError, ValueError, pd.errors.EmptyDataError):
            cache_ready = False
    if cache_ready and not force:
        log("using cached AD/PIG radial-bin robustness; pass --force to recompute")
        return {"ad_pig_radial_bin_robustness": str(path.relative_to(PROJECT_ROOT))}

    ensure_dirs()

    # Load the AD/PIG panel data and select genes for comparison.
    samples, _meta, sample_sets = load_panel("ad_pig")
    genes = _select_comparison_genes("ad_pig", samples, presence_threshold, max_genes)
    real_set = next(s for s in sample_sets if s.null_type == "real")  # 1 real sample set
    perm_sets = [s for s in sample_sets if s.null_type == "perm"]  # 12 shuffled sample sets

    # Run the spatial test per n_radial_bins choice
    fitted: dict[tuple[int, str], tuple[Any, list[Path], int]] = {}
    observed: dict[tuple[int, str, str], pd.DataFrame] = {}
    for n_bins in bin_counts:
        for binning in AD_PIG_FULL_BINNINGS:
            log(f"AD/PIG radial-bin robustness n_bins={n_bins} binning={binning}")
            comparator, paths_order, actual_n_bins = _fit_scenario_samples(
                samples,
                genes,
                binning=binning,
                feature_mode="radial",
                n_radial_bins=n_bins,
            )
            if actual_n_bins != n_bins:
                log(f"  adapted requested {n_bins} bins to actual {actual_n_bins}")
            fitted[(n_bins, binning)] = (comparator, paths_order, actual_n_bins)
            real_subset = _subset_design(real_set, paths_order)
            if real_subset["is_design"]:
                raise ValueError("AD/PIG radial-bin robustness requires a binary real contrast")
            # Run DF and DF-norm tests
            for test, normalize_shape in SPATIAL_TESTS:
                observed[(n_bins, binning, test)] = _spatial_test(
                    comparator, real_subset, normalize_shape=normalize_shape
                )[["Feature", "P_value"]].copy()

    # Compare each bin-count run with the same-binning requested-30-bin
    # reference so linear and log grids are judged on their own scale.
    rows: list[dict[str, Any]] = []
    for (n_bins, binning), (comparator, paths_order, actual_n_bins) in fitted.items():
        for test, normalize_shape in SPATIAL_TESTS:
            frame = observed[(n_bins, binning, test)]
            ref = observed.get((AD_PIG_REFERENCE_N_RADIAL_BINS, binning, test), pd.DataFrame())
            merged = frame.merge(ref, on="Feature", suffixes=("", "_ref"))
            current_score = -np.log10(np.clip(merged["P_value"].to_numpy(float), 1e-300, 1.0))
            ref_score = -np.log10(np.clip(merged["P_value_ref"].to_numpy(float), 1e-300, 1.0))
            if (
                current_score.size < 2
                or ref_score.size < 2
                or np.nanstd(current_score) == 0
                or np.nanstd(ref_score) == 0
            ):
                corr = np.nan
            else:
                rho = spearmanr(current_score, ref_score, nan_policy="omit").statistic
                corr = float(rho) if np.isfinite(rho) else np.nan

            current_q = apply_bh_correction(frame["P_value"].to_numpy(float))
            ref_q = (
                apply_bh_correction(ref["P_value"].to_numpy(float))
                if not ref.empty
                else np.array([], dtype=float)
            )
            current_hits = set(frame.loc[current_q <= AD_PIG_FDR_ALPHA, "Feature"])
            ref_hits = set(ref.loc[ref_q <= AD_PIG_FDR_ALPHA, "Feature"])

            null_parts: list[np.ndarray] = []
            for perm_set in perm_sets:
                subset = _subset_design(perm_set, paths_order)
                if subset["is_design"]:
                    raise ValueError("AD/PIG permutation requires a binary comparison")
                perm = _spatial_test(comparator, subset, normalize_shape=normalize_shape)
                null_parts.append(perm["P_value"].to_numpy(float))
            null_p = np.concatenate(null_parts) if null_parts else np.array([], dtype=float)
            rank_summary = _effective_rank_summary(
                comparator,
                paths_order,
                real_set,
                normalize_shape=normalize_shape,
            )
            recovered = len(current_hits & ref_hits)
            rows.append(
                {
                    "panel": "ad_pig",
                    "input": "raw",
                    "binning": binning,
                    "feature_mode": "radial",
                    "test": test,
                    "requested_n_radial_bins": int(n_bins),
                    "n_radial_bins": int(actual_n_bins),
                    **rank_summary,
                    "n_perm_sets": len(perm_sets),
                    "n_null_pvalues": int(null_p.size),
                    "perm_fpr_0p05": (float((null_p < 0.05).mean()) if null_p.size else np.nan),
                    "perm_median_p": (float(np.nanmedian(null_p)) if null_p.size else np.nan),
                    "n_fdr_hits": int(len(current_hits)),
                    "ref_n_fdr_hits": int(len(ref_hits)),
                    "pvalue_spearman_vs_ref": corr,
                    "ref_fdr_recovered_n": int(recovered),
                    "ref_fdr_recovered_fraction": (
                        float(recovered / len(ref_hits)) if ref_hits else np.nan
                    ),
                    "n_genes": int(frame.shape[0]),
                }
            )

    save_csv(pd.DataFrame(rows), path)
    return {"ad_pig_radial_bin_robustness": str(path.relative_to(PROJECT_ROOT))}


def run_ad_pig_spectral_replicate_distance_cache(
    *,
    force: bool = False,
    presence_threshold: float = PRESENCE_THRESHOLD,
    max_genes: int = MAX_GENES,
) -> dict[str, Any]:
    """Relative DC/radial-bin distances for within- vs cross-animal sections."""
    path = RESULT_DIR / AD_PIG_SPECTRAL_REPLICATE_DISTANCE_FILE
    if ad_pig_spectral_replicate_distance_cache_ready(path) and not force:
        log("using cached AD/PIG spectral replicate distances; pass --force to recompute")
        return {"ad_pig_spectral_replicate_distance": str(path.relative_to(PROJECT_ROOT))}

    ensure_dirs()

    # Load the AD/PIG panel data and select genes for comparison.
    samples, _meta, sample_sets = load_panel("ad_pig")
    real_set = next(s for s in sample_sets if s.null_type == "real")
    paths_order = sorted([*real_set.a_paths, *real_set.b_paths])
    real_samples = {path_i: samples[path_i] for path_i in paths_order}
    genes = _select_comparison_genes("ad_pig", real_samples, presence_threshold, max_genes)
    pig_gene_set = {gene for gene in AD_PIG_POS_CONTROLS if gene in set(genes)}
    condition_by_path = {
        **dict.fromkeys(real_set.a_paths, "WT"),
        **dict.fromkeys(real_set.b_paths, "AppNL-G-F"),
    }

    log(f"AD/PIG spectral replicate distances samples={len(paths_order)} genes={len(genes)}")
    # Compute binned radial spectra across genes and samples.
    comparator, paths_order = fit_all_samples(
        real_samples,
        genes,
        AD_PIG_REFERENCE_FEATURE_MODE,
        n_radial_bins=AD_PIG_REFERENCE_N_RADIAL_BINS,
    )
    if comparator.dc_ is None or comparator.spectra_ is None:
        raise RuntimeError("Comparator did not produce dc_ and spectra_ caches")

    # For each spectral bin + the DC component, compute the per-(gene, comparison)
    # relative distance between each pair of samples.
    dc = np.asarray(comparator.dc_, dtype=float)[..., None]
    spectra = np.asarray(comparator.spectra_, dtype=float)
    feature_values = np.concatenate([dc, spectra], axis=2)
    n_genes, n_bins = feature_values.shape[1], feature_values.shape[2]
    bin_index = np.arange(n_bins, dtype=int)
    bin_labels = np.array(["DC", *[f"k{i:02d}" for i in range(1, n_bins)]], dtype=object)
    bin_types = np.array(["DC", *["radial"] * (n_bins - 1)], dtype=object)
    gene_arr = np.asarray(genes, dtype=object)
    gene_set_arr = np.where(np.isin(gene_arr, list(pig_gene_set)), "PIG", "non-PIG").astype(object)
    flat_gene = np.repeat(gene_arr, n_bins)
    flat_gene_set = np.repeat(gene_set_arr, n_bins)
    flat_bin = np.tile(bin_labels, n_genes)
    flat_bin_index = np.tile(bin_index, n_genes)
    flat_bin_type = np.tile(bin_types, n_genes)

    rows: list[pd.DataFrame] = []
    eps = 1e-12
    for i, path_a in enumerate(paths_order):
        sample_a = path_a.stem
        condition = condition_by_path[path_a]
        animal_a = sample_a.split("_", maxsplit=1)[0]
        for j, path_b in enumerate(paths_order[i + 1 :], start=i + 1):
            if condition_by_path[path_b] != condition:
                # Skip the pairwise comparison if from different conditions.
                continue
            sample_b = path_b.stem
            # Samples are named as <animal>_<sample_id>, e.g., "N05_C2"
            animal_b = sample_b.split("_", maxsplit=1)[0]
            relation = "within_animal" if animal_a == animal_b else "cross_animal"
            # Symmetric relative difference, computed independently for the DC
            # component and each non-DC radial bin.
            dist = np.abs(feature_values[i] - feature_values[j]) / (
                (feature_values[i] + feature_values[j]) / 2.0 + eps
            )
            rows.append(
                pd.DataFrame(
                    {
                        "panel": "ad_pig",
                        "condition": condition,
                        "relation": relation,
                        "comparison": f"{sample_a}__{sample_b}",
                        "sample_a": sample_a,
                        "sample_b": sample_b,
                        "animal_a": animal_a,
                        "animal_b": animal_b,
                        "gene": flat_gene,
                        "gene_set": flat_gene_set,
                        "bin": flat_bin,
                        "bin_index": flat_bin_index,
                        "bin_type": flat_bin_type,
                        "relative_distance": dist.reshape(-1),
                    }
                )
            )

    if not rows:
        raise RuntimeError("No same-condition AD/PIG sample comparisons were available")
    save_csv(pd.concat(rows, ignore_index=True), path)
    return {"ad_pig_spectral_replicate_distance": str(path.relative_to(PROJECT_ROOT))}


def _raw_2d_anisotropy_scores(
    *,
    samples: dict[Path, Any],
    paths_order: list[Path],
    genes: list[str],
    panel: str,
    group_by_path: dict[Path, str],
    gene_set_by_gene: dict[str, str],
) -> pd.DataFrame:
    """Compute raw per-sample/gene 2D spectrum anisotropy before radial collapse."""
    from sonic import ComparatorIrregular
    from sonic.comparators.features import power_spectrum_anisotropy

    adatas = [samples[path_i][:, genes].copy() for path_i in paths_order]
    comparator = ComparatorIrregular(
        samples=adatas,
        gene_names=genes,
        feature_mode="radial",
        n_radial_bins=N_RADIAL_BINS,
        obsm_key="spatial",
    )
    comparator._resolve_freq_edges()

    rows: list[pd.DataFrame] = []
    for sample_idx, path_i in enumerate(paths_order):
        # Use SONIC's raw spectrum chunker directly: these are 2D spectra
        # before any rotational alignment or radial bin collapse.
        spectrum_chunker, _dc, _presence, n_genes, grid_shape = comparator._nufft_spectrum_chunker(
            sample_idx
        )
        power = spectrum_chunker(0, n_genes)
        scores = power_spectrum_anisotropy(
            power,
            grid_shape,
            comparator._spacings[sample_idx],
            fft_solver=comparator._spectrum_fft_solver,
        )
        scores.insert(0, "gene", genes)
        scores["gene_set"] = [gene_set_by_gene[gene] for gene in genes]
        scores.insert(0, "group", group_by_path[path_i])
        scores.insert(0, "sample", path_i.stem)
        scores.insert(0, "panel", panel)
        rows.append(scores)
    return pd.concat(rows, ignore_index=True)


def run_ad_pig_anisotropy_cache(
    *,
    force: bool = False,
    presence_threshold: float = PRESENCE_THRESHOLD,
    max_genes: int = MAX_GENES,
) -> dict[str, Any]:
    """Raw 2D power-spectrum anisotropy for AD/PIG genes and background genes."""
    path = RESULT_DIR / AD_PIG_ANISOTROPY_FILE
    if ad_pig_anisotropy_cache_ready(path) and not force:
        log("using cached AD/PIG anisotropy; pass --force to recompute")
        return {"ad_pig_anisotropy": str(path.relative_to(PROJECT_ROOT))}

    ensure_dirs()

    # Load the AD/PIG panel data and select genes for comparison.
    samples, _meta, sample_sets = load_panel("ad_pig")
    real_set = next(s for s in sample_sets if s.null_type == "real")
    paths_order = list(real_set.a_paths + real_set.b_paths)
    groups = {
        **dict.fromkeys(real_set.a_paths, "WT"),
        **dict.fromkeys(real_set.b_paths, "AppNL-G-F"),
    }
    real_samples = {path_i: samples[path_i] for path_i in paths_order}
    present = set.intersection(*(set(samples[path_i].var_names) for path_i in paths_order))
    # Keep PIG genes explicit and use the expression-ranked non-PIG panel only
    # as the background distribution for the anisotropy supplement.
    pig_genes = [gene for gene in AD_PIG_POS_CONTROLS if gene in present]
    if not pig_genes:
        raise RuntimeError("No AD/PIG positive-control genes are present in real samples")
    background_genes = [
        gene
        for gene in select_gene_panel(real_samples, presence_threshold, max_genes)
        if gene in present and gene not in set(pig_genes)
    ]
    genes = list(dict.fromkeys(pig_genes + background_genes))
    pig_gene_set = set(pig_genes)

    log(
        "AD/PIG anisotropy "
        f"samples={len(paths_order)} genes={len(genes)} "
        f"(PIG={len(pig_genes)}, background={len(background_genes)})"
    )
    gene_set_by_gene = {gene: ("PIG" if gene in pig_gene_set else "non-PIG") for gene in genes}
    long = _raw_2d_anisotropy_scores(
        samples=samples,
        paths_order=paths_order,
        genes=genes,
        panel="ad_pig",
        group_by_path=groups,
        gene_set_by_gene=gene_set_by_gene,
    )
    long["is_pig"] = long["gene_set"].eq("PIG")
    save_csv(long, path)
    return {"ad_pig_anisotropy": str(path.relative_to(PROJECT_ROOT))}


def run_brain_dev_anisotropy_cache(
    *,
    force: bool = False,
    presence_threshold: float = PRESENCE_THRESHOLD,
    max_genes: int = MAX_GENES,
) -> dict[str, Any]:
    """Raw 2D power-spectrum anisotropy for brain-dev marker and background genes."""
    path = RESULT_DIR / BRAIN_DEV_ANISOTROPY_FILE
    if brain_dev_anisotropy_cache_ready(path) and not force:
        log("using cached brain-development anisotropy; pass --force to recompute")
        return {"brain_dev_anisotropy": str(path.relative_to(PROJECT_ROOT))}

    ensure_dirs()

    # Load the brain development panel data and select genes for comparison.
    samples, meta, sample_sets = load_panel("brain_dev")
    real_set = next(s for s in sample_sets if s.null_type == "real")
    paths_order = list(real_set.a_paths)
    real_samples = {path_i: samples[path_i] for path_i in paths_order}
    present = set.intersection(*(set(samples[path_i].var_names) for path_i in paths_order))

    # Separate marker genes and background genes for anisotropy distribution comparison.
    marker_genes = [gene for gene in brain_dev_forced_controls() if gene in present]
    if not marker_genes:
        raise RuntimeError("No brain-dev marker genes are present in real samples")
    background_genes = [
        gene
        for gene in select_gene_panel(real_samples, presence_threshold, max_genes)
        if gene in present and gene not in set(marker_genes)
    ]
    genes = list(dict.fromkeys(marker_genes + background_genes))
    marker_set = set(marker_genes)

    meta_by_path = meta.assign(_path=meta["h5ad_path"].map(Path)).set_index("_path")
    groups = {path_i: str(meta_by_path.loc[path_i, "timepoint"]) for path_i in paths_order}
    gene_set_by_gene = {gene: ("marker" if gene in marker_set else "non-marker") for gene in genes}
    log(
        "brain-development anisotropy "
        f"samples={len(paths_order)} genes={len(genes)} "
        f"(markers={len(marker_genes)}, background={len(background_genes)})"
    )
    long = _raw_2d_anisotropy_scores(
        samples=samples,
        paths_order=paths_order,
        genes=genes,
        panel="brain_dev",
        group_by_path=groups,
        gene_set_by_gene=gene_set_by_gene,
    )
    long["is_marker"] = long["gene_set"].eq("marker")
    save_csv(long, path)
    return {"brain_dev_anisotropy": str(path.relative_to(PROJECT_ROOT))}


def _reference_tests_wide(
    full_tests: pd.DataFrame,
    *,
    panel: str,
    input_transform: str,
    binning: str,
    feature_mode: str,
    requested_n_radial_bins: int,
) -> pd.DataFrame:
    """Select the reference scenario from the full tests and pivot to wide format for scatter panels."""
    ref = full_tests[
        full_tests["panel"].eq(panel)
        & full_tests["input"].eq(input_transform)
        & full_tests["binning"].eq(binning)
        & full_tests["feature_mode"].eq(feature_mode)
        & full_tests["requested_n_radial_bins"].eq(requested_n_radial_bins)
    ].copy()
    if ref.empty:
        raise RuntimeError(f"Reference {panel} scenario is missing from tests cache")
    return (
        ref.pivot_table(
            index="Feature",
            columns="test",
            values="P_value",
            aggfunc="first",
        )
        .rename(columns={test: f"P_{test}" for test in TEST_LABELS})
        .reset_index()
    )


def _ad_pig_reference_tests_wide(full_tests: pd.DataFrame) -> pd.DataFrame:
    return _reference_tests_wide(
        full_tests,
        panel="ad_pig",
        input_transform=AD_PIG_REFERENCE_INPUT,
        binning=AD_PIG_REFERENCE_BINNING,
        feature_mode=AD_PIG_REFERENCE_FEATURE_MODE,
        requested_n_radial_bins=AD_PIG_REFERENCE_N_RADIAL_BINS,
    )


def _brain_dev_reference_tests_wide(full_tests: pd.DataFrame) -> pd.DataFrame:
    return _reference_tests_wide(
        full_tests,
        panel="brain_dev",
        input_transform=BRAIN_DEV_REFERENCE_INPUT,
        binning=BRAIN_DEV_REFERENCE_BINNING,
        feature_mode=BRAIN_DEV_REFERENCE_FEATURE_MODE,
        requested_n_radial_bins=BRAIN_DEV_REFERENCE_N_RADIAL_BINS,
    )


def _safe_term_key(row: pd.Series) -> str:
    source = str(row.get("source", ""))
    native = str(row.get("native", ""))
    name = str(row.get("name", ""))
    if native and native != "nan":
        return f"{source}:{native}"
    return f"{source}:{name}"


def _read_top_enrichment(panel: str, test: str, top_n: int) -> pd.DataFrame:
    path = RESULT_DIR / "enrichment_cache" / f"{panel}_{test}_topN{top_n}_enrich.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df.copy()
    df.insert(0, "panel", panel)
    df.insert(1, "test", test)
    df.insert(2, "variant", f"top{top_n}")
    # Use source/native ID when possible so GO terms with similar names still
    # merge correctly between DF and DF_norm.
    df["term_key"] = df.apply(_safe_term_key, axis=1)
    df["score"] = -np.log10(np.maximum(df["p_adjust"].to_numpy(float), 1e-300))
    return df


def _df_norm_enrichment_contrasts(panel: str, top_n: int) -> pd.DataFrame:  # noqa: C901
    parts: list[pd.DataFrame] = []
    base_cols = [
        "panel",
        "variant",
        "term_key",
        "source",
        "native",
        "name",
        "display_name",
        "description",
        "term_size",
        "effective_domain_size",
    ]
    test_cols = [
        "p_value",
        "p_raw",
        "p_adjust",
        "q_value",
        "score",
        "significant",
        "query_size",
        "intersection_size",
        "precision",
        "recall",
        "hit_genes",
        "hit_entrez",
        "geneID",
        "semantic_simplified",
        "semantic_simplify_cutoff",
        "semantic_simplify_measure",
        "method",
    ]
    for test in ("DF", "DF_norm"):
        df = _read_top_enrichment(panel, test, top_n)
        if df.empty:
            continue
        keep = [col for col in base_cols + test_cols if col in df.columns]
        out = df[keep].copy()
        out = out.rename(columns={col: f"{col}_{test}" for col in test_cols if col in out.columns})
        parts.append(out)
    if len(parts) != 2:
        return pd.DataFrame()

    # Missing terms on one side are retained with p_adjust=1/score=0 so the
    # contrast table can rank both shared and side-specific enrichments.
    merged = parts[0].merge(
        parts[1],
        on=["panel", "variant", "term_key"],
        how="outer",
        suffixes=("_DF", "_DF_norm"),
    )
    for col in (
        "source",
        "native",
        "name",
        "display_name",
        "description",
        "term_size",
        "effective_domain_size",
    ):
        left = f"{col}_DF"
        right = f"{col}_DF_norm"
        if left in merged.columns and right in merged.columns:
            merged[col] = merged[left].combine_first(merged[right])
        elif left in merged.columns:
            merged[col] = merged[left]
        elif right in merged.columns:
            merged[col] = merged[right]

    for test in ("DF", "DF_norm"):
        for col, default in (
            ("p_value", 1.0),
            ("p_raw", 1.0),
            ("p_adjust", 1.0),
            ("q_value", 1.0),
            ("score", 0.0),
            ("query_size", 0.0),
            ("intersection_size", 0.0),
            ("precision", 0.0),
            ("recall", 0.0),
        ):
            name = f"{col}_{test}"
            if name not in merged.columns:
                merged[name] = default
            merged[name] = merged[name].fillna(default).astype(float)
        for col in ("hit_genes", "hit_entrez", "geneID", "method"):
            name = f"{col}_{test}"
            if name not in merged.columns:
                merged[name] = ""
            merged[name] = merged[name].fillna("").astype(str)
        name = f"significant_{test}"
        if name not in merged.columns:
            merged[name] = False
        merged[name] = merged[name].where(merged[name].notna(), False).astype(bool)

    merged["delta_norm_minus_DF"] = merged["score_DF_norm"] - merged["score_DF"]
    merged["abs_delta"] = merged["delta_norm_minus_DF"].abs()
    merged["direction"] = np.where(merged["delta_norm_minus_DF"] >= 0, "DF-norm", "DF")
    merged["stronger_score"] = merged[["score_DF", "score_DF_norm"]].max(axis=1)
    merged["stronger_p_adjust"] = np.where(
        merged["direction"].eq("DF-norm"),
        merged["p_adjust_DF_norm"],
        merged["p_adjust_DF"],
    )
    merged["stronger_hit_genes"] = np.where(
        merged["direction"].eq("DF-norm"),
        merged["hit_genes_DF_norm"],
        merged["hit_genes_DF"],
    )
    merged = merged.sort_values(
        ["panel", "variant", "abs_delta", "stronger_score"],
        ascending=[True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    merged["rank_abs_delta"] = merged.groupby(["panel", "variant"], sort=False).cumcount() + 1
    return merged


def _run_topn_enrichment_cache(
    *,
    panel: str,
    reference: pd.DataFrame,
    top_n: int,
    contrast_file: str,
    force: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Cache top-N DF/DF-norm clusterProfiler results for one panel."""
    cache_dir = RESULT_DIR / "enrichment_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    top_by_test = {
        test: reference.sort_values(f"P_{test}", kind="stable")["Feature"]
        .head(top_n)
        .astype(str)
        .tolist()
        for test in ("DF", "DF_norm")
        if f"P_{test}" in reference.columns
    }
    genes_path = cache_dir / f"{panel}_topN{top_n}_genes.csv"
    if genes_path.exists():
        try:
            cached_genes = pd.read_csv(genes_path)
        except (OSError, pd.errors.EmptyDataError):
            cached_genes = pd.DataFrame()
    else:
        cached_genes = pd.DataFrame()
    genes_changed = force or any(
        test not in cached_genes.columns
        or cached_genes[test].dropna().astype(str).tolist() != genes
        for test, genes in top_by_test.items()
    )
    if genes_changed:
        save_csv(pd.DataFrame(top_by_test), genes_path)

    # Only rerun clusterProfiler when the top-gene table changed, the output is
    # stale, or the caller explicitly requested a refresh.
    job_rows: list[dict[str, Any]] = []
    for test, genes in top_by_test.items():
        enrich_path = cache_dir / f"{panel}_{test}_topN{top_n}_enrich.csv"
        enrich_stale = (
            enrich_path.exists()
            and genes_path.exists()
            and enrich_path.stat().st_mtime < genes_path.stat().st_mtime
        )
        if enrich_path.exists() and not genes_changed and not enrich_stale and not force:
            continue
        variant = f"topN{top_n}"
        job_rows.extend(
            {
                "job_id": f"{panel}:{test}:{variant}",
                "panel": panel,
                "test": test,
                "variant": variant,
                "rank": rank,
                "gene": gene,
                "orgdb": "org.Mm.eg.db",
                "organism": "mmu",
                "out_path": str(enrich_path),
            }
            for rank, gene in enumerate(dict.fromkeys(genes), start=1)
        )

    # Run clusterProfiler enrichment analysis in R using the wrapper function
    provenance = None
    if job_rows:
        log(f"clusterProfiler enrichment jobs={len({row['job_id'] for row in job_rows})}")
        with tempfile.TemporaryDirectory(prefix="clusterprofiler_jobs_") as tmp:
            jobs_path = Path(tmp) / "jobs.csv"
            pd.DataFrame(job_rows).to_csv(jobs_path, index=False)
            provenance = run_clusterprofiler(
                jobs_path,
                cwd=PROJECT_ROOT,
                timeout_seconds=int(os.environ.get("CLUSTERPROFILER_TIMEOUT_SECONDS", "900")),
            )

    contrast_path = RESULT_DIR / contrast_file
    contrasts = _df_norm_enrichment_contrasts(panel, top_n)
    save_csv(contrasts, contrast_path)
    return (
        {
            "enrichment_cache": str(cache_dir.relative_to(PROJECT_ROOT)),
            f"{panel}_enrichment_contrasts": str(contrast_path.relative_to(PROJECT_ROOT)),
        },
        provenance,
    )


def run_ad_pig_enrichment_cache(
    *,
    force: bool = False,
    top_n: int = AD_PIG_ENRICHMENT_TOP_N,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Cache reference-scenario top genes, enrichment tables, and DF-vs-DF-norm contrasts."""
    ensure_dirs()

    # Reuse the reference test results from the full_scenarios run
    full_path = RESULT_DIR / AD_PIG_FULL_TESTS_FILE
    if not full_path.exists():
        raise FileNotFoundError(f"Missing {full_path}; run run_ad_pig_full_scenarios() first.")
    full_tests = pd.read_csv(full_path)
    reference = _ad_pig_reference_tests_wide(full_tests)
    return _run_topn_enrichment_cache(
        panel="ad_pig",
        reference=reference,
        top_n=top_n,
        contrast_file=AD_ENRICH_CONTRAST_FILE,
        force=force,
    )


def run_brain_dev_enrichment_cache(
    *,
    force: bool = False,
    top_n: int = BRAIN_DEV_ENRICHMENT_TOP_N,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Cache brain-dev top genes, enrichment tables, and DF-vs-DF-norm contrasts."""
    ensure_dirs()

    # Reuse the reference test results from the full_scenarios run
    test_path = RESULT_DIR / BRAIN_DEV_TESTS_FILE
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}; run _run_brain_dev_full_scenarios() first.")
    reference = _brain_dev_reference_tests_wide(pd.read_csv(test_path))
    return _run_topn_enrichment_cache(
        panel="brain_dev",
        reference=reference,
        top_n=top_n,
        contrast_file=BRAIN_DEV_ENRICH_CONTRAST_FILE,
        force=force,
    )


def _brain_dev_samples_by_input(samples: dict[Path, Any]) -> dict[str, dict[Path, Any]]:
    """Build raw/log1p sample dictionaries once for brain-dev scenario runs."""
    out: dict[str, dict[Path, Any]] = {}
    for input_transform in BRAIN_DEV_FULL_INPUTS:
        if input_transform == "raw":
            out[input_transform] = samples
        elif input_transform == "log1p":
            out[input_transform] = log1p_normalized_samples(samples)
        else:
            raise ValueError(f"Unknown input transform: {input_transform!r}")
    return out


def _run_brain_dev_full_scenarios(
    *,
    samples: dict[Path, Any],
    sample_sets: list[Any],
    genes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute observed tests and permutation-null p-values for brain-dev scenarios."""
    real_set = next(s for s in sample_sets if s.null_type == "real")
    perm_sets = [s for s in sample_sets if s.null_type == "perm"]

    # Precompute sample adata with raw counts and log1p inputs
    samples_by_input = _brain_dev_samples_by_input(samples)

    # For each scenario, compute the observed test results and null ECDFs on shuffled data.
    observed_parts: list[pd.DataFrame] = []
    null_parts: list[pd.DataFrame] = []
    for input_transform in BRAIN_DEV_FULL_INPUTS:
        for feature_mode in BRAIN_DEV_FULL_FEATURE_MODES:
            log(
                "brain-development scenario "
                f"{input_transform}_{BRAIN_DEV_REFERENCE_BINNING}_{feature_mode} "
                f"requested_bins={BRAIN_DEV_REFERENCE_N_RADIAL_BINS}"
            )
            comparator, paths_order, actual_n_bins = _fit_scenario_samples(
                samples_by_input[input_transform],
                genes,
                binning=BRAIN_DEV_REFERENCE_BINNING,
                feature_mode=feature_mode,
                n_radial_bins=BRAIN_DEV_REFERENCE_N_RADIAL_BINS,
            )
            # Real and permuted tests below reuse this one fitted comparator;
            # the null only shuffles the continuous time design values.
            if actual_n_bins != BRAIN_DEV_REFERENCE_N_RADIAL_BINS:
                log(
                    f"  adapted requested {BRAIN_DEV_REFERENCE_N_RADIAL_BINS} "
                    f"bins to actual {actual_n_bins} bins"
                )
            suffix = f"bins{actual_n_bins}"
            if actual_n_bins != BRAIN_DEV_REFERENCE_N_RADIAL_BINS:
                suffix = f"{suffix}_requested{BRAIN_DEV_REFERENCE_N_RADIAL_BINS}"
            meta = {
                "panel": "brain_dev",
                "scenario_id": (
                    f"{input_transform}_{BRAIN_DEV_REFERENCE_BINNING}_{feature_mode}_{suffix}"
                ),
                "input": input_transform,
                "binning": BRAIN_DEV_REFERENCE_BINNING,
                "feature_mode": feature_mode,
                "requested_n_radial_bins": BRAIN_DEV_REFERENCE_N_RADIAL_BINS,
                "n_radial_bins": int(actual_n_bins),
            }

            # Run Comparator test on the real design
            real_subset = _subset_design(real_set, paths_order)
            if not real_subset["is_design"]:
                raise RuntimeError(
                    f"Missing brain-development real design for {meta['scenario_id']}"
                )
            observed_parts.append(
                _observed_test_rows(
                    comparator=comparator,
                    subset=real_subset,
                    genes=genes,
                    meta=meta,
                )
            )

            # Run Comparator tests on the permuted designs and build null ECDF rows.
            for perm_idx, perm_set in enumerate(perm_sets):
                subset = _subset_design(perm_set, paths_order)
                if not subset["is_design"]:
                    raise ValueError("Brain-development permutation requires a continuous design")
                null_parts.append(
                    _null_ecdf_rows(
                        comparator=comparator,
                        subset=subset,
                        perm_idx=perm_idx,
                        meta=meta,
                    )
                )

    return (
        pd.concat(observed_parts, ignore_index=True),
        pd.concat(null_parts, ignore_index=True),
    )


def run_ad_pig_workflow(
    *,
    force: bool = False,
    presence_threshold: float = PRESENCE_THRESHOLD,
    max_genes: int = MAX_GENES,
) -> dict[str, Any]:
    """Compute/cache AD/PIG benchmark inputs."""
    ensure_dirs()
    outputs: dict[str, Any] = {}
    # The order matters: enrichment and figure selections depend on the
    # reference rows produced by the full scenario grid.
    log("AD/PIG step 1/3: full scenario grid and permuted-null ECDFs")
    outputs.update(
        run_ad_pig_full_scenarios(
            force=force,
            presence_threshold=presence_threshold,
            max_genes=max_genes,
        )
    )

    log("AD/PIG step 2/3: raw radial bin-count robustness")
    outputs.update(
        run_ad_pig_radial_bin_robustness(
            force=force,
            presence_threshold=presence_threshold,
            max_genes=max_genes,
        )
    )

    log("AD/PIG step 3/3: reference enrichment, anisotropy, and replicate spectra")
    enrichment_outputs, provenance = run_ad_pig_enrichment_cache(force=force)
    outputs.update(enrichment_outputs)
    outputs.update(
        run_ad_pig_anisotropy_cache(
            force=force,
            presence_threshold=presence_threshold,
            max_genes=max_genes,
        )
    )
    outputs.update(
        run_ad_pig_spectral_replicate_distance_cache(
            force=force,
            presence_threshold=presence_threshold,
            max_genes=max_genes,
        )
    )
    result: dict[str, Any] = {"outputs": outputs}
    if provenance is not None:
        result["provenance"] = {"clusterprofiler": provenance}
    return result


def run_brain_dev_workflow(
    *,
    force: bool = False,
    presence_threshold: float = PRESENCE_THRESHOLD,
    max_genes: int = MAX_GENES,
) -> dict[str, Any]:
    """Cache the brain-development continuous-time benchmark inputs."""
    ensure_dirs()
    outputs: dict[str, Any] = {}
    controls = brain_dev_positive_control_table(force=force)
    outputs["brain_dev_positive_controls"] = str(
        (RESULT_DIR / BRAIN_DEV_POSITIVE_CONTROLS_FILE).relative_to(PROJECT_ROOT)
    )
    log(
        "Brain-dev step 1/3: build positive-control gene sets "
        f"groups={controls['control_group'].nunique()} genes={controls['gene'].nunique()}"
    )

    log("Brain-dev step 2/3: full scenario grid and permuted-null ECDFs")
    test_path = RESULT_DIR / BRAIN_DEV_TESTS_FILE
    null_path = RESULT_DIR / BRAIN_DEV_NULL_ECDF_FILE
    if _brain_dev_cache_ready(test_path, null_path) and not force:
        log("using cached brain-development tests; pass --force to recompute")
    else:
        log(
            "brain-development continuous-design scenario grid "
            f"permutations={BRAIN_DEV_NULL_PERMUTATIONS}"
        )
        samples, _meta, sample_sets = load_panel("brain_dev")
        genes = _select_comparison_genes("brain_dev", samples, presence_threshold, max_genes)
        observed, null = _run_brain_dev_full_scenarios(
            samples=samples,
            sample_sets=sample_sets,
            genes=genes,
        )
        save_csv(observed, test_path)
        save_csv(null, null_path)

    outputs["brain_dev_tests"] = str(test_path.relative_to(PROJECT_ROOT))
    outputs["brain_dev_null_ecdf"] = str(null_path.relative_to(PROJECT_ROOT))

    log("Brain-dev step 3/3: anisotropy and reference enrichment")
    outputs.update(
        run_brain_dev_anisotropy_cache(
            force=force,
            presence_threshold=presence_threshold,
            max_genes=max_genes,
        )
    )
    enrichment_outputs, provenance = run_brain_dev_enrichment_cache(force=force)
    outputs.update(enrichment_outputs)
    result: dict[str, Any] = {"outputs": outputs}
    if provenance is not None:
        result["provenance"] = {"clusterprofiler": provenance}
    return result


def run_workflow(
    *,
    analysis: str = "all",
    force: bool = False,
    presence_threshold: float = PRESENCE_THRESHOLD,
    max_genes: int = MAX_GENES,
) -> dict[str, Any]:
    """Run one of the panel-specific comparator-benchmark workflows."""
    started_at = time.perf_counter()
    if analysis == "ad-pig":
        result = _run_guarded_panel(
            "ad_pig",
            run_ad_pig_workflow,
            force=force,
            presence_threshold=presence_threshold,
            max_genes=max_genes,
        )
    elif analysis == "brain-dev":
        result = _run_guarded_panel(
            "brain_dev",
            run_brain_dev_workflow,
            force=force,
            presence_threshold=presence_threshold,
            max_genes=max_genes,
        )
    elif analysis == "all":
        ad = _run_guarded_panel(
            "ad_pig",
            run_ad_pig_workflow,
            force=force,
            presence_threshold=presence_threshold,
            max_genes=max_genes,
        )
        brain = _run_guarded_panel(
            "brain_dev",
            run_brain_dev_workflow,
            force=force,
            presence_threshold=presence_threshold,
            max_genes=max_genes,
        )
        result = {"outputs": {**ad["outputs"], **brain["outputs"]}}
    else:
        raise ValueError(f"Unknown analysis mode: {analysis!r}")

    result["outputs"]["cache_manifest"] = project_relative(RESULT_DIR / "cache_manifest.json")
    manifest_path = write_workflow_manifest(
        "comparator_benchmark",
        RESULT_DIR,
        started_at=started_at,
        parameters={
            "analysis": analysis,
            "cache_parameters": CACHE_PARAMETERS,
            "cache_revisions": CACHE_REVISIONS,
            "force": force,
            "presence_threshold": presence_threshold,
            "max_genes": max_genes,
        },
        outputs=result["outputs"],
        summary={"n_outputs": len(result["outputs"])},
    )
    result["outputs"]["workflow_manifest"] = project_relative(manifest_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and cache comparator benchmark inputs.")
    parser.add_argument("--force", action="store_true", help="Recompute cached outputs.")
    parser.add_argument(
        "--analysis",
        choices=("ad-pig", "brain-dev", "all"),
        default="all",
        help="Panel workflow to run. Default is all analyses.",
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=int(os.environ.get("MAX_GENES", MAX_GENES)),
    )
    parser.add_argument("--presence-threshold", type=float, default=PRESENCE_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_workflow(
        analysis=args.analysis,
        force=args.force,
        presence_threshold=args.presence_threshold,
        max_genes=args.max_genes,
    )


if __name__ == "__main__":
    main()
