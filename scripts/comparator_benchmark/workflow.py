"""Run and cache the comparator-benchmark analysis inputs.

The AD/PIG workflow is organized around three components:

1. The full eight-scenario comparison grid used for normalization/binning/
   feature-mode sensitivity.
2. Raw radial bin-count robustness diagnostics.
3. The reference AD/PIG analysis used for enrichment, gene-profile selection,
   anisotropy summaries, and technical-replicate spectral diagnostics.

Figures are generated separately from the cached CSVs by ``visualization.ipynb``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.comparator_benchmark"

from quadsv.statistics import apply_bh_correction

from .config import (
    AD_ENRICH_CONTRAST_FILE,
    AD_PIG_ANISOTROPY_FILE,
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
    AD_PIG_SPECTRAL_REPLICATE_DISTANCE_FILE,
    AD_PIG_STALE_PROBE_NAMES,
    ANISOTROPY_METHOD,
    CLUSTERPROFILER_RSCRIPT,
    MAX_GENES,
    N_RADIAL_BINS,
    PRESENCE_THRESHOLD,
    PROJECT_ROOT,
    RADIAL_BIN_DIAGNOSTIC_BINS,
    RESULT_DIR,
    SPATIAL_TESTS,
    TEST_LABELS,
)
from .utils import (
    de_test_binary,
    de_test_continuous,
    ensure_dirs,
    fit_all_samples,
    geometric_frequency_edges,
    load_panel,
    log,
    log1p_normalized_samples,
    save_csv,
    select_gene_panel,
)


def _ad_pig_cache_has_current_qc(path: Path, gene_col: str = "Feature") -> bool:
    """Return whether a cache no longer contains pre-QC AD/PIG probe names."""
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, usecols=lambda col: col in {"panel", gene_col})
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False
    if "panel" not in df.columns or gene_col not in df.columns:
        return False
    ad_genes = set(df.loc[df["panel"] == "ad_pig", gene_col].dropna().astype(str))
    return bool(ad_genes) and not (ad_genes & AD_PIG_STALE_PROBE_NAMES)


def _cache_has_columns(path: Path, required: set[str]) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, nrows=1)
    except (OSError, pd.errors.EmptyDataError):
        return False
    return required.issubset(df.columns)


def _ad_pig_full_cache_ready(test_path: Path, null_path: Path) -> bool:
    if not _cache_has_columns(
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
    if not _cache_has_columns(
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
    if not _ad_pig_cache_has_current_qc(test_path):
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
    observed_tests = set(
        tests.loc[tests["panel"].eq("ad_pig"), "test"].dropna().astype(str)
    )
    null_tests = set(null.loc[null["panel"].eq("ad_pig"), "test"].dropna().astype(str))
    return observed_tests == set(TEST_LABELS) and null_tests == {
        test for test, _ in SPATIAL_TESTS
    }


def _ad_pig_anisotropy_cache_ready(path: Path) -> bool:
    if not _ad_pig_cache_has_current_qc(path, gene_col="gene"):
        return False
    if not _cache_has_columns(
        path,
        {
            "panel",
            "sample",
            "group",
            "gene",
            "gene_set",
            "anisotropy",
            "dominant_angle_deg",
            "anisotropy_method",
        },
    ):
        return False
    try:
        long = pd.read_csv(path, usecols=["gene_set", "anisotropy_method"])
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False
    return {"PIG", "non-PIG"}.issubset(set(long["gene_set"].astype(str))) and set(
        long["anisotropy_method"].dropna().astype(str)
    ) == {ANISOTROPY_METHOD}


def _ad_pig_spectral_replicate_distance_cache_ready(path: Path) -> bool:
    if not _ad_pig_cache_has_current_qc(path, gene_col="gene"):
        return False
    required = {
        "panel",
        "condition",
        "relation",
        "comparison",
        "sample_a",
        "sample_b",
        "animal_a",
        "animal_b",
        "gene",
        "gene_set",
        "bin",
        "bin_index",
        "bin_type",
        "relative_distance",
    }
    if not _cache_has_columns(path, required):
        return False
    try:
        cols = ["panel", "condition", "relation", "bin_type", "relative_distance"]
        long = pd.read_csv(path, usecols=cols)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False
    if long.empty:
        return False
    return (
        set(long["panel"].astype(str)) == {"ad_pig"}
        and {"WT", "AppNL-G-F"}.issubset(set(long["condition"].astype(str)))
        and {"within_animal", "cross_animal"}.issubset(
            set(long["relation"].astype(str))
        )
        and {"DC", "radial"}.issubset(set(long["bin_type"].astype(str)))
        and np.isfinite(long["relative_distance"].to_numpy(float)).all()
    )


def _select_comparison_genes(
    panel: str,
    samples: dict[Path, Any],
    presence_threshold: float = PRESENCE_THRESHOLD,
    max_genes: int = MAX_GENES,
) -> list[str]:
    """Select benchmark genes, forcing present AD/PIG controls into AD/PIG panels."""
    genes = select_gene_panel(samples, presence_threshold, max_genes)
    if panel != "ad_pig":
        return genes

    present = set.intersection(*(set(adata.var_names) for adata in samples.values()))
    controls = [gene for gene in AD_PIG_POS_CONTROLS if gene in present]
    return list(dict.fromkeys(genes + controls))


def _subset_design(sample_set: Any, paths_order: list[Path]) -> dict[str, Any] | None:
    # Convert stored sample-set metadata into the compact design object expected
    # by the comparator API. Binary and continuous contrasts share this shape.
    path_to_idx = {path: i for i, path in enumerate(paths_order)}
    if sample_set.design_values is not None:
        idx_list = [path_to_idx[p] for p in sample_set.a_paths if p in path_to_idx]
        if not idx_list:
            return None
        keep = [i for i, p in enumerate(sample_set.a_paths) if p in path_to_idx]
        return {
            "is_design": True,
            "idx": np.asarray(idx_list, dtype=int),
            "design_values": sample_set.design_values[keep],
            "design_name": sample_set.design_name,
        }

    a_idx = [path_to_idx[p] for p in sample_set.a_paths if p in path_to_idx]
    b_idx = [path_to_idx[p] for p in sample_set.b_paths if p in path_to_idx]
    if not a_idx or not b_idx:
        return None
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
        keep = [
            col
            for col in ("Feature", "Statistic", "P_value", "P_adj")
            if col in out.columns
        ]
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
    if binning == "linear":
        comparator, paths_order = fit_all_samples(
            samples, genes, feature_mode, n_radial_bins=n_radial_bins
        )
        return comparator, paths_order, int(n_radial_bins)
    if binning != "log":
        raise ValueError(f"Unknown binning: {binning!r}")

    # Build log edges from the radial grid first, then reuse those frequency
    # edges for either radial or 2D feature extraction in the scenario grid.
    linear_cmp, _paths = fit_all_samples(
        samples,
        genes,
        "radial",
        n_radial_bins=n_radial_bins,
    )
    log_edges = geometric_frequency_edges(np.asarray(linear_cmp.freq_edges), linear_cmp)
    comparator, paths_order = fit_all_samples(
        samples,
        genes,
        feature_mode,
        n_radial_bins=n_radial_bins,
        freq_edges=log_edges,
    )
    return comparator, paths_order, int(log_edges.size - 1)


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
                # Compute per-sample spectra for the current scenario.
                comparator, paths_order, actual_n_bins = _fit_scenario_samples(
                    samples_by_input[input_transform],
                    genes,
                    binning=binning,
                    feature_mode=feature_mode,
                    n_radial_bins=N_RADIAL_BINS,
                )
                if actual_n_bins != N_RADIAL_BINS:
                    # Log-binning may lead to empty bins; collapse the frequency edges.
                    log(
                        f"  collapsed requested {N_RADIAL_BINS} bins "
                        f"to actual {actual_n_bins} bins"
                    )
                suffix = f"bins{actual_n_bins}"
                if actual_n_bins != N_RADIAL_BINS:
                    suffix = f"{suffix}_requested{N_RADIAL_BINS}"
                meta = {
                    "panel": "ad_pig",
                    "scenario_id": (
                        f"{input_transform}_{binning}_{feature_mode}_{suffix}"
                    ),
                    "input": input_transform,
                    "binning": binning,
                    "feature_mode": feature_mode,
                    "requested_n_radial_bins": N_RADIAL_BINS,
                    "n_radial_bins": int(actual_n_bins),
                }

                # The observed cache keeps full per-gene rows because it feeds
                # ranking, enrichment, and scatter-plot panels.
                real_subset = _subset_design(real_set, paths_order)
                if real_subset is None:
                    raise RuntimeError(f"Missing real subset for {meta['scenario_id']}")
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
                    if subset is None:
                        continue
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
    if subset is None or subset["is_design"]:
        raise ValueError(
            "AD/PIG effective-rank summary requires a binary real contrast"
        )
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
        "sample_effective_rank_samples": json.dumps(
            sample_names, separators=(",", ":")
        ),
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
    cache_ready = _cache_has_columns(path, robustness_columns)
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
    real_set = next(
        s for s in sample_sets if s.null_type == "real"
    )  # 1 real sample set
    perm_sets = [
        s for s in sample_sets if s.null_type == "perm"
    ]  # 12 shuffled sample sets

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
            # Log-binning may lead to empty bins; collapse the frequency edges.
            if actual_n_bins != n_bins:
                log(f"  collapsed requested {n_bins} bins to actual {actual_n_bins}")
            fitted[(n_bins, binning)] = (comparator, paths_order, actual_n_bins)
            real_subset = _subset_design(real_set, paths_order)
            if real_subset is None or real_subset["is_design"]:
                raise ValueError(
                    "AD/PIG radial-bin robustness requires a binary real contrast"
                )
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
            ref = observed.get(
                (AD_PIG_REFERENCE_N_RADIAL_BINS, binning, test), pd.DataFrame()
            )
            merged = frame.merge(ref, on="Feature", suffixes=("", "_ref"))
            current_score = -np.log10(
                np.clip(merged["P_value"].to_numpy(float), 1e-300, 1.0)
            )
            ref_score = -np.log10(
                np.clip(merged["P_value_ref"].to_numpy(float), 1e-300, 1.0)
            )
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
            current_hits = set(frame.loc[current_q < AD_PIG_FDR_ALPHA, "Feature"])
            ref_hits = set(ref.loc[ref_q < AD_PIG_FDR_ALPHA, "Feature"])

            null_parts: list[np.ndarray] = []
            for perm_set in perm_sets:
                subset = _subset_design(perm_set, paths_order)
                if subset is None or subset["is_design"]:
                    continue
                perm = _spatial_test(
                    comparator, subset, normalize_shape=normalize_shape
                )
                null_parts.append(perm["P_value"].to_numpy(float))
            null_p = (
                np.concatenate(null_parts) if null_parts else np.array([], dtype=float)
            )
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
                    "perm_fpr_0p05": (
                        float((null_p < 0.05).mean()) if null_p.size else np.nan
                    ),
                    "perm_median_p": (
                        float(np.nanmedian(null_p)) if null_p.size else np.nan
                    ),
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


def run_ad_pig_anisotropy_cache(
    *,
    force: bool = False,
    presence_threshold: float = PRESENCE_THRESHOLD,
    max_genes: int = MAX_GENES,
) -> dict[str, Any]:
    """Raw 2D power-spectrum anisotropy for AD/PIG genes and background genes."""
    path = RESULT_DIR / AD_PIG_ANISOTROPY_FILE
    if _ad_pig_anisotropy_cache_ready(path) and not force:
        log("using cached AD/PIG anisotropy; pass --force to recompute")
        return {"ad_pig_anisotropy": str(path.relative_to(PROJECT_ROOT))}

    from quadsv import ComparatorIrregular
    from quadsv.comparators.features import power_spectrum_anisotropy

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
    present = set.intersection(
        *(set(samples[path_i].var_names) for path_i in paths_order)
    )
    # Keep PIG genes explicit and use the expression-ranked non-PIG panel only
    # as the background distribution for the anisotropy supplement.
    pig_genes = [gene for gene in AD_PIG_POS_CONTROLS if gene in present]
    if not pig_genes:
        raise RuntimeError(
            "No AD/PIG positive-control genes are present in real samples"
        )
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
        # Anisotropy is computed from raw 2D spectra before radial collapse, so
        # this intentionally uses the comparator's spectrum chunker directly.
        spectrum_chunker, _dc, _presence, n_genes, grid_shape = (
            comparator._nufft_spectrum_chunker(sample_idx)
        )
        power = spectrum_chunker(0, n_genes)
        scores = power_spectrum_anisotropy(
            power,
            grid_shape,
            comparator._spacings[sample_idx],
            fft_solver=comparator._spectrum_fft_solver,
        )
        scores.insert(0, "gene", genes)
        scores["anisotropy_method"] = ANISOTROPY_METHOD
        scores["is_pig"] = scores["gene"].isin(pig_gene_set)
        scores["gene_set"] = np.where(scores["is_pig"], "PIG", "non-PIG")
        scores.insert(0, "group", groups[path_i])
        scores.insert(0, "sample", path_i.stem)
        scores.insert(0, "panel", "ad_pig")
        rows.append(scores)

    save_csv(pd.concat(rows, ignore_index=True), path)
    return {"ad_pig_anisotropy": str(path.relative_to(PROJECT_ROOT))}


def run_ad_pig_spectral_replicate_distance_cache(
    *,
    force: bool = False,
    presence_threshold: float = PRESENCE_THRESHOLD,
    max_genes: int = MAX_GENES,
) -> dict[str, Any]:
    """Relative DC/radial-bin distances for within- vs cross-animal sections."""
    path = RESULT_DIR / AD_PIG_SPECTRAL_REPLICATE_DISTANCE_FILE
    if _ad_pig_spectral_replicate_distance_cache_ready(path) and not force:
        log(
            "using cached AD/PIG spectral replicate distances; pass --force to recompute"
        )
        return {
            "ad_pig_spectral_replicate_distance": str(path.relative_to(PROJECT_ROOT))
        }

    ensure_dirs()

    # Load the AD/PIG panel data and select genes for comparison.
    samples, _meta, sample_sets = load_panel("ad_pig")
    real_set = next(s for s in sample_sets if s.null_type == "real")
    paths_order = sorted([*real_set.a_paths, *real_set.b_paths])
    real_samples = {path_i: samples[path_i] for path_i in paths_order}
    genes = _select_comparison_genes(
        "ad_pig", real_samples, presence_threshold, max_genes
    )
    pig_gene_set = {gene for gene in AD_PIG_POS_CONTROLS if gene in set(genes)}
    condition_by_path = {
        **dict.fromkeys(real_set.a_paths, "WT"),
        **dict.fromkeys(real_set.b_paths, "AppNL-G-F"),
    }

    log(
        "AD/PIG spectral replicate distances "
        f"samples={len(paths_order)} genes={len(genes)}"
    )
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
    bin_labels = np.array(
        ["DC", *[f"k{i:02d}" for i in range(1, n_bins)]], dtype=object
    )
    bin_types = np.array(["DC", *["radial"] * (n_bins - 1)], dtype=object)
    gene_arr = np.asarray(genes, dtype=object)
    gene_set_arr = np.where(
        np.isin(gene_arr, list(pig_gene_set)), "PIG", "non-PIG"
    ).astype(object)
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


def _reference_tests_wide(full_tests: pd.DataFrame) -> pd.DataFrame:
    ref = full_tests[
        full_tests["panel"].eq("ad_pig")
        & full_tests["input"].eq(AD_PIG_REFERENCE_INPUT)
        & full_tests["binning"].eq(AD_PIG_REFERENCE_BINNING)
        & full_tests["feature_mode"].eq(AD_PIG_REFERENCE_FEATURE_MODE)
        & full_tests["n_radial_bins"].eq(AD_PIG_REFERENCE_N_RADIAL_BINS)
    ].copy()
    if ref.empty:
        raise RuntimeError("Reference AD/PIG scenario is missing from full tests cache")
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


def _clusterprofiler_command() -> list[str]:
    env_rscript = os.environ.get("CLUSTERPROFILER_RSCRIPT")
    if env_rscript:
        return [env_rscript]
    conda = shutil.which("conda")
    if conda:
        return [
            conda,
            "run",
            "-n",
            os.environ.get("CLUSTERPROFILER_CONDA_ENV", "quadsv_paper"),
            "Rscript",
        ]
    path_rscript = shutil.which("Rscript")
    if path_rscript:
        return [path_rscript]
    raise FileNotFoundError(
        "Could not find conda or Rscript for clusterProfiler enrichment."
    )


def run_clusterprofiler_jobs(job_rows: list[dict[str, Any]]) -> None:
    """Run clusterProfiler GO BP + KEGG enrichment for pending gene-set jobs."""
    if not job_rows:
        return
    if not CLUSTERPROFILER_RSCRIPT.exists():
        raise FileNotFoundError(f"Missing {CLUSTERPROFILER_RSCRIPT}")

    with tempfile.TemporaryDirectory(prefix="clusterprofiler_jobs_") as tmp:
        job_path = Path(tmp) / "jobs.csv"
        pd.DataFrame(job_rows).to_csv(job_path, index=False)
        cmd = _clusterprofiler_command() + [str(CLUSTERPROFILER_RSCRIPT), str(job_path)]
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    if proc.stdout.strip():
        log(proc.stdout.strip())
    if proc.stderr.strip():
        log(proc.stderr.strip())


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
        out = out.rename(
            columns={col: f"{col}_{test}" for col in test_cols if col in out.columns}
        )
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
    merged["rank_abs_delta"] = (
        merged.groupby(["panel", "variant"], sort=False).cumcount() + 1
    )
    return merged


def run_ad_pig_enrichment_cache(
    *,
    force: bool = False,
    top_n: int = AD_PIG_ENRICHMENT_TOP_N,
) -> dict[str, Any]:
    """Cache reference-scenario top genes, enrichment tables, and DF-vs-DF-norm contrasts."""
    ensure_dirs()

    # Reuse the reference test results from the full_scenarios run
    full_path = RESULT_DIR / AD_PIG_FULL_TESTS_FILE
    if not full_path.exists():
        raise FileNotFoundError(
            f"Missing {full_path}; run run_ad_pig_full_scenarios() first."
        )
    full_tests = pd.read_csv(full_path)
    reference = _reference_tests_wide(full_tests)
    top_by_test = {
        test: reference.sort_values(f"P_{test}", kind="stable")["Feature"]
        .head(top_n)
        .astype(str)
        .tolist()
        for test in ("DF", "DF_norm")
        if f"P_{test}" in reference.columns
    }

    # Reuse cached gene lists if available
    cache_dir = RESULT_DIR / "enrichment_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    genes_path = cache_dir / f"ad_pig_topN{top_n}_genes.csv"
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
        enrich_path = cache_dir / f"ad_pig_{test}_topN{top_n}_enrich.csv"
        enrich_stale = (
            enrich_path.exists()
            and genes_path.exists()
            and enrich_path.stat().st_mtime < genes_path.stat().st_mtime
        )
        if (
            enrich_path.exists()
            and not genes_changed
            and not enrich_stale
            and not force
        ):
            continue
        variant = f"topN{top_n}"
        job_rows.extend(
            {
                "job_id": f"ad_pig:{test}:{variant}",
                "panel": "ad_pig",
                "test": test,
                "variant": variant,
                "rank": rank,
                "gene": gene,
                "out_path": str(enrich_path),
            }
            for rank, gene in enumerate(dict.fromkeys(genes), start=1)
        )

    # Run clusterProfiler enrichment analysis in R using the wrapper function
    if job_rows:
        log(
            f"clusterProfiler enrichment jobs={len({row['job_id'] for row in job_rows})}"
        )
        run_clusterprofiler_jobs(job_rows)

    contrast_path = RESULT_DIR / AD_ENRICH_CONTRAST_FILE
    contrasts = _df_norm_enrichment_contrasts("ad_pig", top_n)
    save_csv(contrasts, contrast_path)
    return {
        "enrichment_cache": str(cache_dir.relative_to(PROJECT_ROOT)),
        "ad_pig_enrichment_contrasts": str(contrast_path.relative_to(PROJECT_ROOT)),
    }


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
    outputs.update(run_ad_pig_enrichment_cache(force=force))
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
    return {"outputs": outputs}


def run_brain_dev_workflow(
    *,
    force: bool = False,
    presence_threshold: float = PRESENCE_THRESHOLD,
    max_genes: int = MAX_GENES,
) -> dict[str, Any]:
    """Placeholder for the brain-development workflow after the AD/PIG split."""
    _ = (force, presence_threshold, max_genes)
    log("brain-development workflow is not part of the AD/PIG cache reorganization")
    return {"outputs": {}}


def run_workflow(
    *,
    analysis: str = "ad-pig",
    force: bool = False,
    presence_threshold: float = PRESENCE_THRESHOLD,
    max_genes: int = MAX_GENES,
) -> dict[str, Any]:
    """Run one of the panel-specific comparator-benchmark workflows."""
    if analysis == "ad-pig":
        return run_ad_pig_workflow(
            force=force,
            presence_threshold=presence_threshold,
            max_genes=max_genes,
        )
    if analysis == "brain-dev":
        return run_brain_dev_workflow(
            force=force,
            presence_threshold=presence_threshold,
            max_genes=max_genes,
        )
    if analysis == "all":
        ad = run_ad_pig_workflow(
            force=force,
            presence_threshold=presence_threshold,
            max_genes=max_genes,
        )
        brain = run_brain_dev_workflow(
            force=force,
            presence_threshold=presence_threshold,
            max_genes=max_genes,
        )
        return {"outputs": {**ad["outputs"], **brain["outputs"]}}
    raise ValueError(f"Unknown analysis mode: {analysis!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and cache comparator benchmark inputs."
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute cached outputs."
    )
    parser.add_argument(
        "--analysis",
        choices=("ad-pig", "brain-dev", "all"),
        default="ad-pig",
        help="Panel workflow to run. Default is the AD/PIG analysis.",
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
