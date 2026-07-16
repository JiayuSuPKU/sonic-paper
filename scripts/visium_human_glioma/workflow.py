"""Run and cache the Visium human glioma IDH-status analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.visium_human_glioma"

from .config import (
    CACHE_SUMMARY_FILE,
    ENRICHMENT_TERMS_FILE,
    FDR_ALPHA,
    GROUP_LABELS,
    META_PROGRAM_GENE_MAP_FILE,
    META_PROGRAM_PATHWAY_KEYWORDS,
    META_PROGRAMS,
    MIN_FRAC_PER_SAMPLE,
    MIN_SAMPLES_FRAC,
    MIN_SAMPLES_PER_GROUP,
    N_GENES,
    N_LABEL_PERMUTATIONS,
    N_RADIAL_BINS,
    PERMUTATION_NULL_FILE,
    PROJECT_ROOT,
    REAL_TESTS_FILE,
    SAMPLE_MAX_QC_EXCLUDED_FRAC,
    SAMPLES,
    SPATIAL_DIAGNOSTICS_FILE,
    SPECTRA_FILE,
    SPOT_MAX_PCT_MITO,
    SPOT_MIN_COUNTS,
)
from .utils import (
    align_samples_to_genes,
    ensure_dirs,
    load_normalized_samples,
    log,
    merge_real_tests,
    project_relative,
    resolve_data_root,
    sample_metadata,
    save_csv,
    select_gene_panel,
)

CLUSTERPROFILER_SCRIPT = PROJECT_ROOT / "scripts" / "run_clusterprofiler.R"
CLUSTERPROFILER_TIMEOUT_SECONDS = 900
CACHE_SCHEMA_VERSION = 1

COMPARISON_CACHE_FILES = {
    "real_tests": REAL_TESTS_FILE,
    "permutation_null": PERMUTATION_NULL_FILE,
    "spectra": SPECTRA_FILE,
    "spatial_diagnostics": SPATIAL_DIAGNOSTICS_FILE,
}


def _file_fingerprint(path: Path, *, digest: bool = False) -> dict[str, Any]:
    """Return stable metadata for one cache input or output."""
    resolved = path.resolve()
    if not path.exists():
        return {"path": str(resolved), "exists": False}
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if digest:
        sha256 = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                sha256.update(chunk)
        result["sha256"] = sha256.hexdigest()
    return result


def _load_cache_summary() -> dict[str, Any]:
    """Read the cache manifest, treating a missing or malformed file as stale."""
    try:
        summary = json.loads(CACHE_SUMMARY_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {"schema_version": CACHE_SCHEMA_VERSION}
    if summary.get("schema_version") != CACHE_SCHEMA_VERSION:
        return {"schema_version": CACHE_SCHEMA_VERSION}
    return summary


def _write_cache_summary(
    section: str,
    *,
    signature: dict[str, Any],
    outputs: dict[str, Any],
    invalidate: tuple[str, ...] = (),
) -> None:
    """Atomically update one cache stage and invalidate its dependents."""
    summary = _load_cache_summary()
    for name in invalidate:
        summary.pop(name, None)
    summary[section] = {"signature": signature, "outputs": outputs}
    CACHE_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_SUMMARY_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    tmp.replace(CACHE_SUMMARY_FILE)
    log(f"wrote {project_relative(CACHE_SUMMARY_FILE)}")


def _output_fingerprints(
    paths: dict[str, Path], *, digest: bool = False
) -> dict[str, dict[str, Any]]:
    return {
        name: _file_fingerprint(path, digest=digest) for name, path in paths.items()
    }


def _comparison_signature(data_root: Path) -> dict[str, Any]:
    """Describe every input that changes the IDH-status comparison."""
    import quadsv

    meta = sample_metadata(data_root)
    inputs = []
    for row in meta.itertuples(index=False):
        inputs.append(
            {
                "sample_id": str(row.sample_id),
                "group": str(row.group),
                "h5ad": _file_fingerprint(Path(row.h5ad_path)),
            }
        )
    return {
        "data_root": str(data_root.resolve()),
        "inputs": inputs,
        "quadsv_version": quadsv.__version__,
        "parameters": {
            "n_genes": N_GENES,
            "spot_min_counts": SPOT_MIN_COUNTS,
            "spot_max_pct_mito": SPOT_MAX_PCT_MITO,
            "sample_max_qc_excluded_frac": SAMPLE_MAX_QC_EXCLUDED_FRAC,
            "min_frac_per_sample": MIN_FRAC_PER_SAMPLE,
            "min_samples_frac": MIN_SAMPLES_FRAC,
            "min_samples_per_group": MIN_SAMPLES_PER_GROUP,
            "n_radial_bins": N_RADIAL_BINS,
            "n_label_permutations": N_LABEL_PERMUTATIONS,
        },
    }


def _enrichment_signature() -> dict[str, Any]:
    """Describe the tested genes and wrapper used for pathway enrichment."""
    return {
        "real_tests": _file_fingerprint(REAL_TESTS_FILE, digest=True),
        "clusterprofiler_wrapper": _file_fingerprint(
            CLUSTERPROFILER_SCRIPT, digest=True
        ),
        "top_n": 100,
        "sources": ["GO:BP", "KEGG"],
        "orgdb": "org.Hs.eg.db",
        "organism": "hsa",
    }


def _fit_radial_comparator(
    aligned_samples: dict[str, Any],
    sample_ids: list[str],
    genes: list[str],
    *,
    n_jobs: int,
) -> Any:
    from quadsv import ComparatorIrregular

    panel = [aligned_samples[sid] for sid in sample_ids]
    cmp = ComparatorIrregular(
        samples=panel,
        gene_names=genes,
        feature_mode="radial",
        n_radial_bins=N_RADIAL_BINS,
        adaptive_freq_edges=True,
        obsm_key="spatial",
        presence_threshold=MIN_FRAC_PER_SAMPLE,
    )
    cmp.compute_spectra(n_jobs=n_jobs, progress=False)
    return cmp


def _idh_status_comparison_cache_ready(data_root: Path) -> bool:
    """Return whether the complete IDH-status comparison cache is valid."""
    if not all(
        path.exists() and path.stat().st_size > 0
        for path in (
            REAL_TESTS_FILE,
            SPECTRA_FILE,
            PERMUTATION_NULL_FILE,
            SPATIAL_DIAGNOSTICS_FILE,
        )
    ):
        return False
    section = _load_cache_summary().get("comparison", {})
    if section.get("signature") != _comparison_signature(data_root):
        return False
    if section.get("outputs") != _output_fingerprints(COMPARISON_CACHE_FILES):
        return False
    if not _spatial_diagnostics_cache_ready():
        return False
    required = {
        "Feature",
        "P_DE",
        "P_DF",
        "P_DF_norm",
        "P_adj_DE",
        "P_adj_DF",
        "P_adj_DF_norm",
    }
    null_required = {"perm_idx", "test", "P_value", "P_adj"}
    try:
        real_tests = pd.read_csv(REAL_TESTS_FILE)
        null = pd.read_csv(PERMUTATION_NULL_FILE)
        if not required.issubset(real_tests.columns) or not null_required.issubset(
            null.columns
        ):
            return False
        with np.load(SPECTRA_FILE, allow_pickle=True) as spectra:
            if not {"spectra", "freq_edges", "sample_ids", "groups", "genes"}.issubset(
                spectra.files
            ):
                return False
            spectra_shape = spectra["spectra"].shape
            n_edges = spectra["freq_edges"].size
            cached_samples = spectra["sample_ids"].astype(str).tolist()
            cached_groups = spectra["groups"].astype(str).tolist()
            cached_genes = spectra["genes"].astype(str).tolist()
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False
    expected_samples = [sample_id for sample_id, _group in SAMPLES]
    expected_groups = [group for _sample_id, group in SAMPLES]
    n_genes = real_tests.shape[0]
    if (
        real_tests["Feature"].nunique() != n_genes
        or set(real_tests["Feature"].astype(str)) != set(cached_genes)
        or cached_samples != expected_samples
        or cached_groups != expected_groups
        or spectra_shape != (len(SAMPLES), n_genes, N_RADIAL_BINS)
        or n_edges != N_RADIAL_BINS + 1
    ):
        return False
    expected_null_rows = 2 * N_LABEL_PERMUTATIONS * n_genes
    counts = null.groupby(["test", "perm_idx"], sort=False).size()
    return (
        len(null) == expected_null_rows
        and set(counts.index.get_level_values("test")) == {"DF", "DF_norm"}
        and counts.size == 2 * N_LABEL_PERMUTATIONS
        and counts.eq(n_genes).all()
    )


def _permuted_group_labels(
    groups: np.ndarray,
    *,
    n_permutations: int = N_LABEL_PERMUTATIONS,
) -> list[np.ndarray]:
    """Return deterministic unique label shuffles that preserve group sizes."""
    labels = np.asarray(groups, dtype=str)
    rng = np.random.default_rng(42)
    seen = {tuple(labels.tolist())}
    shuffled: list[np.ndarray] = []
    while len(shuffled) < n_permutations:
        candidate = rng.permutation(labels)
        key = tuple(candidate.tolist())
        if key in seen:
            continue
        seen.add(key)
        shuffled.append(candidate)
    return shuffled


def _permutation_null_table(comparator: Any, groups: np.ndarray) -> pd.DataFrame:
    """Test DF and DF-norm after sample-label shuffling for null calibration."""
    from quadsv.statistics import apply_bh_correction

    rows: list[pd.DataFrame] = []
    for perm_idx, shuffled in enumerate(_permuted_group_labels(groups)):
        for test, normalize_shape in (("DF", False), ("DF_norm", True)):
            result = comparator.test_diff_freq(
                shuffled,
                statistic="log_l2",
                null="analytic",
                normalize_shape=normalize_shape,
                min_samples_per_group=MIN_SAMPLES_PER_GROUP,
            )
            p_value = result["P_value"].to_numpy(float)
            rows.append(
                pd.DataFrame(
                    {
                        "perm_idx": perm_idx,
                        "test": test,
                        "P_value": p_value,
                        "P_adj": apply_bh_correction(p_value),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _sample_gene_means(
    aligned_samples: dict[str, Any],
    sample_ids: list[str],
) -> np.ndarray:
    """Mean log-normalized expression per sample/gene for DE testing."""
    rows = []
    for sid in sample_ids:
        x = aligned_samples[sid].X
        mean = x.mean(axis=0) if hasattr(x, "toarray") else np.asarray(x).mean(axis=0)
        rows.append(np.asarray(mean, dtype=float).ravel())
    return np.asarray(rows, dtype=float)


def _effective_rank_from_observations(values: np.ndarray) -> float:
    """Effective rank of the gene covariance from observation x gene values."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or min(x.shape) < 2:
        return float("nan")
    x = x - x.mean(axis=0, keepdims=True)
    eig = np.linalg.svd(x, compute_uv=False, full_matrices=False) ** 2
    total = float(eig.sum())
    if total <= 0:
        return float("nan")
    return float(total * total / float((eig**2).sum()))


def _spatial_diagnostics_table(
    comparator: Any,
    spectra: np.ndarray,
    aligned_samples: dict[str, Any],
    sample_ids: list[str],
    groups: np.ndarray,
    genes: list[str],
) -> pd.DataFrame:
    """Per-sample anisotropy and expression/spectrum covariance effective ranks."""
    from quadsv.comparators.features import power_spectrum_anisotropy

    rows: list[pd.DataFrame] = []
    for sample_idx, sample_id in enumerate(sample_ids):
        group = str(groups[sample_idx])
        group_label = GROUP_LABELS[group]

        # Effective rank of gene-by-gene covariance from log-normalized spot expression.
        x_expr = aligned_samples[sample_id].X
        if hasattr(x_expr, "toarray"):
            x_expr = x_expr.toarray()
        x_expr = np.asarray(x_expr, dtype=float)
        expression_rank = _effective_rank_from_observations(x_expr)
        rows.append(
            pd.DataFrame(
                [
                    {
                        "metric": "effective_rank",
                        "basis": "expression_spots",
                        "sample_id": sample_id,
                        "group": group,
                        "group_label": group_label,
                        "gene": "",
                        "value": expression_rank,
                        "effective_rank": expression_rank,
                        "anisotropy": np.nan,
                        "dominant_angle_deg": np.nan,
                        "total_power": np.nan,
                        "n_observations": int(x_expr.shape[0]),
                        "n_genes": int(x_expr.shape[1]),
                    }
                ]
            )
        )

        # Effective rank of gene-by-gene covariance from log radial power profiles.
        x_power = np.log(np.maximum(spectra[sample_idx], 1e-12)).T
        power_rank = _effective_rank_from_observations(x_power)
        rows.append(
            pd.DataFrame(
                [
                    {
                        "metric": "effective_rank",
                        "basis": "radial_power_bins",
                        "sample_id": sample_id,
                        "group": group,
                        "group_label": group_label,
                        "gene": "",
                        "value": power_rank,
                        "effective_rank": power_rank,
                        "anisotropy": np.nan,
                        "dominant_angle_deg": np.nan,
                        "total_power": np.nan,
                        "n_observations": int(x_power.shape[0]),
                        "n_genes": int(x_power.shape[1]),
                    }
                ]
            )
        )

        # Radial bins discard angle, so re-stream raw 2D power in bounded chunks.
        spectrum_chunker, _dc, _presence, n_genes, grid_shape = (
            comparator._nufft_spectrum_chunker(sample_idx)
        )
        chunk_size = int(comparator._nufft_chunk_size)
        anis = pd.concat(
            [
                power_spectrum_anisotropy(
                    spectrum_chunker(start, min(start + chunk_size, n_genes)),
                    grid_shape,
                    comparator._spacings[sample_idx],
                    fft_solver=comparator._spectrum_fft_solver,
                )
                for start in range(0, n_genes, chunk_size)
            ],
            ignore_index=True,
        )
        anis.insert(0, "gene", genes)
        anis.insert(0, "group_label", group_label)
        anis.insert(0, "group", group)
        anis.insert(0, "sample_id", sample_id)
        anis.insert(0, "basis", "raw_2d_power")
        anis.insert(0, "metric", "anisotropy")
        anis["value"] = anis["anisotropy"]
        anis["effective_rank"] = np.nan
        anis["n_observations"] = np.nan
        anis["n_genes"] = np.nan
        rows.append(anis)

    return pd.concat(rows, ignore_index=True, sort=False)


def _spatial_diagnostics_cache_ready() -> bool:
    """Return whether the diagnostics cache has every required metric basis."""
    if (
        not SPATIAL_DIAGNOSTICS_FILE.exists()
        or SPATIAL_DIAGNOSTICS_FILE.stat().st_size == 0
    ):
        return False
    try:
        diagnostics = pd.read_csv(SPATIAL_DIAGNOSTICS_FILE, usecols=["metric", "basis"])
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    observed = set(diagnostics[["metric", "basis"]].itertuples(index=False, name=None))
    return {
        ("anisotropy", "raw_2d_power"),
        ("effective_rank", "expression_spots"),
        ("effective_rank", "radial_power_bins"),
    }.issubset(observed)


def _clusterprofiler_command() -> list[str]:
    """Resolve Rscript without baking a conda environment into the public script."""
    active_env_rscript = Path(sys.prefix) / "bin" / "Rscript"
    if active_env_rscript.exists():
        return [str(active_env_rscript)]
    path_rscript = shutil.which("Rscript")
    if path_rscript:
        return [path_rscript]
    raise FileNotFoundError(
        "Could not find Rscript in the active Python environment or PATH."
    )


def _run_clusterprofiler(job_rows: list[dict[str, Any]]) -> None:
    """Write one temporary job table and run the shared clusterProfiler wrapper."""
    if not job_rows:
        return
    if not CLUSTERPROFILER_SCRIPT.exists():
        raise FileNotFoundError(f"Missing {CLUSTERPROFILER_SCRIPT}")
    job_path = Path(job_rows[0]["job_path"])
    pd.DataFrame(job_rows).drop(columns=["job_path"]).to_csv(job_path, index=False)
    cmd = _clusterprofiler_command() + [str(CLUSTERPROFILER_SCRIPT), str(job_path)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=CLUSTERPROFILER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "clusterProfiler enrichment timed out after "
            f"{CLUSTERPROFILER_TIMEOUT_SECONDS}s for {len({row['job_id'] for row in job_rows})} jobs"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"clusterProfiler enrichment failed\nstdout:\n{exc.stdout}\n\nstderr:\n{exc.stderr}"
        ) from exc
    if proc.stdout.strip():
        log(proc.stdout.strip())
    if proc.stderr.strip():
        log(proc.stderr.strip())


def _ranked_gene_sets(
    real_tests: pd.DataFrame, *, top_n: int = 100
) -> dict[str, list[str]]:
    """Return equal-sized DE and DF-norm gene sets ordered by raw p-value."""
    ranked: dict[str, list[str]] = {}
    for gene_set, p_col in (("DE", "P_DE"), ("DF_norm", "P_DF_norm")):
        sub = (
            real_tests.dropna(subset=[p_col])
            .sort_values(p_col, kind="stable")
            .head(top_n)
        )
        ranked[gene_set] = sub["Feature"].astype(str).drop_duplicates().tolist()
    return ranked


def _missing_enrichment_pairs(terms: pd.DataFrame) -> set[tuple[str, str]]:
    """Return expected gene-set/source combinations absent from an enrichment table."""
    expected = {
        (gene_set, source)
        for gene_set in ("DE", "DF_norm")
        for source in ("GO:BP", "KEGG")
    }
    if not {"gene_set", "source"}.issubset(terms.columns):
        return expected
    observed = set(
        terms[["gene_set", "source"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    return expected - observed


def _enrichment_cache_ready() -> bool:
    """Return whether cached GO BP and KEGG results cover both ranked gene sets."""
    if not ENRICHMENT_TERMS_FILE.exists() or ENRICHMENT_TERMS_FILE.stat().st_size == 0:
        return False
    section = _load_cache_summary().get("enrichment", {})
    if section.get("signature") != _enrichment_signature():
        return False
    outputs = {"enrichment": _file_fingerprint(ENRICHMENT_TERMS_FILE, digest=True)}
    if section.get("outputs") != outputs:
        return False
    try:
        terms = pd.read_csv(ENRICHMENT_TERMS_FILE)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    return not _missing_enrichment_pairs(terms)


def _read_enrichment_outputs(
    out_paths: dict[str, Path], gene_sets: dict[str, list[str]]
) -> list[pd.DataFrame]:
    """Read non-empty clusterProfiler job outputs and restore gene-set labels."""
    parts: list[pd.DataFrame] = []
    for gene_set, out_path in out_paths.items():
        if not out_path.exists() or out_path.stat().st_size == 0:
            continue
        part = pd.read_csv(out_path)
        if part.empty:
            continue
        part.insert(0, "gene_set", gene_set)
        part.insert(1, "n_query_genes", len(gene_sets[gene_set]))
        parts.append(part)
    return parts


def run_enrichment_cache(
    *,
    force: bool = False,
) -> dict[str, str]:
    """Run GO BP + KEGG enrichment for top DE and DF-norm genes."""
    ensure_dirs()
    if _enrichment_cache_ready() and not force:
        log("using cached clusterProfiler enrichment; pass --force to recompute")
        return {"enrichment": project_relative(ENRICHMENT_TERMS_FILE)}
    if ENRICHMENT_TERMS_FILE.exists() and not force:
        log("cached enrichment is incomplete; recomputing GO BP and KEGG")
    if not REAL_TESTS_FILE.exists():
        raise FileNotFoundError(f"Missing {REAL_TESTS_FILE}; run workflow.py first.")

    top_n = 100
    real_tests = pd.read_csv(REAL_TESTS_FILE)
    gene_sets = _ranked_gene_sets(real_tests, top_n=top_n)
    # Use the tested gene panel as the enrichment universe.
    background = real_tests["Feature"].astype(str).drop_duplicates().tolist()

    with tempfile.TemporaryDirectory(prefix="glioma_clusterprofiler_") as tmpdir:
        tmp = Path(tmpdir)
        job_path = tmp / "jobs.csv"
        universe_path = tmp / "background.csv"
        pd.DataFrame({"gene": background}).to_csv(universe_path, index=False)

        # The R wrapper groups rows by job_id, so each gene is one row.
        job_rows: list[dict[str, Any]] = []
        out_paths: dict[str, Path] = {}
        for gene_set, genes in gene_sets.items():
            log(f"{gene_set} genes for enrichment: {len(genes):,}")
            if len(genes) < 5:
                continue
            out_path = tmp / f"{gene_set}_clusterprofiler.csv"
            out_paths[gene_set] = out_path
            job_rows.extend(
                {
                    "job_path": str(job_path),
                    "job_id": f"visium_glioma_idh_status:{gene_set}:top{top_n}",
                    "panel": "visium_glioma_idh_status",
                    "test": gene_set,
                    "variant": f"top{top_n}",
                    "rank": rank,
                    "gene": gene,
                    "orgdb": "org.Hs.eg.db",
                    "organism": "hsa",
                    "universe_path": str(universe_path),
                    "out_path": str(out_path),
                }
                for rank, gene in enumerate(genes, start=1)
            )

        if job_rows:
            log(
                f"clusterProfiler enrichment jobs={len({row['job_id'] for row in job_rows})}"
            )
            _run_clusterprofiler(job_rows)
        parts = _read_enrichment_outputs(out_paths, gene_sets)

    terms = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    missing = _missing_enrichment_pairs(terms)
    if missing:
        formatted = ", ".join(
            f"{gene_set}/{source}" for gene_set, source in sorted(missing)
        )
        raise RuntimeError(
            "clusterProfiler returned incomplete enrichment results: "
            f"{formatted}. Check the R output and KEGG network access."
        )
    save_csv(terms, ENRICHMENT_TERMS_FILE)
    META_PROGRAM_GENE_MAP_FILE.unlink(missing_ok=True)
    _write_cache_summary(
        "enrichment",
        signature=_enrichment_signature(),
        outputs={"enrichment": _file_fingerprint(ENRICHMENT_TERMS_FILE, digest=True)},
    )
    return {"enrichment": project_relative(ENRICHMENT_TERMS_FILE)}


def _eligible_df_norm_pathways(enrichment: pd.DataFrame) -> pd.DataFrame:
    """Filter and score DF-norm-preferred GO/KEGG pathways for gene assignment."""
    required = {
        "gene_set",
        "source",
        "native",
        "name",
        "p_adjust",
        "term_size",
        "hit_genes",
        "semantic_simplified",
    }
    missing = required - set(enrichment.columns)
    if missing:
        raise KeyError(f"Enrichment table is missing columns: {sorted(missing)}")

    keys = ["source", "native"]
    de_fdr = (
        enrichment.loc[enrichment["gene_set"].eq("DE"), keys + ["p_adjust"]]
        .assign(p_adjust=lambda x: pd.to_numeric(x["p_adjust"], errors="coerce"))
        .groupby(keys, as_index=False)["p_adjust"]
        .min()
        .rename(columns={"p_adjust": "de_fdr"})
    )
    terms = (
        enrichment.loc[enrichment["gene_set"].eq("DF_norm")]
        .sort_values("p_adjust", kind="stable")
        .drop_duplicates(keys)
        .merge(de_fdr, on=keys, how="left", validate="one_to_one")
    )
    terms["df_norm_fdr"] = pd.to_numeric(terms["p_adjust"], errors="coerce")
    terms["de_fdr"] = pd.to_numeric(terms["de_fdr"], errors="coerce").fillna(1.0)
    terms["term_size"] = pd.to_numeric(terms["term_size"], errors="coerce")
    go_simplified = terms["semantic_simplified"].astype(str).str.lower().eq("true")
    covid_text = (
        terms.reindex(columns=["name", "display_name", "description"], fill_value="")
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
    terms = terms.loc[
        terms["df_norm_fdr"].lt(FDR_ALPHA)
        & terms["df_norm_fdr"].lt(terms["de_fdr"])
        & terms["term_size"].ge(1)
        & (terms["source"].ne("GO:BP") | go_simplified)
        & ~covid_text.str.contains("covid|sars|coronavirus", case=False, regex=True)
    ].copy()

    eps = np.finfo(float).tiny
    terms["contrast_score"] = np.log10(
        np.maximum(terms["de_fdr"], eps) / np.maximum(terms["df_norm_fdr"], eps)
    )
    terms["selection_score"] = terms["contrast_score"] / np.log2(terms["term_size"] + 1)
    pathway_text = (
        terms.reindex(columns=["name", "display_name", "description"], fill_value="")
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    terms["metaprograms"] = [
        tuple(
            program
            for program, keywords in META_PROGRAM_PATHWAY_KEYWORDS.items()
            if any(keyword.lower() in text for keyword in keywords)
        )
        for text in pathway_text
    ]
    return terms


def assign_top_df_norm_genes_to_metaprograms(
    real_tests: pd.DataFrame,
    enrichment: pd.DataFrame,
    *,
    top_n: int = 100,
) -> pd.DataFrame:
    """Assign top DF-norm genes to their strongest supported meta-program."""
    if not {"Feature", "P_DF_norm"}.issubset(real_tests.columns):
        raise KeyError("Real-test table must contain Feature and P_DF_norm columns.")
    ranked = (
        real_tests.dropna(subset=["P_DF_norm"])
        .sort_values("P_DF_norm", kind="stable")
        .drop_duplicates("Feature")
        .head(top_n)
    )
    top_genes = ranked["Feature"].astype(str).tolist()
    top_gene_set = set(top_genes)
    terms = _eligible_df_norm_pathways(enrichment)

    program_order = {program["key"]: i for i, program in enumerate(META_PROGRAMS)}

    eligible_hits: set[str] = set()
    candidates: dict[str, list[dict[str, Any]]] = {gene: [] for gene in top_genes}
    for term in terms.itertuples(index=False):
        hits = top_gene_set.intersection(
            gene for gene in re.split(r"[/,;|]", str(term.hit_genes)) if gene
        )
        eligible_hits.update(hits)
        for gene in hits:
            for program in term.metaprograms:
                candidates[gene].append(
                    {
                        "evidence_metaprogram": program,
                        "evidence_source": str(term.source),
                        "evidence_id": str(term.native),
                        "evidence_term": str(term.name),
                        "term_size": int(term.term_size),
                        "df_norm_fdr": float(term.df_norm_fdr),
                        "de_fdr": float(term.de_fdr),
                        "contrast_score": float(term.contrast_score),
                        "selection_score": float(term.selection_score),
                    }
                )

    rows: list[dict[str, Any]] = []
    for rank, gene in enumerate(top_genes, start=1):
        pool = candidates[gene]
        if pool:
            matched = sorted(
                {candidate["evidence_metaprogram"] for candidate in pool},
                key=program_order.__getitem__,
            )
            ordered = sorted(
                pool,
                key=lambda x: (
                    -x["selection_score"],
                    -x["contrast_score"],
                    x["term_size"],
                    program_order[x["evidence_metaprogram"]],
                    x["evidence_source"],
                    x["evidence_id"],
                ),
            )
            best = ordered[0]
            matched_terms = "|".join(
                dict.fromkeys(
                    f"{candidate['evidence_source']}:{candidate['evidence_id']}:{candidate['evidence_term']}"
                    for candidate in ordered
                )
            )
            rows.append(
                {
                    "rank": rank,
                    "gene": gene,
                    "metaprogram": best["evidence_metaprogram"],
                    "matched_metaprograms": "|".join(matched),
                    "matched_terms": matched_terms if len(matched) > 1 else np.nan,
                    "is_mixed": len(matched) > 1,
                    **best,
                    "assignment_basis": "preferred_pathway",
                }
            )
            continue
        rows.append(
            {
                "rank": rank,
                "gene": gene,
                "metaprogram": "unassigned",
                "matched_metaprograms": np.nan,
                "matched_terms": np.nan,
                "is_mixed": False,
                "assignment_basis": (
                    "no_curated_metaprogram_term"
                    if gene in eligible_hits
                    else "no_significant_preferred_term"
                ),
                "evidence_metaprogram": np.nan,
                "evidence_source": np.nan,
                "evidence_id": np.nan,
                "evidence_term": np.nan,
                "term_size": np.nan,
                "df_norm_fdr": np.nan,
                "de_fdr": np.nan,
                "contrast_score": np.nan,
                "selection_score": np.nan,
            }
        )
    columns = [
        "rank",
        "gene",
        "metaprogram",
        "matched_metaprograms",
        "matched_terms",
        "is_mixed",
        "assignment_basis",
        "evidence_metaprogram",
        "evidence_source",
        "evidence_id",
        "evidence_term",
        "term_size",
        "df_norm_fdr",
        "de_fdr",
        "contrast_score",
        "selection_score",
    ]
    return pd.DataFrame(rows).reindex(columns=columns)


def run_metaprogram_gene_map_cache() -> dict[str, str]:
    """Regenerate the top-100 DF-norm meta-program map from current caches."""
    missing = [
        path for path in (REAL_TESTS_FILE, ENRICHMENT_TERMS_FILE) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing inputs for meta-program assignment: {missing}"
        )
    gene_map = assign_top_df_norm_genes_to_metaprograms(
        pd.read_csv(REAL_TESTS_FILE),
        pd.read_csv(ENRICHMENT_TERMS_FILE),
    )
    save_csv(gene_map, META_PROGRAM_GENE_MAP_FILE)
    assigned = int(gene_map["metaprogram"].ne("unassigned").sum())
    mixed = int(gene_map["is_mixed"].sum())
    log(
        f"meta-program assignments: {assigned} assigned ({mixed} mixed), "
        f"{len(gene_map) - assigned} unassigned"
    )
    return {"metaprogram_gene_map": project_relative(META_PROGRAM_GENE_MAP_FILE)}


def run_idh_status_comparison_cache(
    *,
    data_root: Path | str | None = None,
    force: bool = False,
    n_jobs: int = 1,
) -> dict[str, str]:
    """Cache the IDH-status DE/DF comparison, nulls, spectra, and diagnostics."""
    ensure_dirs()
    resolved_data_root = resolve_data_root(data_root)
    if _idh_status_comparison_cache_ready(resolved_data_root) and not force:
        log("using cached glioma IDH-status comparison; pass --force to recompute")
        return {
            "real_tests": project_relative(REAL_TESTS_FILE),
            "permutation_null": project_relative(PERMUTATION_NULL_FILE),
            "spectra": project_relative(SPECTRA_FILE),
            "spatial_diagnostics": project_relative(SPATIAL_DIAGNOSTICS_FILE),
        }

    ENRICHMENT_TERMS_FILE.unlink(missing_ok=True)
    META_PROGRAM_GENE_MAP_FILE.unlink(missing_ok=True)
    log(f"data_root={resolved_data_root}")
    # Load all samples once, normalize in memory, then keep only the selected genes.
    samples, meta = load_normalized_samples(resolved_data_root)
    sample_ids = meta["sample_id"].tolist()
    groups = meta["group"].astype(str).to_numpy()

    genes, gene_summary = select_gene_panel(
        samples,
        max_genes=N_GENES,
        min_frac_per_sample=MIN_FRAC_PER_SAMPLE,
        min_samples_frac=MIN_SAMPLES_FRAC,
    )
    if not genes:
        raise RuntimeError("No genes passed the configured detection filters.")
    log(f"gene panel candidates summarized in memory ({gene_summary.shape[0]:,} genes)")

    # QuadSV expects every sample to expose the same feature order.
    log("aligning samples to the selected gene panel")
    aligned_abundance = align_samples_to_genes(samples, genes)
    log(
        "fitting radial ComparatorIrregular "
        f"({len(sample_ids)} samples x {len(genes):,} genes x {N_RADIAL_BINS} bins)"
    )
    cmp = _fit_radial_comparator(aligned_abundance, sample_ids, genes, n_jobs=n_jobs)

    log("testing DE on abundance and DF/DF-norm on centered relative expression")
    from quadsv.comparators.multisample import compare_two_groups_scalar

    de = compare_two_groups_scalar(
        _sample_gene_means(aligned_abundance, sample_ids),
        groups,
        gene_names=genes,
    )
    df = cmp.test_diff_freq(
        groups,
        statistic="log_l2",
        null="analytic",
        min_samples_per_group=MIN_SAMPLES_PER_GROUP,
    )
    df_norm = cmp.test_diff_freq(
        groups,
        statistic="log_l2",
        null="analytic",
        normalize_shape=True,
        min_samples_per_group=MIN_SAMPLES_PER_GROUP,
    )
    real_tests = merge_real_tests(de, df, df_norm)
    save_csv(real_tests, REAL_TESTS_FILE)

    # Run the same DF/DF-norm tests under sample-label shuffling to show null calibration.
    log(f"testing DF/DF-norm under {N_LABEL_PERMUTATIONS} sample-label shuffles")
    save_csv(_permutation_null_table(cmp, groups), PERMUTATION_NULL_FILE)

    # Store only what plotting needs: spectra, frequency bins, sample metadata, and genes.
    np.savez_compressed(
        SPECTRA_FILE,
        spectra=np.asarray(cmp.spectra_, dtype=np.float32),
        freq_edges=np.asarray(cmp.freq_edges, dtype=np.float64),
        sample_ids=np.asarray(sample_ids, dtype=object),
        groups=np.asarray(groups, dtype=object),
        group_labels=np.asarray(
            [GROUP_LABELS[group] for group in groups],
            dtype=object,
        ),
        h5ad_paths=meta["h5ad_path"].astype(str).to_numpy(dtype=object),
        genes=np.asarray(genes, dtype=object),
    )
    log(f"wrote {project_relative(SPECTRA_FILE)}")

    # Compute per-sample anisotropy and effective ranks of expression and radial power.
    # Radial bins discard angle, so stream raw 2D power from the fitted comparator.
    log("computing per-sample anisotropy and covariance effective ranks")
    save_csv(
        _spatial_diagnostics_table(
            cmp,
            np.asarray(cmp.spectra_, dtype=float),
            aligned_abundance,
            sample_ids,
            groups,
            genes,
        ),
        SPATIAL_DIAGNOSTICS_FILE,
    )

    _write_cache_summary(
        "comparison",
        signature=_comparison_signature(resolved_data_root),
        outputs=_output_fingerprints(COMPARISON_CACHE_FILES),
        invalidate=("enrichment",),
    )

    return {
        "real_tests": project_relative(REAL_TESTS_FILE),
        "permutation_null": project_relative(PERMUTATION_NULL_FILE),
        "spectra": project_relative(SPECTRA_FILE),
        "spatial_diagnostics": project_relative(SPATIAL_DIAGNOSTICS_FILE),
    }


def run_workflow(
    *,
    data_root: Path | str | None = None,
    force: bool = False,
    n_jobs: int = 1,
    run_enrichment: bool = True,
) -> dict[str, Any]:
    """Compute/cache the analysis inputs consumed by ``visualization.ipynb``."""
    outputs = run_idh_status_comparison_cache(
        data_root=data_root,
        force=force,
        n_jobs=n_jobs,
    )
    if run_enrichment:
        outputs.update(run_enrichment_cache(force=force))
        outputs.update(run_metaprogram_gene_map_cache())
    outputs["cache_summary"] = project_relative(CACHE_SUMMARY_FILE)
    return {"outputs": outputs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run/cache the Visium human glioma IDH-status analysis."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root containing adata/*.h5ad. Defaults to data/visium_glioma_idh_status.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute cached outputs."
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip clusterProfiler enrichment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_workflow(
        data_root=args.data_root,
        force=args.force,
        n_jobs=args.n_jobs,
        run_enrichment=not args.skip_enrichment,
    )


if __name__ == "__main__":
    main()
