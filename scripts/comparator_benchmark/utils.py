"""Shared helpers for the comparator benchmark workflow and downloaders."""

from __future__ import annotations

import gzip
import logging
import math
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .config import (
    AD_PIG_DROP_PROBES,
    AD_PIG_RENAME_PROBES,
    BRAIN_DEV_NULL_PERMUTATIONS,
    FIGURE_DIR,
    MAX_GENES,
    N_RADIAL_BINS,
    PANELS,
    PRESENCE_THRESHOLD,
    PROJECT_ROOT,
    RESULT_DIR,
)

logger = logging.getLogger("comparator_benchmark.utils")


def _is_comparison_gene(gene: str) -> bool:
    """Return whether ``gene`` should be eligible for comparison sweeps."""
    is_predicted_gene = gene.startswith("Gm") and len(gene) > 2 and gene[2].isdigit()
    return not (is_predicted_gene or gene.startswith("mt-"))


# ---------------------------------------------------------------------------
# 1. Logging / filesystem
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    """Timestamped print, mirrors ``visiumhd_mouse_cbs.utils.log``."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def ensure_dirs() -> None:
    """Create the project's result + figure directories if missing."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    try:
        rel = path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = path
    log(f"wrote {rel} ({df.shape[0]:,} rows)")


def cache_has_columns(path: Path, required: set[str]) -> bool:
    """Return whether a CSV exists and has at least the required columns."""
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, nrows=1)
    except (OSError, pd.errors.EmptyDataError):
        return False
    return required.issubset(df.columns)


def format_pvalue(p: float) -> str:
    """Compact p-value string for small manuscript panels."""
    p = float(p)
    if not np.isfinite(p):
        return "NA"
    if p == 0:
        return "<1e-300"
    if p < 1e-3:
        exp = int(math.floor(math.log10(p)))
        mant = p / 10**exp
        return f"{mant:.1f}e{exp}"
    return f"{p:.3f}".rstrip("0").rstrip(".")


def parse_gene_list(values: Sequence[str] | None) -> list[str]:
    """Parse CLI-style gene values that may be comma- or whitespace-separated."""
    if not values:
        return []
    genes: list[str] = []
    for value in values:
        genes.extend(g.strip() for g in value.replace(",", " ").split())
    return [gene for gene in genes if gene]


def unique(items: Iterable[str]) -> list[str]:
    """Return unique items while preserving first-seen order."""
    return list(dict.fromkeys(items))


def l1_normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L1-normalize the last axis of an array."""
    arr = np.asarray(values, dtype=float)
    return arr / np.maximum(arr.sum(axis=-1, keepdims=True), eps)


def radial_frequency_axis(comparator: Any) -> np.ndarray:
    """Return radial-bin centers for a fitted comparator."""
    # Comparator class records retained frequency bins in intervals
    intervals = getattr(comparator, "frequency_bin_intervals_", None)
    if intervals is not None:
        intervals = np.asarray(intervals, dtype=float)
        if intervals.ndim == 2 and intervals.shape[0] == comparator.spectra_.shape[-1]:
            return 0.5 * (intervals[:, 0] + intervals[:, 1])

    # Fall back to requested edges or unit-spaced bins if intervals are missing.
    n_bins = comparator.spectra_.shape[-1]
    edges = np.asarray(getattr(comparator, "freq_edges", []), dtype=float)
    if len(edges) == n_bins + 1:
        return 0.5 * (edges[:-1] + edges[1:])
    if len(edges) == n_bins + 2:
        return 0.5 * (edges[1:-1] + edges[2:])
    return np.arange(n_bins, dtype=float)


def scale_visium_spatial_to_spot_diameter_um(
    adata: ad.AnnData,
    spot_diameter_um: float,
    *,
    obsm_key: str = "spatial",
) -> ad.AnnData:
    """Set Visium spatial coordinates in micrometers from raw fullres pixels."""
    spatial_uns = adata.uns.setdefault("spatial", {})
    scalefactors = (
        spatial_uns.setdefault("scalefactors", {})
        if isinstance(spatial_uns, dict)
        else {}
    )
    fullres_spot_diameter = float(scalefactors.get("spot_diameter_fullres", 0.0))
    if not np.isfinite(fullres_spot_diameter) or fullres_spot_diameter <= 0:
        return adata

    if {"pxl_col_in_fullres", "pxl_row_in_fullres"}.issubset(adata.obs.columns):
        coords = np.column_stack(
            [
                adata.obs["pxl_col_in_fullres"].to_numpy(float),
                adata.obs["pxl_row_in_fullres"].to_numpy(float),
            ]
        )
    elif obsm_key in adata.obsm:
        coords = np.asarray(adata.obsm[obsm_key], dtype=float)
    else:
        return adata
    if coords.ndim != 2 or coords.shape[1] < 2:
        return adata

    # Map the current coordinates from fullres pixels to micrometers
    scale = float(spot_diameter_um) / fullres_spot_diameter
    adata.obsm[obsm_key] = coords * scale
    if isinstance(spatial_uns, dict):
        scalefactors["spot_diameter_um"] = float(spot_diameter_um)
        scalefactors["coordinate_um_per_input_unit"] = scale
        scalefactors["coordinate_units"] = "um"
        scalefactors["coordinate_scale_source"] = "spot_diameter_fullres"
    return adata


def add_length_scale_axis(
    ax: Any,
    ticks: Sequence[float] = (1 / 1000, 1 / 500),
    labels: Sequence[str] = ("1 mm", "500 um"),
) -> Any:
    """Add a top axis with physical length scales for radial frequency plots."""
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(ticks)
    sec.set_xticklabels(labels, fontsize=5)
    sec.tick_params(width=0.4, length=2, pad=1)
    for tick in ticks:
        ax.axvline(tick, color="0.72", lw=0.5, ls=":", zorder=0)
    return sec


# ---------------------------------------------------------------------------
# 2. Download helpers (deduplicates the 3 copies in download_*.py)
# ---------------------------------------------------------------------------


def human_size(nbytes: float) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} TB"


def download_with_progress(url: str, dest: Path, expected_min_bytes: int) -> bool:
    """Stream ``url`` to ``dest`` with a CR-overwriting progress line.

    Returns True on success, False on network failure or remote-size sanity
    check fail. Atomic via a ``.part`` temp file.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            total = int(resp.headers.get("content-length", 0) or 0)
            if total and total < expected_min_bytes:
                logger.warning(
                    "Remote %s reports %s (< %s min); skipping.",
                    url,
                    human_size(total),
                    human_size(expected_min_bytes),
                )
                return False
            with tmp.open("wb") as fh:
                done = 0
                chunk = 1 << 20
                t0 = time.time()
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
                    done += len(buf)
                    if total:
                        pct = 100.0 * done / total
                        dt = max(time.time() - t0, 1e-3)
                        print(
                            f"  {dest.name}: {human_size(done)}/{human_size(total)} "
                            f"({pct:5.1f}%, {human_size(done / dt)}/s)\r",
                            end="",
                            flush=True,
                        )
                print()
        tmp.rename(dest)
        return True
    except (urllib.error.URLError, TimeoutError) as e:
        logger.error("Download failed for %s: %s", url, e)
        if tmp.exists():
            tmp.unlink()
        return False


def gunzip_to(src: Path, dst: Path) -> None:
    """Decompress ``src.gz`` → ``dst``. Skip if dst already non-empty."""
    if dst.exists() and dst.stat().st_size > 0:
        return
    with gzip.open(src, "rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout)


# ---------------------------------------------------------------------------
# 3. SampleSet + sample-set builders (real contrast + null variants)
# ---------------------------------------------------------------------------


@dataclass
class SampleSet:
    """One contrast over a list of .h5ad samples.

    Two flavours, distinguished by ``design_values``:

    - **Binary** (``design_values is None``): ``a_paths`` + ``b_paths``
      hold the two arms; the test goes through ``compare_two_groups``.
    - **Continuous-design** (``design_values`` is a 1-D array): ``a_paths``
      holds *every* sample in design-row order, ``b_paths`` is empty,
      ``design_values[i]`` is the per-sample covariate value (e.g.,
      ``log_pn_day``), and the test goes through ``compare_glm``
      with a 1-DOF slope contrast.

    ``null_type`` distinguishes:

    - ``"real"``     — the biological contrast (``is_null=False``)
    - ``"perm"``     — label-shuffled cohort (binary: the (n_a, n_b)
                       marginal is preserved; continuous: the
                       ``design_values`` vector is permuted across the
                       same sample list)
    """

    label: str
    a_paths: list[Path]
    b_paths: list[Path]
    is_null: bool = False
    null_type: str = "real"
    # Continuous-design fields. ``design_values`` parallels ``a_paths``
    # when set; ``design_name`` is the column name passed to
    # ``compare_glm(design=DataFrame({design_name: design_values}),
    # contrast=design_name, ...)``.
    design_values: np.ndarray | None = None
    design_name: str = "covariate"


def build_permutation_nulls(
    real_set: SampleSet,
    n_perm: int = 12,
    rng_seed: int = 42,
    continuous_groups: Sequence[str] | None = None,
) -> list[SampleSet]:
    """Generate null contrasts by shuffling sample-level design values.

    For binary contrasts, the shuffled design is represented back as
    ``a_paths`` / ``b_paths`` so the downstream two-group test path remains
    unchanged. For continuous-design contrasts, the sample list is fixed and
    only ``design_values`` is shuffled. When ``continuous_groups`` is set, each
    permuted design value must come from a different group than the target
    sample, which avoids within-timepoint-only null swaps for brain-dev.
    """
    if real_set.design_values is not None:
        design = np.asarray(real_set.design_values, dtype=float)
        groups = (
            None
            if continuous_groups is None
            else np.asarray(list(continuous_groups), dtype=str)
        )
        if groups is not None and groups.shape[0] != design.shape[0]:
            raise ValueError("continuous_groups must align with design_values")
        rng = np.random.default_rng(rng_seed)
        seen = {tuple(design.tolist())}
        out: list[SampleSet] = []
        attempts = 0
        max_attempts = 1000 * max(n_perm, 1)
        while len(out) < n_perm and attempts < max_attempts:
            attempts += 1
            order = rng.permutation(design.shape[0])
            # When groups are set, skip permutations that would assign any sample the
            # same group as itself. This avoids within-timepoint-only null swaps for
            # the brain development dataset.
            if groups is not None and np.any(groups[order] == groups):
                continue
            permuted = design[order].astype(float)
            key = tuple(permuted.tolist())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                SampleSet(
                    label=f"null_perm_{len(out):02d}",
                    a_paths=list(real_set.a_paths),
                    b_paths=[],
                    is_null=True,
                    null_type="perm",
                    design_values=permuted,
                    design_name=real_set.design_name,
                )
            )
        return out

    pool = list(real_set.a_paths) + list(real_set.b_paths)
    n_a = len(real_set.a_paths)
    n_b = len(real_set.b_paths)
    n = n_a + n_b
    total_unique = math.comb(n, n_a)
    n_perm = min(n_perm, max(0, total_unique - 1))
    rng = np.random.default_rng(rng_seed)
    labels = np.asarray([0] * n_a + [1] * n_b, dtype=np.int8)

    seen = {tuple(labels.tolist())}
    out: list[SampleSet] = []
    attempts = 0
    while len(out) < n_perm and attempts < 200 * n_perm:
        attempts += 1
        permuted = rng.permutation(labels)
        key = tuple(permuted.tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            SampleSet(
                label=f"null_perm_{len(out):02d}",
                a_paths=[
                    path
                    for path, label in zip(pool, permuted, strict=True)
                    if label == 0
                ],
                b_paths=[
                    path
                    for path, label in zip(pool, permuted, strict=True)
                    if label == 1
                ],
                is_null=True,
                null_type="perm",
            )
        )
    return out


def build_sample_sets(panel: str, meta: pd.DataFrame) -> list[SampleSet]:
    """Build the real contrast and permutation nulls for one panel.

    Permutations are deterministic via a per-panel RNG seed so reruns
    reproduce. AD/PIG uses genotype/age metadata; brain development uses
    a continuous postnatal-time covariate.
    """
    if panel == "ad_pig":
        wt18 = meta[(meta["genotype"] == "WT") & (meta["age_months"] == 18)]
        tg18 = meta[(meta["genotype"] == "TG") & (meta["age_months"] == 18)]
        wt = [Path(p) for p in wt18["h5ad_path"]]
        tg = [Path(p) for p in tg18["h5ad_path"]]
        sets = [SampleSet("real_WT18_vs_TG18", wt, tg, is_null=False, null_type="real")]
        sets += build_permutation_nulls(sets[0], n_perm=12, rng_seed=42)
        return sets

    if panel == "brain_dev":
        # All 6 timepoints (P5, P7, P9, P12, P21, P60) of the ANTERIOR
        # plane participate in a single 1-DOF slope test on
        # ``log_pn_day = log10(t+1)``.
        sub = meta[meta["plane"] == "ANTERIOR"].copy()
        sub["h5ad_path"] = sub["h5ad_path"].astype(str)
        # Sort by h5ad_path so the design vector aligns with how
        # ``fit_all_samples`` later indexes via ``sorted(samples.keys())``.
        sub = sub.sort_values("h5ad_path").reset_index(drop=True)
        # Map "P5" → 5.0, "P60" → 60.0, etc., then log-flatten the
        # 5–60-day range to a ~0.95-log-unit span (closer to the natural
        # log-scale of postnatal developmental rate).
        pn_day = sub["timepoint"].str.lstrip("P").astype(float)
        log_pn_day = np.log10(pn_day + 1.0).to_numpy()
        paths = [Path(p) for p in sub["h5ad_path"]]
        real = SampleSet(
            label="real_log_pn_day_anterior",
            a_paths=paths,
            b_paths=[],
            is_null=False,
            null_type="real",
            design_values=log_pn_day,
            design_name="log_pn_day",
        )
        sets = [
            real,
            *build_permutation_nulls(
                real,
                n_perm=BRAIN_DEV_NULL_PERMUTATIONS,
                rng_seed=42,
                continuous_groups=sub["timepoint"].astype(str).tolist(),
            ),
        ]
        return sets

    raise ValueError(f"Unknown panel: {panel!r}")


# ---------------------------------------------------------------------------
# 4. Panel loading
# ---------------------------------------------------------------------------


def _aggregate_var_matrix(x, target_cols: list[int], n_targets: int):
    """Collapse variable columns according to a sparse column mapper."""
    if sparse.issparse(x):
        rows = np.arange(len(target_cols), dtype=int)
        mapper = sparse.csr_matrix(
            (
                np.ones(len(target_cols), dtype=np.float32),
                (rows, np.asarray(target_cols, dtype=int)),
            ),
            shape=(len(target_cols), n_targets),
        )
        return x @ mapper
    arr = np.asarray(x)
    out = np.zeros((arr.shape[0], n_targets), dtype=arr.dtype)
    for source_col, target_col in enumerate(target_cols):
        out[:, target_col] += arr[:, source_col]
    return out


def _apply_ad_pig_probe_qc(adata: ad.AnnData) -> ad.AnnData:
    """Apply AD/PIG probe removals and family-symbol renaming.

    Cross-reactive probes can map to the same output symbol, so columns are
    summed after renaming. This keeps downstream gene names unique and makes
    the raw-count input match the QC described in the manuscript analysis.
    """
    original = pd.Index(adata.var_names.astype(str))
    keep = ~original.isin(AD_PIG_DROP_PROBES)
    names = [AD_PIG_RENAME_PROBES.get(str(gene), str(gene)) for gene in original[keep]]

    target_to_col: dict[str, int] = {}
    col_idx: list[int] = []
    source_probes: dict[str, list[str]] = {}
    for old, new in zip(original[keep], names, strict=True):
        if new not in target_to_col:
            target_to_col[new] = len(target_to_col)
            source_probes[new] = []
        col_idx.append(target_to_col[new])
        source_probes[new].append(str(old))

    subset = adata[:, keep].copy()
    var = pd.DataFrame(index=pd.Index(target_to_col.keys(), name=adata.var.index.name))
    var["source_probes"] = ["/".join(source_probes[name]) for name in var.index]
    var["n_source_probes"] = [len(source_probes[name]) for name in var.index]
    out = ad.AnnData(
        X=_aggregate_var_matrix(subset.X, col_idx, len(target_to_col)),
        obs=subset.obs.copy(),
        var=var,
        uns=subset.uns.copy(),
    )
    for key in subset.obsm.keys():
        out.obsm[key] = subset.obsm[key].copy()
    for key in subset.layers.keys():
        out.layers[key] = _aggregate_var_matrix(
            subset.layers[key], col_idx, len(target_to_col)
        )
    return out


def load_panel(
    panel: str,
) -> tuple[dict[Path, ad.AnnData], pd.DataFrame, list[SampleSet]]:
    """Load AnnData for every sample referenced by any sample-set in ``panel``.

    Reads the metadata table from ``PANELS[panel].metadata_csv``. Skips
    samples whose .h5ad is missing or empty.
    """
    if panel not in PANELS:
        raise ValueError(f"Unknown panel {panel!r}; choices: {sorted(PANELS)}")
    meta_path = PANELS[panel].metadata_csv
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.csv at {meta_path}")
    meta = pd.read_csv(meta_path)

    sets = build_sample_sets(panel, meta)
    needed_paths = sorted({p for s in sets for p in (*s.a_paths, *s.b_paths)})
    samples: dict[Path, ad.AnnData] = {}
    for p in needed_paths:
        if not p.exists():
            logger.warning("missing %s; skipping", p)
            continue
        a = ad.read_h5ad(p)
        if a.n_obs == 0:
            logger.warning("%s has 0 spots; skipping", p)
            continue
        if panel == "ad_pig":
            a = _apply_ad_pig_probe_qc(a)
        else:
            a.var_names_make_unique()
        samples[p] = a
    return samples, meta, sets


def select_gene_panel(
    samples: dict[Path, ad.AnnData],
    presence_threshold: float = PRESENCE_THRESHOLD,
    max_genes: int = MAX_GENES,
) -> list[str]:
    """Intersection of genes present in ≥presence_threshold spots across ALL
    samples, excluding predicted Gm[0-9]*/mt-* genes before capping to
    top-``max_genes`` by mean expression for tractable sweep size."""
    if not samples:
        return []
    gene_sets: list[set[str]] = []
    for _, a in samples.items():
        X = a.X if sparse.issparse(a.X) else sparse.csr_matrix(a.X)
        nnz = np.asarray((X > 0).sum(axis=0)).ravel()
        frac = nnz / max(a.n_obs, 1)
        present = a.var_names[frac >= presence_threshold]
        gene_sets.append(set(present))
    common = sorted(set.intersection(*gene_sets))
    n_before_filter = len(common)
    common = [gene for gene in common if _is_comparison_gene(str(gene))]
    logger.info(
        "  shared gene panel: %d genes across %d samples (%d Gm[0-9]*/mt-* filtered)",
        len(common),
        len(samples),
        n_before_filter - len(common),
    )
    if max_genes is not None and len(common) > max_genes:
        scores = np.zeros(len(common), dtype=float)
        for a in samples.values():
            sub = a[:, common]
            X = sub.X if sparse.issparse(sub.X) else sparse.csr_matrix(sub.X)
            scores += np.asarray(X.mean(axis=0)).ravel()
        scores /= len(samples)
        order = np.argsort(scores)[::-1][:max_genes]
        common = [common[i] for i in sorted(order)]
        logger.info("  capped to top %d genes by mean expression", max_genes)
    return common


def log1p_normalized_samples(samples: dict[Path, ad.AnnData]) -> dict[Path, ad.AnnData]:
    """Return Scanpy median-total-normalized, log1p-transformed sample copies."""
    import scanpy as sc

    out = {}
    for path, adata in samples.items():
        normalized = adata.copy()
        source = "counts" if "counts" in adata.layers else "X"
        matrix = adata.layers[source] if source == "counts" else adata.X
        values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            if np.nanmin(values) < 0:
                raise ValueError(f"{path.name} {source} contains negative values")
            max_fractional = np.nanmax(np.abs(values - np.rint(values)))
            if max_fractional > 1e-4:
                raise ValueError(
                    f"{path.name} {source} is not count-like; "
                    "log normalization expects raw counts"
                )
        normalized.X = (
            matrix.copy() if sparse.issparse(matrix) else np.array(matrix, copy=True)
        )
        sc.pp.normalize_total(normalized)
        sc.pp.log1p(normalized)
        out[path] = normalized
    return out


# ---------------------------------------------------------------------------
# 5. Input transforms + comparator fitting
# ---------------------------------------------------------------------------


def geometric_frequency_edges(linear_edges: np.ndarray) -> np.ndarray:
    """Return the requested geometric radial-frequency edge grid.

    Empty-bin handling is delegated to ``ComparatorIrregular`` via
    ``adaptive_freq_edges=True`` so fitted comparators record both the requested
    and actual retained bin counts.
    """
    edges = np.asarray(linear_edges, dtype=float)
    if (
        edges.ndim != 1
        or edges.size < 3
        or edges[0] != 0
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("linear_edges must start at 0 and be strictly increasing.")

    return np.concatenate(([0.0], np.geomspace(edges[1], edges[-1], edges.size - 1)))


def fit_all_samples(
    samples: dict[Path, ad.AnnData],
    common_genes: list[str],
    feature_mode: str,
    n_radial_bins: int = N_RADIAL_BINS,
    freq_edges: np.ndarray | None = None,
    adaptive_freq_edges: bool = True,
):
    """Fit a ``ComparatorIrregular`` on ALL samples in the panel.

    Returns ``(cmp, sample_paths_in_order)``. The comparator is design-
    agnostic with cached ``cmp.spectra_`` array sub-indexed by sample
    and passed to any binary or GLM contrast at call time via
    ``compare_two_groups`` / ``compare_glm``. Frequency bins are adaptive by
    default so empty radial bins are merged by SONIC and the fitted comparator
    records the actual retained bin count.
    """
    from sonic import ComparatorIrregular

    paths_order = sorted(samples.keys())
    adatas = [samples[p][:, common_genes].copy() for p in paths_order]

    cmp = ComparatorIrregular(
        samples=adatas,
        gene_names=common_genes,
        feature_mode=feature_mode,
        n_radial_bins=n_radial_bins,
        freq_edges=freq_edges,
        adaptive_freq_edges=adaptive_freq_edges,
        obsm_key="spatial",
    )
    t0 = time.time()
    # NB: n_jobs=1 — joblib parallelism with -1 can hang for minutes when other
    # multithreaded numpy work is active. The per-sample loop is sequential here;
    # finufft + scipy.fft are still multi-threaded internally via OpenMP.
    cmp.compute_spectra(n_jobs=1, progress=False)
    dt = time.time() - t0
    logger.info("    fit(%s) on %d samples in %.1fs", feature_mode, len(adatas), dt)
    return cmp, paths_order


# ---------------------------------------------------------------------------
# 6. DE-companion tests on per-sample DC scalars
# ---------------------------------------------------------------------------


def de_test_binary(
    dc_values: np.ndarray,
    groups: np.ndarray,
    gene_names: list[str],
) -> pd.DataFrame:
    """Welch t-test on per-sample DC scalars; thin wrapper around
    ``compare_two_groups_scalar``. Returns a ``Feature / Statistic /
    P_value / P_adj`` DataFrame matching the DF-test output schema.

    The SONIC scalar companion uses an analytic Welch t null.
    """
    from sonic.comparators.multisample import compare_two_groups_scalar

    return compare_two_groups_scalar(
        dc_values,
        groups,
        gene_names=gene_names,
    )


def de_test_continuous(
    dc_values: np.ndarray,
    covariate: np.ndarray,
    gene_names: list[str],
    design_name: str = "covariate",
) -> pd.DataFrame:
    """Per-gene scalar GLM between DC scalars and a continuous covariate.

    Thin wrapper around SONIC's ``compare_glm_scalar``. Returns the same
    ``Feature / Statistic / P_value / P_adj`` schema as the binary helper;
    ``Statistic`` is ``abs(t)`` for the one-DOF OLS contrast.

    This is the natural DE companion for ``brain_dev``'s
    continuous postnatal-time regression — every gene whose mean
    expression scales with ``log_pn_day`` shows up here.
    """
    from sonic.comparators.multisample import compare_glm_scalar

    design = pd.DataFrame({design_name: np.asarray(covariate, dtype=float)})
    return compare_glm_scalar(
        dc_values,
        design,
        contrast=design_name,
        gene_names=gene_names,
    )
