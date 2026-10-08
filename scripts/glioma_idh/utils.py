"""Shared helpers for the Visium human glioma workflow and visualization notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from scripts.paper_utils import (
    ensure_directories,
    log,
    project_relative,
    save_csv,
)

from .config import (
    DATA_ROOT,
    FIGURE_DIR,
    GROUP_LABELS,
    RESULT_DIR,
    SAMPLES,
)

__all__ = ["log", "project_relative", "save_csv"]


def ensure_dirs() -> None:
    """Create workflow output directories."""
    ensure_directories(RESULT_DIR, FIGURE_DIR)


def sample_metadata() -> pd.DataFrame:
    """Build a deterministic sample metadata table for the configured cohort."""
    data_root = DATA_ROOT.expanduser().resolve()
    rows = []
    for sample_id, group in SAMPLES:
        rows.append(
            {
                "sample_id": sample_id,
                "group": group,
                "group_label": GROUP_LABELS[group],
                "h5ad_path": str(data_root / "adata" / f"{sample_id}.h5ad"),
            }
        )
    return pd.DataFrame(rows)


def normalize_total_log1p(adata: ad.AnnData, *, target_sum: float = 1e4) -> None:
    """Normalize one sample in-place using Scanpy normalize_total + log1p."""
    import scanpy as sc

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)


def _load_normalized_samples_from_meta(
    meta: pd.DataFrame,
    *,
    missing_context: str,
    genes: list[str] | None = None,
) -> tuple[dict[str, ad.AnnData], pd.DataFrame]:
    """Load h5ad paths from a metadata table and return log-normalized objects."""
    if "h5ad_path" not in meta.columns:
        raise KeyError("Sample metadata must include an 'h5ad_path' column.")
    meta = meta.copy()
    if any(not Path(path).expanduser().is_absolute() for path in meta["h5ad_path"]):
        raise ValueError("Expected absolute sample paths under config.DATA_ROOT.")
    meta["h5ad_path"] = [str(Path(path).expanduser()) for path in meta["h5ad_path"]]
    missing = [Path(p) for p in meta["h5ad_path"] if not Path(p).exists()]
    if missing:
        formatted = "\n".join(f"  - {p}" for p in missing[:8])
        extra = "" if len(missing) <= 8 else f"\n  ... and {len(missing) - 8} more"
        raise FileNotFoundError(
            f"Missing {len(missing)} glioma .h5ad files {missing_context}:\n"
            f"{formatted}{extra}\n"
            "Check DATA_ROOT in config.py and run download_glioma_data.py."
        )

    samples: dict[str, ad.AnnData] = {}
    for row in meta.itertuples(index=False):
        sid = str(row.sample_id)
        path = Path(row.h5ad_path)
        log(f"loading {sid}")
        from .download_glioma_data import _h5ad_is_current

        if not _h5ad_is_current(path):
            raise ValueError(
                f"Stale QC or coordinate metadata in {path}; rerun download_glioma_data."
            )
        a = ad.read_h5ad(path)
        if a.n_obs == 0:
            raise ValueError(f"{path} has 0 spots")
        a.var_names_make_unique()
        normalize_total_log1p(a)
        samples[sid] = a if genes is None else align_samples_to_genes({sid: a}, genes)[sid]
        del a
    return samples, meta


def load_normalized_samples() -> tuple[dict[str, ad.AnnData], pd.DataFrame]:
    """Load every configured glioma sample under config.DATA_ROOT."""
    meta = sample_metadata()
    return _load_normalized_samples_from_meta(
        meta,
        missing_context=f"under {DATA_ROOT}",
    )


def load_normalized_samples_from_metadata(
    meta: pd.DataFrame,
    *,
    genes: list[str] | None = None,
) -> tuple[dict[str, ad.AnnData], pd.DataFrame]:
    """Normalize full libraries, then retain only requested genes when supplied."""
    return _load_normalized_samples_from_meta(
        meta.copy(),
        missing_context="referenced by cached spectra metadata",
        genes=genes,
    )


def _as_sparse(x: Any) -> sparse.spmatrix:
    return x if sparse.issparse(x) else sparse.csr_matrix(np.asarray(x))


def select_gene_panel(
    samples: dict[str, ad.AnnData],
    *,
    max_genes: int,
    min_frac_per_sample: float,
    min_samples_frac: float,
) -> tuple[list[str], pd.DataFrame]:
    """Select the top expressed genes after QC, with detection fields for audit."""
    n_samples = len(samples)
    n_needed = int(np.ceil(min_samples_frac * n_samples))
    detected_counts: dict[str, int] = {}
    mean_sums: dict[str, float] = {}

    for a in samples.values():
        x = _as_sparse(a.X)
        nnz = np.asarray((x > 0).sum(axis=0)).ravel()
        means = np.asarray(x.mean(axis=0)).ravel()
        det_frac = nnz / max(a.n_obs, 1)
        for gene, frac, mean in zip(a.var_names.astype(str), det_frac, means, strict=True):
            mean_sums[gene] = mean_sums.get(gene, 0.0) + float(mean)
            if frac >= min_frac_per_sample:
                detected_counts[gene] = detected_counts.get(gene, 0) + 1

    rows = []
    for gene in sorted(mean_sums):
        n_detected = detected_counts.get(gene, 0)
        rows.append(
            {
                "gene": gene,
                "n_detected_samples": n_detected,
                "mean_log_norm": mean_sums[gene] / max(n_samples, 1),
                "passes_detection": n_detected >= n_needed,
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["mean_log_norm", "gene"],
        ascending=[False, True],
        kind="stable",
    )
    genes = summary.loc[summary["passes_detection"], "gene"].head(max_genes).tolist()
    log(
        f"selected top {len(genes):,} detected genes by mean log-normalized "
        f"expression after QC; threshold is >= {min_frac_per_sample:.0%} "
        f"spots in >= {n_needed}/{n_samples} samples"
    )
    return genes, summary


def align_samples_to_genes(
    samples: dict[str, ad.AnnData],
    genes: list[str],
) -> dict[str, ad.AnnData]:
    """Return AnnData copies with identical gene order, zero-filling missing genes."""
    aligned: dict[str, ad.AnnData] = {}
    gene_index = pd.Index(genes)
    for sid, a in samples.items():
        src = {str(g): i for i, g in enumerate(a.var_names)}
        idx = [src.get(g, -1) for g in genes]
        present = np.asarray([i >= 0 for i in idx], dtype=bool)
        src_idx = np.asarray([i for i in idx if i >= 0], dtype=int)
        dst_idx = np.flatnonzero(present)
        if sparse.issparse(a.X):
            if present.any():
                x_sub = a.X[:, src_idx].tocoo()
                x_new = sparse.csr_matrix(
                    (
                        x_sub.data.astype(np.float32, copy=False),
                        (x_sub.row, dst_idx[x_sub.col]),
                    ),
                    shape=(a.n_obs, len(genes)),
                )
            else:
                x_new = sparse.csr_matrix((a.n_obs, len(genes)), dtype=np.float32)
        else:
            x_new = np.zeros((a.n_obs, len(genes)), dtype=np.float32)
            if present.any():
                x_new[:, dst_idx] = np.asarray(a.X[:, src_idx], dtype=np.float32)
        new = ad.AnnData(X=x_new, obs=a.obs.copy(), var=pd.DataFrame(index=gene_index))
        if "spatial" not in a.obsm:
            raise KeyError(f"{sid} has no obsm['spatial']")
        new.obsm["spatial"] = np.asarray(a.obsm["spatial"], dtype=np.float64)
        new.uns["sample_id"] = sid
        aligned[sid] = new
    return aligned


def merge_real_tests(de: pd.DataFrame, df: pd.DataFrame, df_norm: pd.DataFrame) -> pd.DataFrame:
    """Merge DE, DF, and DF-norm outputs into the visualization table."""
    out = (
        de[["Feature", "Statistic", "P_value", "P_adj"]]
        .rename(columns={"Statistic": "DE_stat", "P_value": "P_DE", "P_adj": "P_adj_DE"})
        .merge(
            df[["Feature", "Statistic", "P_value", "P_adj"]].rename(
                columns={"Statistic": "DF_stat", "P_value": "P_DF", "P_adj": "P_adj_DF"}
            ),
            on="Feature",
            how="inner",
        )
        .merge(
            df_norm[["Feature", "Statistic", "P_value", "P_adj"]].rename(
                columns={
                    "Statistic": "DF_norm_stat",
                    "P_value": "P_DF_norm",
                    "P_adj": "P_adj_DF_norm",
                }
            ),
            on="Feature",
            how="inner",
        )
    )
    return out.sort_values("P_DF", kind="stable").reset_index(drop=True)
