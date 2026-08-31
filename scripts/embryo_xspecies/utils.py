"""Shared loading, gene selection, spectral, and staging helpers."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config as C

logger = logging.getLogger("embryo_xspecies.utils")


def cache_signature(paths: list[Path], arrays: list[np.ndarray] | None = None) -> str:
    """Hash file inventories and small arrays used to validate derived caches."""
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        if not path.exists():
            raise FileNotFoundError(path)
        stat = path.stat()
        try:
            label = path.relative_to(C.PROJECT_ROOT)
        except ValueError:
            label = path
        digest.update(f"{label}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    for array in arrays or []:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def current_cache_manifest(
    result_dir: Path,
    artifacts: tuple[str, ...],
    signature: str,
) -> dict[str, Any] | None:
    """Return the manifest only when every owned artifact matches current inputs."""
    if not all((result_dir / name).exists() for name in artifacts):
        return None
    manifest = json.loads((result_dir / "manifest.json").read_text())
    return manifest if manifest.get("cache_signature") == signature else None


def require_current_bin_size(table: Any, *, label: str = "prepared zarr") -> None:
    """Fail if a prepared table was built with an older physical bin size."""
    got = table.uns.get("bin_size_um", None)
    try:
        ok = np.isclose(float(got), C.BIN_SIZE_UM)
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise RuntimeError(
            f"{label} has bin_size_um={got!r}; expected {C.BIN_SIZE_UM:.1f}. Rerun preprocessing."
        )


def filtered_metadata(
    cfg: Any,
    *,
    stages: list[str] | None = None,
    section_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Return configured sections after stage and explicit-ID selection."""
    stages = list(stages) if stages else list(cfg.KEPT_TIMEPOINTS)
    # Load all sample metadata produced during data download for a species.
    if not Path(cfg.META_CSV).exists():
        raise FileNotFoundError(f"Missing {cfg.META_CSV} — run the download step first.")
    meta = pd.read_csv(cfg.META_CSV)

    # Make sure requested stages and section_ids are configured for this species.
    configured = {section_id for stage in stages for section_id in cfg.ALL_SECTIONS[stage]}
    meta = meta[
        meta["timepoint"].isin(stages) & meta["section_id"].astype(str).isin(configured)
    ].copy()
    missing_configured = sorted(configured - set(meta["section_id"].astype(str)))
    if missing_configured:
        raise RuntimeError(
            f"{cfg.PANEL_KEY} metadata is missing configured sections: {missing_configured[:10]}"
        )
    if not Path(cfg.GEOMETRY_CSV).exists():
        raise FileNotFoundError(f"Missing {cfg.GEOMETRY_CSV} — run preprocessing first.")
    geom = pd.read_csv(cfg.GEOMETRY_CSV)
    cols = [c for c in ("section_id", "ny", "nx", "min_extent_um", "n_bins") if c in geom]
    meta = meta.merge(geom[cols], on="section_id", how="left")
    missing_geometry = meta.loc[meta["ny"].isna(), "section_id"].astype(str).tolist()
    if missing_geometry:
        raise RuntimeError(
            f"{cfg.PANEL_KEY} geometry is missing configured sections: {missing_geometry[:10]}"
        )
    if section_ids is not None:
        requested = set(map(str, section_ids))
        available = set(meta["section_id"].astype(str))
        missing = sorted(requested - available)
        if missing:
            raise RuntimeError(
                f"{cfg.PANEL_KEY} requested sections are not configured: {missing[:10]}"
            )
        meta = meta[meta["section_id"].astype(str).isin(requested)].copy()
    stage_rank = {tp: i for i, tp in enumerate(stages)}
    meta = (
        meta.assign(_stage_rank=meta["timepoint"].map(stage_rank).fillna(len(stage_rank)))
        .sort_values(["_stage_rank", "section_id"])
        .drop(columns="_stage_rank")
        .reset_index(drop=True)
    )
    if meta.empty:
        raise RuntimeError(f"No configured {cfg.PANEL_KEY} sections for {stages}.")
    return meta


# ---------------------------------------------------------------------------
# Gene-level filtering and selection
# ---------------------------------------------------------------------------
def _prepared_genes_from_qc(cfg: Any) -> list[str]:
    """Return the selected prepared genes recorded by preprocessing QC."""
    path = Path(cfg.RESULT_DIR) / "gene_qc.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run preprocessing first.")
    qc = pd.read_csv(path)
    required = {"gene", "selected"}
    missing = required - set(qc.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    selected = qc["selected"].astype(str).str.lower().isin({"true", "1", "yes"})
    return qc.loc[selected, "gene"].astype(str).tolist()


_UNNAMED_GENE_RE = re.compile(
    r"^(?:ENS[A-Z]*G\d+|LOC\d+|Gm\d+|\d+[A-Z]\d+Rik|[A-Z]{2}\d{6}(?:\.\d+)?|RP\d+-|CT[A-Z]?-\d+)"
)
_LNCRNA_GENE_RE = re.compile(r"(?:^LINC\d+|^MIR\d+HG$|^SNHG\d+$|-AS\d*$|-DT$|-IT\d*$)")


def _is_named_non_lnc_non_mito_gene(gene: str) -> bool:
    """Return whether ``gene`` is suitable for requested-size test panels."""
    gene = str(gene).strip()
    if not gene or gene.lower() in {"nan", "none"}:
        return False
    upper = gene.upper()
    if upper.startswith("MT-"):
        return False
    if gene.endswith("Rik") or _UNNAMED_GENE_RE.search(gene) or _LNCRNA_GENE_RE.search(gene):
        return False
    return True


def _genes_detected_in_every_section(
    cfg: Any,
    *,
    min_detected_fraction: float,
) -> set[str]:
    """Return genes detected above a raw-count fraction in every configured section."""
    if min_detected_fraction < 0:
        raise ValueError("min_detected_fraction must be non-negative.")
    meta = filtered_metadata(cfg)
    cache_dir = Path(cfg.RESULT_DIR) / "gene_qc_cache"
    eligible: set[str] | None = None

    # Use the cached per-section gene qc for expression filtering
    for sid in meta["section_id"].astype(str):
        path = cache_dir / f"{sid}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; rerun preprocessing first.")
        with np.load(path, allow_pickle=False) as z:
            genes = z["genes"].astype(str)
            detected_bins = z["detected_bins"].astype(np.float64, copy=False)
            section_qc = json.loads(str(z["section_qc_json"].item()))
        n_bins = int(section_qc["n_bins_raw"])
        if n_bins < 1:
            raise ValueError(f"{path} records no input bins.")
        passing = set(genes[detected_bins / n_bins > min_detected_fraction])
        eligible = passing if eligible is None else eligible & passing

    if not eligible:
        raise RuntimeError(f"No {cfg.PANEL_KEY} genes pass the all-section detection threshold.")
    return eligible


def select_gene_panel(
    cfg: Any,
    n_genes: int | None = None,
    *,
    qc_path: Path | None = None,
) -> list[str]:
    """Return the shared downstream panel for one species.

    Genes must pass preprocessing QC, be named/non-lncRNA/non-mitochondrial,
    and be detected in strictly more than
    ``config.MIN_GENE_DETECTED_BIN_FRACTION`` of observed bins in every
    configured section. Genes are ranked by mean expression. ``n_genes``
    optionally limits the result for explicitly size-constrained analyses.
    """
    if n_genes is not None and n_genes < 1:
        raise ValueError("n_genes must be >= 1.")
    qc_path = qc_path or Path(cfg.RESULT_DIR) / "gene_qc.csv"
    if not qc_path.exists():
        raise FileNotFoundError(f"Missing {qc_path}; run preprocessing first.")
    qc = pd.read_csv(qc_path)
    required = {"gene", "selected", "mean_expr_all_bins"}
    missing = required - set(qc.columns)
    if missing:
        raise ValueError(f"{qc_path} is missing required columns: {sorted(missing)}")
    detected = _genes_detected_in_every_section(
        cfg,
        min_detected_fraction=C.MIN_GENE_DETECTED_BIN_FRACTION,
    )
    eligible = qc[
        qc["selected"].astype(str).str.lower().isin({"true", "1", "yes"})
        & qc["gene"].astype(str).map(_is_named_non_lnc_non_mito_gene)
        & qc["gene"].astype(str).isin(detected)
    ].copy()
    eligible = eligible.sort_values(
        ["mean_expr_all_bins", "gene"], ascending=[False, True], kind="stable"
    )
    genes = eligible["gene"].astype(str).tolist()
    if n_genes is not None and len(genes) < n_genes:
        raise ValueError(
            f"Only {len(genes)} eligible {cfg.PANEL_KEY} genes are available "
            f"after filtering; requested {n_genes}."
        )
    return genes if n_genes is None else genes[:n_genes]


# ---------------------------------------------------------------------------
# adata / spatialdata handling
# ---------------------------------------------------------------------------
def parse_bins_table(x_csc: Any, obs: pd.DataFrame, var: pd.DataFrame) -> Any:
    """Wrap a matrix and metadata as a SpatialData table for the ``bins`` shapes.

    The ``var`` DataFrame is reduced to its index (gene names only) before
    construction. The upstream MOSTA ``.var`` carries metadata columns like
    ``"Regulon - <TF>"`` whose names contain spaces and so are rejected by
    ``spatialdata.models.TableModel.parse`` (allowed: alphanumerics, ``_``,
    ``.``, ``-``).
    """
    import anndata as ad
    from spatialdata.models import TableModel

    var_index_only = pd.DataFrame(index=pd.Index(var.index.copy(), name=var.index.name))
    a = ad.AnnData(X=sp.csc_matrix(x_csc), obs=obs.copy(), var=var_index_only)
    a.obs["region"] = pd.Categorical(["bins"] * a.n_obs)
    a.obs["instance_id"] = np.arange(a.n_obs, dtype=np.int64)
    return TableModel.parse(a, region="bins", region_key="region", instance_key="instance_id")


def _gene_matrix_for_expression_mode(
    table: Any,
    gene_indices: np.ndarray,
    *,
    expression_mode: str,
    normalization_target_sum: float,
) -> sp.csc_matrix:
    """Return selected gene columns as sparse raw counts or log1p-normalized values."""
    if expression_mode not in {"raw_counts", "log1p_norm"}:
        raise ValueError("expression_mode must be 'raw_counts' or 'log1p_norm'.")
    if sp.issparse(table.X):
        x_genes = table.X[:, gene_indices].astype(np.float64, copy=False).tocsr(copy=True)
    else:
        x_all = np.asarray(table.X)
        x_genes = sp.csr_matrix(x_all[:, gene_indices].astype(np.float64, copy=False))
    if expression_mode == "raw_counts":
        return x_genes.tocsc()

    # Compute library size from all genes, then normalize only selected genes.
    if normalization_target_sum <= 0:
        raise ValueError("normalization_target_sum must be positive.")
    lib = np.asarray(table.X.sum(axis=1)).ravel().astype(np.float64, copy=False)
    scale = np.divide(
        float(normalization_target_sum),
        lib,
        out=np.zeros_like(lib, dtype=np.float64),
        where=lib > 0,
    )
    x_genes = x_genes.multiply(scale[:, None]).tocsr()
    if x_genes.nnz:
        x_genes.data = np.log1p(x_genes.data)
    return x_genes.tocsc()


def _domain_union(tables: list[Any]) -> list[str]:
    """Sorted union of non-empty spatial-domain labels across the section tables."""
    labels: set[str] = set()
    for t in tables:
        if "annotation" not in t.obs.columns:
            raise KeyError("Prepared bins_table is missing the 'annotation' column.")
        vals = t.obs["annotation"].astype("string")
        vals = vals[vals.notna() & (vals != "")]
        labels |= set(map(str, vals.unique()))
    return sorted(labels)


def prepared_domain_labels(cfg: Any) -> list[str]:
    """Return the annotation-label union without retaining section tables."""
    import spatialdata as sd

    labels: set[str] = set()
    for sid in filtered_metadata(cfg)["section_id"].astype(str):
        section = sd.read_zarr(cfg.SDATA_DIR / f"{sid}.zarr")
        labels.update(_domain_union([section.tables["bins_table"]]))
        del section
        gc.collect()
    return sorted(labels)


def _table_with_domain_channels(
    table: Any,
    gene_names: list[str],
    domain_union: list[str],
    *,
    expression_mode: str = "raw_counts",
    normalization_target_sum: float = 1e4,
) -> Any:
    """Return a SpatialData table whose ``.X`` is [gene panel | one-hot domain channels].

    The domain channels are the *union* across sections (binary 1/0 per bin),
    so every section's table has identical ``var_names`` — required by
    ``ComparatorGrid`` — with all-zero columns for domains absent from a given
    section. The table is re-parsed via ``TableModel`` so it stays a valid
    SpatialData table; the ``array_row`` / ``array_col`` / ``annotation`` /
    ``obsm['spatial']`` columns are preserved.
    """
    name_to_idx = {str(g): i for i, g in enumerate(table.var_names)}
    missing = [g for g in gene_names if g not in name_to_idx]
    if missing:
        raise KeyError(
            "Selected genes are missing from a section table; rebuild prepared zarrs "
            f"or remove them from the explicit panel. Missing examples: {missing[:10]}"
        )
    gidx = np.array([name_to_idx[g] for g in gene_names], dtype=int)
    x_genes = _gene_matrix_for_expression_mode(
        table,
        gidx,
        expression_mode=expression_mode,
        normalization_target_sum=normalization_target_sum,
    )

    n_obs = table.n_obs
    if domain_union:
        anno = table.obs["annotation"].astype(str).to_numpy()
        cols = [
            sp.csc_matrix((anno == d).astype(np.float64).reshape(n_obs, 1)) for d in domain_union
        ]
        x_dom = sp.hstack(cols, format="csc")
    else:
        x_dom = sp.csc_matrix((n_obs, 0))
    x_all = sp.hstack([x_genes, x_dom], format="csc") if domain_union else x_genes

    var = pd.DataFrame(
        index=list(gene_names) + [f"{C.DOMAIN_CHANNEL_PREFIX}{d}" for d in domain_union]
    )
    # Keep only the obs columns the comparator + figure need (region/instance_id
    # are re-created inside parse_bins_table).
    obs = table.obs[["array_row", "array_col", "annotation"]].copy()
    new = parse_bins_table(x_all, obs, var)
    new.obsm["spatial"] = np.asarray(table.obsm["spatial"], dtype=float)
    return new


# ---------------------------------------------------------------------------
# Spatial comparison helpers
# ---------------------------------------------------------------------------
def build_comparator(
    sections: list[Any],
    gene_names: list[str],
    *,
    n_radial_bins: int = C.N_RADIAL_BINS,
    bin_size_um: float = C.BIN_SIZE_UM,
    freq_edges: np.ndarray | None = None,
    fft_chunk_size: int | str = "auto",
    adaptive_freq_edges: bool = False,
) -> Any:
    """Construct ``ComparatorGrid`` with cycles/µm physical frequency bins.

    The ``spacing=(bin_size_um, bin_size_um)`` is what makes the auto-derived
    (or supplied) ``freq_edges`` come out in cycles per µm rather than
    normalized frequency. ``fft_chunk_size`` defaults to ``'auto'``, which lets
    ``ComparatorGrid`` size the per-chunk dense block from the (lazily-known)
    lattice shapes so peak RAM stays bounded even for very large lattices.

    ``adaptive_freq_edges`` is frequency-grid bookkeeping: it merges radial
    Fourier bins that have no modes in some samples instead of dropping them, so
    every retained frequency interval is supported by **all** sections. Spatial
    missing-bin masking is handled inside ``sonic`` rasterization.
    """
    from sonic import ComparatorGrid

    return ComparatorGrid(
        sections,
        bins="bins",
        table_name="bins_table",
        col_key="array_col",
        row_key="array_row",
        gene_names=list(gene_names),
        feature_mode="radial",
        n_radial_bins=n_radial_bins,
        fft_solver="rfft2",
        spacing=(float(bin_size_um), float(bin_size_um)),
        freq_edges=freq_edges,
        fft_chunk_size=(fft_chunk_size if fft_chunk_size == "auto" else int(fft_chunk_size)),
        adaptive_freq_edges=adaptive_freq_edges,
    )


def load_prepared_sections(
    cfg: Any,
    *,
    stages: list[str] | None = None,
    section_ids: list[str] | None = None,
    gene_panel: list[str] | None = None,
    expression_mode: str = C.ANALYSIS_EXPRESSION_MODE,
    normalization_target_sum: float = C.ANALYSIS_NORMALIZATION_TARGET_SUM,
    include_domain_channels: bool = True,
    domain_labels: list[str] | None = None,
) -> tuple[list[Any], pd.DataFrame, list[str], list[str]]:
    """Load configured prepared sections and construct comparator-ready tables.

    Prepared zarrs use the standardized ``annotation`` column written by
    preprocessing. Each returned ``bins_table`` contains the selected genes,
    optionally followed by one-hot domain channels shared across sections.

    Parameters
    ----------
    cfg : object
        A species config exposing ``META_CSV``, ``GEOMETRY_CSV``,
        ``SDATA_DIR`` and ``KEPT_TIMEPOINTS``.
    stages : list of str, optional
        Developmental stages to keep (default ``cfg.KEPT_TIMEPOINTS``).
    section_ids : list of str, optional
        Exact configured section IDs to load.
    gene_panel : list of str, optional
        Pre-chosen gene panel from ``select_gene_panel``; if ``None`` the
        prepared zarr gene set is used.
    expression_mode : {"raw_counts", "log1p_norm"}
        Matrix passed to ``sonic``. ``log1p_norm`` library-size normalizes each
        observed bin to ``normalization_target_sum`` using all prepared genes,
        then applies sparse ``log1p`` to selected gene columns.
    include_domain_channels : bool
        Append one-hot annotation columns. Disable for gene-only visualization.
    domain_labels : list of str, optional
        Fixed annotation-label vocabulary. Supplying the species-wide union
        lets callers load one section at a time while keeping identical
        comparator features across all section caches.

    Returns
    -------
    tuple
        ``(sections, meta_df, gene_panel, domain_channels)``.
    """
    import spatialdata as sd

    meta = filtered_metadata(
        cfg,
        stages=stages,
        section_ids=section_ids,
    )

    sections = []
    for sid in meta["section_id"]:
        zpath = cfg.SDATA_DIR / f"{sid}.zarr"
        if not zpath.exists():
            raise FileNotFoundError(f"Missing {zpath} — run preprocessing first.")
        sections.append(sd.read_zarr(zpath))

    raw_tables = [s.tables["bins_table"] for s in sections]
    for sid, table in zip(meta["section_id"], raw_tables, strict=True):
        require_current_bin_size(table, label=f"{cfg.PANEL_KEY} {sid}")
    expected_genes = _prepared_genes_from_qc(cfg)
    stale = [
        str(sid)
        for sid, table in zip(meta["section_id"], raw_tables, strict=True)
        if list(map(str, table.var_names)) != expected_genes
    ]
    if stale:
        raise RuntimeError(
            f"{cfg.PANEL_KEY} prepared zarrs do not match "
            f"{cfg.RESULT_DIR / 'gene_qc.csv'}; rerun "
            "`python -m scripts.embryo_xspecies.preprocessing "
            f"--species {cfg.PANEL_KEY} --force`. "
            f"Stale examples: {stale[:5]}"
        )
    if gene_panel is None:
        gene_panel = expected_genes
    if not include_domain_channels:
        domain_union = []
    elif domain_labels is None:
        domain_union = _domain_union(raw_tables)
    else:
        domain_union = list(domain_labels)
    domain_channels = [f"{C.DOMAIN_CHANNEL_PREFIX}{d}" for d in domain_union]
    cfg_label = cfg.PANEL_KEY
    logger.info(
        "%s: %d genes + %d organ channels across %d sections.",
        cfg_label,
        len(gene_panel),
        len(domain_channels),
        len(sections),
    )
    for s in sections:
        s.tables["bins_table"] = _table_with_domain_channels(
            s.tables["bins_table"],
            list(gene_panel),
            domain_union,
            expression_mode=expression_mode,
            normalization_target_sum=normalization_target_sum,
        )
    keep_cols = [
        c for c in ("section_id", "timepoint", "day", "min_extent_um", "n_bins") if c in meta
    ]
    return (
        sections,
        meta[keep_cols].reset_index(drop=True),
        list(gene_panel),
        domain_channels,
    )


def shared_supported_freq_edges(
    cfgs: list[Any],
    *,
    spacing: str = C.FREQ_SPACING,
    n_radial_bins: int = C.N_RADIAL_BINS,
    bin_size_um: float = C.BIN_SIZE_UM,
) -> np.ndarray:
    """Build one radial grid supported by every configured section.

    The candidate grid spans zero to Nyquist, with geometric or linear spacing.
    SONIC then merges adjacent intervals until each has at least one Fourier
    mode in every section.
    """
    from sonic.comparators.features import adapt_frequency_edges

    if n_radial_bins < 1:
        raise ValueError("n_radial_bins must be positive.")
    if bin_size_um <= 0:
        raise ValueError("bin_size_um must be positive.")
    metadata = [filtered_metadata(cfg) for cfg in cfgs]
    geometry = pd.concat(metadata, ignore_index=True)
    extents = pd.to_numeric(geometry["min_extent_um"], errors="coerce")
    extents = extents[np.isfinite(extents) & (extents > 0)]
    if extents.empty:
        raise RuntimeError("Configured sections have no positive physical extents.")

    nyquist = 1.0 / (2.0 * bin_size_um)
    if spacing == "geom":
        if n_radial_bins == 1:
            requested = np.array([0.0, nyquist * (1.0 + 1e-9)])
        else:
            lowest = 1.0 / float(extents.min())
            if lowest >= nyquist:
                raise RuntimeError("Smallest section cannot resolve a nonzero frequency.")
            requested = np.concatenate(
                [
                    [0.0],
                    np.geomspace(lowest, nyquist * (1.0 + 1e-9), n_radial_bins),
                ]
            )
    elif spacing == "linear":
        requested = np.linspace(0.0, nyquist * (1.0 + 1e-9), n_radial_bins + 1)
    else:
        raise ValueError("spacing must be 'linear' or 'geom'.")

    shapes = list(
        zip(
            geometry["ny"].astype(int),
            geometry["nx"].astype(int),
            strict=True,
        )
    )
    spacings = [(bin_size_um, bin_size_um)] * len(shapes)
    out = adapt_frequency_edges(shapes, spacings, requested, fft_solver="rfft2")
    logger.info(
        "Shared SUPPORTED freq grid: %d→%d bins (merged %d unsupported) across %d sections.",
        requested.size - 1,
        out.size - 1,
        requested.size - out.size,
        len(shapes),
    )
    return out


# ---------------------------------------------------------------------------
# Shared Carnegie-stage cross-species developmental axis
# ---------------------------------------------------------------------------
# Mouse embryonic day (dpc) → morphologically equivalent human Carnegie stage,
# from the UNSW Carnegie Stage Comparison (midpoints of the published CS↔E
# The comparison spans mouse E9-E16 and human CS9-CS23 at half-day/integer-CS
# increments, giving CS = 2 * E - 9 over the embryonic interval.
# Source: https://embryology.med.unsw.edu.au/embryology/index.php/Carnegie_Stage_Comparison
_MOUSE_E_MIN = 9.0
_MOUSE_E_MAX = 16.0


def mouse_e_to_carnegie(e_day: float | np.ndarray) -> float | np.ndarray:
    """Mouse embryonic day (dpc) → equivalent human Carnegie stage.

    The source comparison maps E9-E16 to CS9-CS23 in half-day mouse and
    integer-CS increments. Values outside E9-E16 are rejected because Carnegie
    staging does not define a corresponding interval there.
    """
    e = np.asarray(e_day, dtype=float)
    if np.any((e < _MOUSE_E_MIN) | (e > _MOUSE_E_MAX)):
        raise ValueError("Mouse Carnegie mapping is defined only for E9-E16.")
    cs = 2.0 * e - 9.0
    return float(cs) if cs.ndim == 0 else cs


def human_label_to_carnegie(label: str) -> float:
    """Human HESTA stage label → Carnegie stage number.

    Human embryos are natively Carnegie-staged, so the CS number is parsed
    directly from the label; hyphenated bins are averaged
    (``"CS12-13"`` → 12.5, ``"CS14-15"`` → 14.5, ``"CS17"`` → 17).
    """
    nums = [int(x) for x in re.findall(r"\d+", str(label))]
    if not nums:
        raise ValueError(f"Cannot parse Carnegie stage from {label!r}.")
    return float(np.mean(nums))


def carnegie_stage(
    species: str, *, timepoint: str | None = None, day: float | None = None
) -> float:
    """Carnegie-stage value for a sample on the shared cross-species axis.

    Human samples are mapped from their CS label (:func:`human_label_to_carnegie`);
    mouse samples from their embryonic day (:func:`mouse_e_to_carnegie`).
    """
    if species == "human":
        return human_label_to_carnegie(timepoint)
    if species == "mouse":
        if day is None:
            raise ValueError("mouse carnegie_stage requires day=<embryonic day>.")
        return float(mouse_e_to_carnegie(day))
    raise ValueError(f"Unsupported species {species!r}; expected 'mouse' or 'human'.")
