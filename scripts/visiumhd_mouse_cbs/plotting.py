from __future__ import annotations

from typing import Any

import numpy as np

from .reference_annotations import (
    _as_patterns,
    _select_bed_rows,
    _select_ont_rows,
    _select_reference_rows,
)
from .utils import normalize_gene_symbol

MAX_FIGURE_WIDTH_IN = 185 / 25.4
MIN_FONT_SIZE = 5


def bounded_font_size(size: float) -> float:
    """Keep figure text at or above the publication minimum."""
    return max(MIN_FONT_SIZE, float(size))


def orient_spatial_image(image: np.ndarray, dataset_key: str) -> np.ndarray:
    """Apply the notebook's display orientation for each Visium HD platform."""
    out = np.asarray(image)
    if dataset_key == "ont_tx":
        out = np.flip(out, axis=0)
        return np.rot90(out, 3)
    if dataset_key == "trend_3prime":
        out = np.flip(out, axis=0)
        return np.rot90(out)
    return out


def feature_names_for_gene(
    adata: Any,
    table_cfg: dict[str, Any],
    gene: str,
    *,
    max_features: int | None = 5,
) -> list[str]:
    """Return the most abundant spatial features assigned to one gene."""
    gene_key = normalize_gene_symbol(gene)
    mask = np.zeros(adata.n_vars, dtype=bool)
    for col in (table_cfg.get("gene_names"), table_cfg.get("group_iso_by")):
        if col and col in adata.var:
            mask |= adata.var[col].astype(str).map(normalize_gene_symbol).eq(gene_key).to_numpy()

    names = adata.var_names[mask].astype(str).tolist()
    if max_features is None or len(names) <= max_features:
        return names

    idx = adata.var_names.get_indexer(names)
    counts = np.asarray(adata[:, idx].layers.get("counts", adata[:, idx].X).sum(axis=0)).ravel()
    return [names[i] for i in np.argsort(-counts)[:max_features]]


def short_spatial_feature_label(feature: object) -> str:
    """Shorten transcript, probe, and peak IDs for compact map titles."""
    text = str(feature)
    if text.startswith("ENSMUST"):
        return f"...{text[-5:]}"
    if "|" in text:
        return text.split("|")[-1]
    parts = text.split(":")
    if len(parts) >= 4:
        interval = parts[2]
        if "-" in interval:
            start, end = interval.split("-", 1)
            return f"{start[-5:]}-{end[-5:]}"
        return interval
    return text


def spatial_feature_image(
    sdata_cache: dict[str, Any],
    dataset_key: str,
    feature: str,
) -> tuple[np.ndarray | None, str]:
    """Load one rasterized feature as an oriented log1p count image."""
    if dataset_key not in sdata_cache:
        return None, "Spatial cache not loaded"
    try:
        cache = sdata_cache[dataset_key]
        data = cache["sdata"][cache["raster_key"]].sel(c=[feature]).values
    except Exception:
        return None, f"Missing: {short_spatial_feature_label(feature)}"

    image = np.asarray(data, dtype=float).squeeze()
    image = np.where(image > 0, np.log1p(image), np.nan)
    return orient_spatial_image(image, dataset_key), short_spatial_feature_label(feature)


def spatial_gene_expression_image(
    sdata_cache: dict[str, Any],
    dataset_key: str,
    gene: str,
    *,
    max_features: int | None = None,
) -> tuple[np.ndarray | None, str]:
    """Load total gene-level raster signal as an oriented log1p count image."""
    if not gene or dataset_key not in sdata_cache:
        return None, "Spatial cache not loaded"

    cache = sdata_cache[dataset_key]
    features = feature_names_for_gene(
        cache["adata"], cache["table_cfg"], gene, max_features=max_features
    )
    if not features:
        return None, f"{gene} not found"

    data = cache["sdata"][cache["raster_key"]].sel(c=features).values
    cube = np.moveaxis(np.asarray(data, dtype=float), 0, -1)
    total = np.nansum(cube, axis=-1)
    image = np.where(total > 0, np.log1p(total), np.nan)
    return orient_spatial_image(image, dataset_key), f"{gene} expression"


def enhance_sparse_spatial_image(image: np.ndarray | None, *, dilate: bool = False):
    """Optionally thicken sparse nonzero pixels without changing map resolution."""
    if image is None:
        return None
    arr = np.asarray(image, dtype=float)
    if not dilate:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return arr
    try:
        from scipy.ndimage import maximum_filter
    except ImportError:
        return arr
    values = np.where(finite, arr, 0.0)
    mask = maximum_filter(finite.astype(np.uint8), size=2).astype(bool)
    expanded = maximum_filter(values, size=2)
    return np.where(mask, expanded, np.nan)


def crop_spatial_image_to_signal(image: np.ndarray | None, *, margin: int = 8):
    """Crop empty borders around finite spatial signal."""
    if image is None:
        return None
    arr = np.asarray(image, dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return arr
    rows, cols = np.where(mask)
    r0 = max(0, int(rows.min()) - margin)
    r1 = min(arr.shape[0], int(rows.max()) + margin + 1)
    c0 = max(0, int(cols.min()) - margin)
    c1 = min(arr.shape[1], int(cols.max()) + margin + 1)
    return arr[r0:r1, c0:c1]


def prepare_spatial_map_image(
    image: np.ndarray | None,
    *,
    dilate: bool = False,
    margin: int = 8,
):
    """Apply sparse-map enhancement and crop before equal-size panel padding."""
    enhanced = enhance_sparse_spatial_image(image, dilate=dilate)
    return crop_spatial_image_to_signal(enhanced, margin=margin)


def pad_spatial_image_to_shape(
    image: np.ndarray | None,
    shape: tuple[int, int],
    *,
    x_align: str = "center",
):
    """Center-pad a spatial image with NaNs for equal-sized imshow panels."""
    if image is None:
        return None
    arr = np.asarray(image, dtype=float)
    out = np.full(shape, np.nan, dtype=float)
    top = max(0, (shape[0] - arr.shape[0]) // 2)
    free_x = max(0, shape[1] - arr.shape[1])
    if x_align == "left":
        left = 0
    elif x_align == "right":
        left = free_x
    else:
        left = free_x // 2
    out[top : top + arr.shape[0], left : left + arr.shape[1]] = arr
    return out


def format_colorbar_tick(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def shrink_colorbar_axis(cax: Any, *, width: float = 0.72) -> None:
    """Shrink a colorbar axis within its grid cell to avoid adjacent label overlap."""
    bbox = cax.get_position()
    dx = bbox.width * (1 - width) / 2
    cax.set_position([bbox.x0 + dx, bbox.y0, bbox.width * width, bbox.height])


def _short_tx_id(value: str) -> str:
    tx_id = str(value).strip()
    if tx_id.startswith(("ENSMUST", "ENSMUSG")) and len(tx_id) > 12:
        return f"{tx_id[:7]}...{tx_id[-5:]}"
    if len(tx_id) <= 18:
        return tx_id
    return f"{tx_id[:7]}...{tx_id[-5:]}"


def _transcript_row_label(label: str, *, include_id: bool) -> str:
    label = str(label)
    if "|" not in label:
        return _short_tx_id(label)
    left, tx_name = [part.strip() for part in label.rsplit("|", 1)]
    tx_id = left.split("->", 1)[0].strip()
    if include_id and tx_id and tx_id != tx_name:
        return f"{tx_name} ({_short_tx_id(tx_id)})"
    return tx_name


def _probe_row_label(row: dict[str, Any]) -> str:
    name = str(row.get("name") or row.get("label", ""))
    parts = name.split("|")
    if len(parts) >= 3 and parts[0].startswith("ENSMUSG"):
        return parts[-1]
    if len(parts) >= 2 and parts[0].startswith("ENS"):
        return "|".join(parts[1:])
    return str(row.get("label", name))


def _trend_row_label(row: dict[str, Any]) -> str:
    name = str(row.get("name") or row.get("label", ""))
    parts = name.split(":")
    if len(parts) >= 4:
        return parts[2]
    return str(row.get("label", name))


def _short_group_label(group: str) -> str:
    return {
        "ONT transcripts": "ONT",
        "Reference transcripts": "Reference",
        "FFPE probes": "FFPE probe",
        "3' TREND peaks": "3' TREND peak",
    }.get(group, group)


def _feature_structure_row_label(row: dict[str, Any]) -> str:
    group = row["group"]
    if group == "ONT transcripts":
        return _transcript_row_label(str(row["label"]), include_id=True)
    if group == "Reference transcripts":
        return _transcript_row_label(str(row["label"]), include_id=False)
    if group == "FFPE probes":
        return _probe_row_label(row)
    if group == "3' TREND peaks":
        return _trend_row_label(row)
    return str(row.get("label", ""))[:40]


def _draw_feature_structure(gene_name: str, rows: list[dict[str, Any]]):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {
        "ONT transcripts": "#D55E00",
        "Reference transcripts": "#999999",
        "FFPE probes": "#009E73",
        "3' TREND peaks": "#0072B2",
    }
    x_min = min(start for row in rows for start, _ in row["blocks"])
    x_max = max(end for row in rows for _, end in row["blocks"])
    pad = max(1, (x_max - x_min) * 0.08)

    fig, ax = plt.subplots(figsize=(MAX_FIGURE_WIDTH_IN, max(3, 0.32 * len(rows) + 1.2)))
    y_labels = []
    for y, row in enumerate(rows):
        blocks = sorted(row["blocks"])
        y_labels.append(_feature_structure_row_label(row))
        ax.plot(
            [min(start for start, _ in blocks), max(end for _, end in blocks)],
            [y, y],
            color="0.7",
            linewidth=0.8,
        )
        for start, end in blocks:
            ax.barh(
                y,
                end - start,
                left=start,
                height=0.5,
                color=colors.get(row["group"], "0.5"),
                edgecolor="black",
                linewidth=0.4,
            )

    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(y_labels, fontsize=bounded_font_size(plt.rcParams["font.size"] - 1))
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlabel("Genomic coordinate (bp)")
    ax.set_title(f"Feature structures: {gene_name}")
    ax.grid(axis="x", alpha=0.25)
    legend_groups = list(dict.fromkeys(row["group"] for row in rows))
    ax.legend(
        handles=[
            Patch(facecolor=colors.get(group, "0.5"), edgecolor="black", label=_short_group_label(group))
            for group in legend_groups
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0, 1.02),
        ncol=min(4, len(legend_groups)),
    )
    fig.tight_layout()
    return fig


def plot_feature_structure(
    gene_name: str,
    reference_cache: dict[str, Any],
    *,
    ref_list: str | list[str] | tuple[str, ...] | None = None,
    ont_tx_list: str | list[str] | tuple[str, ...] | None = None,
    trend_list: str | list[str] | tuple[str, ...] | None = None,
    probe_list: str | list[str] | tuple[str, ...] | None = None,
    include_reference: bool = True,
    sdata_cache: dict[str, Any] | None = None,
    resolution: str = "16um",
):
    """Plot ONT isoforms, optional reference transcripts, 3' TREND peaks, and FFPE probes."""
    gene_key = normalize_gene_symbol(gene_name)
    ref_patterns = _as_patterns(ref_list)
    ont_patterns = _as_patterns(ont_tx_list)
    trend_patterns = _as_patterns(trend_list)
    probe_patterns = _as_patterns(probe_list)

    rows, selected_ref = _select_ont_rows(
        gene_name, reference_cache, ont_patterns, sdata_cache, resolution
    )
    if include_reference:
        rows.extend(_select_reference_rows(gene_key, reference_cache, ref_patterns, selected_ref))
    rows.extend(
        _select_bed_rows("trend_3prime", gene_name, reference_cache, trend_patterns, sdata_cache, resolution)
    )
    rows.extend(
        _select_bed_rows("ffpe_probe", gene_name, reference_cache, probe_patterns, sdata_cache, resolution)
    )

    if not rows:
        print(f"No structure rows found for {gene_name}")
        return None

    return _draw_feature_structure(gene_name, rows)
