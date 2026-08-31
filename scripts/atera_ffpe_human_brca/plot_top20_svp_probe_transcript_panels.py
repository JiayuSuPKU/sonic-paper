"""Render top SVP probe/transcript inspection panels from workflow caches.

Source data:
    - Canonical SVP results from ``config.SV_CSV``.
    - Spatial Atera codeword counts from ``config.SDATA_ZARR``.
    - Workflow-owned all-SVP probe mapping caches under
      ``config.SVP_PROBE_RESULT_DIR``.
    - GENCODE and PacBio transcript annotations used only for drawing
      transcript structures.

This script is intentionally downstream-only. It does not align probe
sequences or recompute probe-to-transcript compatibility; those expensive
all-SVP caches are built by ``workflow.py`` through ``probe_mapping.py``.

Idempotent pipeline:

1. Read the top ``--n-top`` SVP genes, or the explicit ``--genes`` list.
2. Load cached all-SVP probe mappings and transcript compatibility tables.
3. Normalize only the selected Atera codeword columns for spatial thumbnails.
4. Write lightweight renderer summaries under
   ``results/atera_ffpe_human_brca/top20_svp_probe_transcript_panels/``.
5. With ``--render``, write per-gene inspection PNGs under
   ``figures/atera_ffpe_human_brca/top20_svp_probe_transcript_panels/``.

Reruns are cheap: existing summaries and PNGs are reused unless ``--force`` is
provided.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spatialdata as sd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.atera_ffpe_human_brca"

from scripts.paper_utils import (
    FONT_CONFIG,
    configure_paper_matplotlib,
)

from .config import (
    GENCODE_GTF,
    LAYER,
    PACBIO_GFF,
    PROBE_CSV,
    SDATA_ZARR,
    SV_CSV,
    SVP_PROBE_RESULT_DIR,
    TABLE_CODEWORD,
    TOP20_FIGURE_DIR,
    TOP20_RESULT_DIR,
)
from .probe_mapping import (
    gene_loci,
    merge_intervals,
    parse_segments,
    read_gencode_subset,
    read_pacbio_subset,
    read_probe_info,
)
from .utils import log, project_path, selected_to_dense

configure_paper_matplotlib(
    fonts=FONT_CONFIG,
    savefig_dpi=450,
)


SVP_CSV = SV_CSV
OUT_DIR = TOP20_RESULT_DIR
FIG_DIR = TOP20_FIGURE_DIR
N_TOP = 20
NORM_SCALE = 10_000.0
MAX_DISPLAY_PER_GROUP = 4
MAX_DISPLAY_PER_SOURCE = 18


def read_svp_top(path: Path, n_top: int) -> pd.DataFrame:
    """Return the top ranked SVP genes from the canonical SV table."""
    svp = pd.read_csv(path).sort_values(
        ["svp_rank", "pvalue_adj", "statistic"], ascending=[True, True, False]
    )
    top = svp.head(n_top).copy()
    top["svp_rank"] = top["svp_rank"].astype(int)
    return top


def build_windows(
    features: pd.DataFrame, probes: pd.DataFrame, locus: pd.Series
) -> list[dict[str, object]]:
    """Compress transcript/probe loci into display windows around relevant exons."""
    intervals: list[tuple[int, int]] = []
    # Start with all exon/probe intervals, then merge nearby blocks for a compact
    # transcript panel that still preserves distant intron breaks.
    exon_rows = features[features["feature"].eq("exon")]
    for _, row in exon_rows.iterrows():
        intervals.append((int(row["start_0based"]) - 100, int(row["end_0based"]) + 100))
    for _, row in probes[probes["mapping_status"].astype(str).str.startswith("mapped")].iterrows():
        for _chrom, start0, end0 in parse_segments(row):
            intervals.append((int(start0) - 180, int(end0) + 180))
    if not intervals:
        intervals.append((int(locus["start_1based"]) - 1, int(locus["end_1based"])))
    intervals = [(max(0, s), e) for s, e in intervals]
    merged = merge_intervals(intervals, gap=900)
    if len(merged) > 26:
        # Very exon-rich genes get a coarser merge before falling back to the full locus.
        merged = merge_intervals(intervals, gap=5_000)
    if len(merged) > 26:
        merged = [(int(locus["start_1based"]) - 1, int(locus["end_1based"]))]
    windows: list[dict[str, object]] = []
    for start, end in merged:
        if end - start < 120:
            mid = (start + end) // 2
            start, end = max(0, mid - 60), mid + 60
        windows.append({"start": int(start), "end": int(end)})
    return windows


def make_mapper(windows: list[dict[str, object]], gap: int = 340):
    """Create a genomic-to-compressed-coordinate mapping for the panel."""
    cursor = 0
    # Store compressed coordinates directly on each window so drawing helpers can
    # map genomic intervals without recomputing offsets.
    for window in windows:
        width = int(window["end"]) - int(window["start"])
        window["mapped_start"] = cursor
        window["mapped_end"] = cursor + width
        cursor += width + gap

    def mapped_coord(coord: float) -> float | None:
        for window in windows:
            if int(window["start"]) <= coord <= int(window["end"]):
                return float(window["mapped_start"]) + float(coord) - int(window["start"])
        return None

    return mapped_coord


def draw_segmented_line(
    ax, windows: list[dict[str, object]], start0: int, end0: int, y: float, **kwargs
) -> None:
    for window in windows:
        lo = max(int(start0), int(window["start"]))
        hi = min(int(end0), int(window["end"]))
        if hi > lo:
            x0 = float(window["mapped_start"]) + lo - int(window["start"])
            x1 = float(window["mapped_start"]) + hi - int(window["start"])
            ax.plot([x0, x1], [y, y], **kwargs)


def draw_segmented_rect(
    ax,
    windows: list[dict[str, object]],
    start0: int,
    end0: int,
    y: float,
    height: float,
    *,
    min_width: float,
    **kwargs,
) -> list[tuple[float, float]]:
    """Draw an interval on compressed windows and return its displayed x-bounds."""
    bounds: list[tuple[float, float]] = []
    for window in windows:
        lo = max(int(start0), int(window["start"]))
        hi = min(int(end0), int(window["end"]))
        if hi > lo:
            x = float(window["mapped_start"]) + lo - int(window["start"])
            width = float(hi - lo)
            if width < min_width:
                x -= (min_width - width) / 2
                width = min_width
            ax.add_patch(Rectangle((x, y - height / 2), width, height, **kwargs))
            bounds.append((x, x + width))
    return bounds


def choose_display_transcripts(compat: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Pick a compact, representative transcript set for the inspection panel."""
    if compat.empty:
        return compat
    df = compat[compat["annotation_source"].eq(source_label)].copy()
    if df.empty:
        return df
    df["has_probe_overlap"] = df["n_compatible_codewords"].gt(0)
    if source_label.startswith("GENCODE"):
        # Prefer interpretable reference transcripts: probe-overlapping,
        # protein-coding, and MANE/APPRIS-tagged when available.
        df["is_protein_coding"] = df["transcript_type"].astype(str).eq("protein_coding")
        df["has_mane_or_appris"] = (
            df["tag"].astype(str).str.contains("MANE_Select|appris_principal", regex=True, na=False)
        )
        df["sort_len"] = df["end_1based"].astype(int) - df["start_1based"].astype(int)
        sort_cols = [
            "has_probe_overlap",
            "is_protein_coding",
            "has_mane_or_appris",
            "sort_len",
            "transcript_id",
        ]
        ascending = [False, False, False, False, True]
    else:
        # For PacBio, prioritize expressed/coding isoforms before trimming to a
        # readable number per compatibility group.
        for col in ["sqanti_FL_counts", "sqanti_FL_samples", "sqanti_ORF_length"]:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["is_coding"] = df.get("sqanti_coding", "").astype(str).eq("coding")
        sort_cols = [
            "has_probe_overlap",
            "is_coding",
            "sqanti_FL_counts",
            "sqanti_FL_samples",
            "sqanti_ORF_length",
            "transcript_id",
        ]
        ascending = [False, False, False, False, False, True]

    display_parts = []
    for _, group in df.groupby("compatibility_group", sort=False):
        display_parts.append(
            group.sort_values(sort_cols, ascending=ascending).head(MAX_DISPLAY_PER_GROUP)
        )
    display = pd.concat(display_parts, ignore_index=True) if display_parts else pd.DataFrame()
    display = display.sort_values(
        ["has_probe_overlap", "compatibility_group"] + sort_cols[1:],
        ascending=[False, True] + ascending[1:],
    )
    return display.head(MAX_DISPLAY_PER_SOURCE).copy()


def plot_spatial_value(ax, coords: np.ndarray, values: np.ndarray, title: str) -> None:
    """Rasterize dense spatial points while preserving text and axes as vector."""
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c="#ECECEC",
        s=0.045,
        linewidths=0,
        alpha=0.23,
        rasterized=True,
    )
    valid = np.isfinite(values)
    positive = valid & (values > 0)
    draw = positive if positive.any() else valid
    if draw.any():
        # Robust upper scaling keeps one bright outlier from washing out the map.
        vmax = float(np.nanpercentile(values[draw], 99.4))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.nanmax(values[draw])) if draw.any() else 1.0
        ax.scatter(
            coords[draw, 0],
            coords[draw, 1],
            c=values[draw],
            s=0.16,
            linewidths=0,
            alpha=0.92,
            cmap="Purples",
            vmin=0.0,
            vmax=vmax,
            rasterized=True,
        )
        ax.text(
            0.02,
            0.02,
            f"p99.4={vmax:.2g}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5,
            color="#555555",
        )
    ax.set_title(title, fontsize=6)


def plot_transcript_panel(  # noqa: C901
    ax,
    gene: str,
    locus: pd.Series,
    features: pd.DataFrame,
    probes: pd.DataFrame,
    compat: pd.DataFrame,
) -> None:
    """Draw transcript structures and cached Atera probe compatibility."""
    source_colors = {"GENCODE v49": "#4C78A8", "PacBio QC pass": "#F58518"}
    codeword_colors = plt.get_cmap("tab10")

    display_rows = []
    for source in ["GENCODE v49", "PacBio QC pass"]:
        chosen = choose_display_transcripts(compat, source)
        if not chosen.empty:
            display_rows.append(chosen)
    display = pd.concat(display_rows, ignore_index=True) if display_rows else pd.DataFrame()

    windows = build_windows(features, probes, locus)
    mapped_coord = make_mapper(windows)
    for i, window in enumerate(windows):
        ax.axvspan(
            float(window["mapped_start"]),
            float(window["mapped_end"]),
            color="#F7F8FA" if i % 2 == 0 else "#FFFFFF",
            zorder=0,
        )
        if i < len(windows) - 1:
            x = float(window["mapped_end"]) + 170
            ax.text(x, -0.25, "//", ha="center", va="center", fontsize=6, color="#777777")

    y = 0.0
    y_rows: list[tuple[float, pd.Series]] = []
    last_source = None
    last_group = None
    if not display.empty:
        display = display.sort_values(
            [
                "annotation_source",
                "n_compatible_codewords",
                "compatibility_group",
                "transcript_id",
            ],
            ascending=[True, False, True, True],
        )
        for _, row in display.iterrows():
            source = str(row["annotation_source"])
            group = str(row["compatibility_group"])
            if source != last_source:
                y += 0.72
                ax.text(
                    -120,
                    y,
                    source,
                    ha="right",
                    va="center",
                    fontsize=5.5,
                    fontweight="bold",
                    color=source_colors.get(source, "#333333"),
                )
                last_group = None
            if group != last_group:
                y += 0.42
                group_n = int(row.get("n_transcripts_in_compatibility_group", 1))
                group_label = group.replace("+", " + ")
                if group == "no_probe_overlap":
                    group_label = "no mapped probe"
                ax.text(
                    -120,
                    y,
                    f"{group_label} (n={group_n})",
                    ha="right",
                    va="center",
                    fontsize=5,
                    color="#555555",
                )
            y += 0.34
            y_rows.append((y, row))
            last_source = source
            last_group = group

    for y, row in y_rows:
        txid = str(row["transcript_id"])
        tx_features = features[features["transcript_id"].astype(str).eq(txid)]
        color = source_colors.get(str(row["annotation_source"]), "#999999")
        draw_segmented_line(
            ax,
            windows,
            int(row["start_1based"]) - 1,
            int(row["end_1based"]),
            y,
            color="#999999",
            linewidth=0.55,
            zorder=1,
        )
        for _, exon in (
            tx_features[tx_features["feature"].eq("exon")].sort_values("start_1based").iterrows()
        ):
            draw_segmented_rect(
                ax,
                windows,
                int(exon["start_0based"]),
                int(exon["end_0based"]),
                y,
                0.18,
                min_width=28,
                facecolor=color,
                edgecolor="#333333",
                linewidth=0.35,
                alpha=0.88,
                zorder=3,
            )
        label = str(row["transcript_name"] or txid)
        if label == "nan":
            label = txid
        if str(row["annotation_source"]).startswith("PacBio"):
            fl = row.get("sqanti_FL_counts", np.nan)
            coding = row.get("sqanti_coding", "")
            label = f"{txid} FL={int(float(fl))} {coding}" if pd.notna(fl) else txid
        ax.text(-120, y, label, ha="right", va="center", fontsize=5, color="#333333")

    probe_y = y + 0.62
    codeword_order = sorted(probes["codeword"].dropna().astype(int).unique().tolist())
    codeword_color_map = {
        codeword: codeword_colors(i % 10) for i, codeword in enumerate(codeword_order)
    }
    for _, probe in probes.iterrows():
        if not str(probe["mapping_status"]).startswith("mapped"):
            continue
        codeword = int(probe["codeword"])
        is_approx = int(float(probe.get("approx_mismatches", 0))) > 0
        y_offset = 0.13 * ((int(probe.get("approx_tie_index", 1)) - 1) % 3)
        segments = parse_segments(probe)
        visible_bounds: list[tuple[float, float]] = []
        for _chrom, start0, end0 in segments:
            visible_bounds.extend(
                draw_segmented_rect(
                    ax,
                    windows,
                    int(start0),
                    int(end0),
                    probe_y + y_offset,
                    0.15,
                    min_width=36,
                    facecolor=codeword_color_map.get(codeword, "#666666"),
                    edgecolor="#222222",
                    linewidth=0.45,
                    hatch="///" if is_approx else None,
                    alpha=0.95,
                    zorder=6,
                )
            )
        visible_bounds = sorted(visible_bounds)
        if len(visible_bounds) > 1:
            for (_left0, right0), (left1, _right1) in zip(
                visible_bounds[:-1], visible_bounds[1:], strict=False
            ):
                ax.plot(
                    [right0, left1],
                    [probe_y + y_offset, probe_y + y_offset],
                    color=codeword_color_map.get(codeword, "#666666"),
                    linewidth=0.4,
                    linestyle=(0, (1.2, 1.2)),
                    alpha=0.75,
                    zorder=5.5,
                )
        if visible_bounds:
            x = (
                min(left for left, _right in visible_bounds)
                + max(right for _left, right in visible_bounds)
            ) / 2
        else:
            x = mapped_coord(float(segments[0][1] if segments else probe["start_0based"]))
        if x is not None:
            label = str(probe["probe_oligo_label"])
            if int(probe.get("approx_tie_count", 1)) > 1:
                label += f" {int(probe['approx_tie_index'])}/{int(probe['approx_tie_count'])}"
            if is_approx:
                label += f" ({int(float(probe['approx_mismatches']))}mm)"
            ax.text(
                x,
                probe_y + y_offset + 0.17,
                label,
                ha="center",
                va="bottom",
                fontsize=5,
                rotation=45,
                color=codeword_color_map.get(codeword, "#333333"),
            )
    ax.text(
        -120,
        probe_y,
        "Atera probes",
        ha="right",
        va="center",
        fontsize=5.5,
        fontweight="bold",
    )

    if probes["mapping_status"].astype(str).str.startswith("mapped").any():
        span_start = int(
            probes.loc[
                probes["mapping_status"].astype(str).str.startswith("mapped"),
                "start_1based",
            ].min()
        )
        span_end = int(
            probes.loc[
                probes["mapping_status"].astype(str).str.startswith("mapped"),
                "end_1based",
            ].max()
        )
        probe_span = f"mapped probe span: {locus['chrom']}:{span_start:,}-{span_end:,}"
    else:
        probe_span = "no mapped probes"
    ax.set_title(
        f"{gene} probe compatibility with GENCODE v49 and PacBio QC-pass transcripts\n{probe_span}",
        loc="left",
        fontsize=6.5,
        fontweight="bold",
        pad=8,
    )
    ax.set_yticks([])
    if windows:
        ax.set_xlim(-780, float(windows[-1]["mapped_end"]) + 180)
    ax.set_ylim(-0.55, probe_y + 0.75)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.45)

    tick_positions = []
    tick_labels = []
    for window in windows:
        mid = (int(window["start"]) + int(window["end"])) // 2
        mapped = mapped_coord(mid)
        if mapped is not None:
            tick_positions.append(mapped)
            tick_labels.append(f"{mid + 1:,}")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.set_xlabel(f"hg38 {locus['chrom']} compressed genomic coordinate")
    ax.get_xaxis().get_offset_text().set_visible(False)

    handles = [
        Line2D([0], [0], color=source_colors["GENCODE v49"], lw=3, label="GENCODE v49"),
        Line2D(
            [0],
            [0],
            color=source_colors["PacBio QC pass"],
            lw=3,
            label="PacBio QC pass",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#999999",
            edgecolor="#222222",
            hatch="///",
            label="best local mismatch",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=3)


def make_gene_figure(
    gene: str,
    rank_row: pd.Series,
    coords: np.ndarray,
    norm_values: pd.DataFrame,
    locus: pd.Series,
    features: pd.DataFrame,
    mappings: pd.DataFrame,
    compat: pd.DataFrame,
    out_png: Path,
) -> None:
    """Render one top-SVP inspection panel."""
    gene_features = [col for col in norm_values.columns if col.startswith(f"{gene}|")]
    n = len(gene_features)
    n_cols = max(1, min(n, 6))
    fig_height = 4.8 if compat.shape[0] <= 35 else 5.4
    fig = plt.figure(figsize=(7.28, fig_height), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.1, 2.9],
        left=0.19,
        right=0.992,
        bottom=0.08,
        top=0.90,
        hspace=0.24,
    )
    spatial_gs = gs[0, 0].subgridspec(1, n_cols, wspace=0.04)
    for i, feature in enumerate(gene_features[:n_cols]):
        ax = fig.add_subplot(spatial_gs[0, i])
        codeword = feature.split("|")[-1]
        plot_spatial_value(
            ax,
            coords,
            norm_values[feature].to_numpy(dtype=float),
            f"{codeword}\nlog norm.",
        )
    ax_tx = fig.add_subplot(gs[1, 0])
    plot_transcript_panel(ax_tx, gene, locus, features, mappings, compat)
    title = (
        f"SVP rank {int(rank_row['svp_rank'])}: {gene} "
        f"HSIC-IR stat={float(rank_row['statistic']):.3g}, FDR={float(rank_row['pvalue_adj']):.2g}"
    )
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=7, fontweight="bold")
    fig.savefig(out_png, dpi=450)
    plt.close(fig)


def build_normalized_values(
    sdata, top_genes: list[str], probe_info: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """Normalize only the codeword columns required for the selected panels."""
    adata = sdata.tables[TABLE_CODEWORD]
    coords = np.asarray(adata.obsm["spatial"])
    all_features = probe_info["feature"].drop_duplicates().tolist()
    available = [feature for feature in all_features if feature in adata.var_names.astype(str)]
    missing = sorted(set(all_features) - set(available))
    if missing:
        raise ValueError(
            "Requested codeword features are absent from table_codeword: " + ", ".join(missing[:12])
        )

    counts_layer = adata.layers[LAYER]
    library = np.asarray(counts_layer.sum(axis=1)).ravel().astype(np.float64)
    library[library <= 0] = np.nan
    selected = selected_to_dense(adata[:, available].layers[LAYER]).astype(np.float32, copy=False)
    norm = np.log1p(selected / library[:, None] * NORM_SCALE)
    norm_df = pd.DataFrame(norm, index=adata.obs_names.astype(str), columns=available)

    ordered_cols = []
    for gene in top_genes:
        cols = (
            probe_info.loc[probe_info["target_name"].eq(gene), "feature"]
            .drop_duplicates()
            .loc[lambda s: s.isin(norm_df.columns)]
            .tolist()
        )
        ordered_cols.extend(cols)
    return coords, norm_df[ordered_cols]


def _inventory_complete(inventory: pd.DataFrame) -> bool:
    """Return true when every cached top-20 inventory row has a rendered PNG."""
    if inventory.empty or "figure" not in inventory:
        return False
    return inventory["figure"].map(lambda path: Path(str(path)).exists()).all()


def _inventory_matches_request(inventory: pd.DataFrame, top: pd.DataFrame) -> bool:
    """Return whether a cached inventory matches the currently requested genes."""
    required = {"gene", "svp_rank"}
    if (
        inventory.empty
        or not required.issubset(inventory.columns)
        or not required.issubset(top.columns)
    ):
        return False
    observed = inventory[["gene", "svp_rank"]].copy().reset_index(drop=True)
    expected = top[["gene", "svp_rank"]].copy().reset_index(drop=True)
    observed["gene"] = observed["gene"].astype(str)
    expected["gene"] = expected["gene"].astype(str)
    observed["svp_rank"] = pd.to_numeric(observed["svp_rank"], errors="coerce").astype("Int64")
    expected["svp_rank"] = pd.to_numeric(expected["svp_rank"], errors="coerce").astype("Int64")
    return observed.equals(expected)


def _required_svp_probe_cache_paths(cache_dir: Path) -> dict[str, Path]:
    """List workflow-owned all-SVP mapping tables consumed by this renderer."""
    return {
        "mapping": cache_dir / "svp_probe_mappings.csv",
        "gencode_compatibility": cache_dir / "svp_gencode_transcript_probe_compatibility.csv",
        "pacbio_compatibility": cache_dir / "svp_pacbio_transcript_probe_compatibility.csv",
    }


def _load_svp_probe_cache(
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all-SVP mapping caches and fail clearly when workflow has not run."""
    paths = _required_svp_probe_cache_paths(cache_dir)
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {project_path(path)}" for path in missing)
        raise FileNotFoundError(
            "Missing all-SVP probe mapping cache. Run "
            "`python scripts/atera_ffpe_human_brca/workflow.py --n-jobs 8` first.\n"
            f"{missing_text}"
        )
    return (
        pd.read_csv(paths["mapping"]),
        pd.read_csv(paths["gencode_compatibility"]),
        pd.read_csv(paths["pacbio_compatibility"]),
    )


def prepare_top20_panel_inputs(
    *,
    force: bool = False,
    n_top: int = N_TOP,
    genes: list[str] | None = None,
    svp_csv: Path = SVP_CSV,
    out_dir: Path = OUT_DIR,
    fig_dir: Path = FIG_DIR,
    cache_dir: Path = SVP_PROBE_RESULT_DIR,
    render_figures: bool = False,
) -> pd.DataFrame:
    """Prepare top-SVP panel summaries and optionally render per-gene PNGs."""
    global SVP_CSV, OUT_DIR, FIG_DIR
    SVP_CSV = svp_csv
    OUT_DIR = out_dir
    FIG_DIR = fig_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    inventory_path = OUT_DIR / "top20_svp_panel_inventory.tsv"

    top = read_svp_top(SVP_CSV, n_top)
    if genes:
        requested = list(dict.fromkeys(genes))
        top = pd.read_csv(SVP_CSV)
        top = top[top["gene"].isin(requested)].copy()
        top["gene"] = pd.Categorical(top["gene"], categories=requested, ordered=True)
        top = top.sort_values("gene").reset_index(drop=True)
    if top.empty:
        raise ValueError("No SVP genes were selected for top-panel rendering")
    genes = top["gene"].astype(str).tolist()
    if inventory_path.exists() and not force:
        inventory = pd.read_csv(inventory_path, sep="\t")
        if _inventory_matches_request(inventory, top) and (
            not render_figures or _inventory_complete(inventory)
        ):
            log(f"top SVP panel inventory matches request: {project_path(inventory_path)}")
            return inventory
    log(f"rendering-ready top SVP genes from cached mappings: {', '.join(genes)}")

    mappings_all, gencode_compat_all, pacbio_compat_all = _load_svp_probe_cache(cache_dir)
    # All-SVP mapping is workflow-owned; this renderer only subsets those caches.
    mappings = mappings_all[mappings_all["target_name"].astype(str).isin(genes)].copy()
    compat_all = pd.concat(
        [
            gencode_compat_all[gencode_compat_all["gene"].astype(str).isin(genes)],
            pacbio_compat_all[pacbio_compat_all["gene"].astype(str).isin(genes)],
        ],
        ignore_index=True,
    )

    probe_info = read_probe_info(PROBE_CSV, genes)
    gencode = read_gencode_subset(GENCODE_GTF, genes)
    pacbio = read_pacbio_subset(PACBIO_GFF, genes)
    gencode_tx = gencode[gencode["feature"].eq("transcript")].copy()
    pacbio_tx = pacbio[pacbio["feature"].eq("transcript")].copy()
    loci = gene_loci(gencode)
    missing_loci = sorted(set(genes) - set(loci["gene_name"].astype(str)))
    if missing_loci:
        raise ValueError(f"Missing GENCODE gene loci for: {missing_loci}")

    # Spatial expression summaries are lightweight and specific to the chosen
    # display genes, so they stay in this renderer instead of the main workflow.
    log(f"reading {project_path(SDATA_ZARR)}")
    sdata = sd.read_zarr(SDATA_ZARR)
    coords, norm_df = build_normalized_values(sdata, genes, probe_info)

    top.to_csv(OUT_DIR / "top20_svp_genes.csv", index=False)
    norm_summary_path = OUT_DIR / "top20_svp_probe_log_normalized_expression_summary.csv"
    summary_rows: list[dict[str, object]] = []
    for feature in norm_df.columns:
        values = norm_df[feature].to_numpy(dtype=float)
        gene, codeword = feature.split("|", 1)
        positive = values > 0
        summary_rows.append(
            {
                "gene": gene,
                "feature": feature,
                "codeword": int(codeword),
                "n_cells": int(values.shape[0]),
                "n_positive_cells": int(positive.sum()),
                "pct_positive_cells": float(positive.mean()),
                "mean_log_norm": float(np.nanmean(values)),
                "median_log_norm": float(np.nanmedian(values)),
                "p95_log_norm_positive": (
                    float(np.nanpercentile(values[positive], 95)) if positive.any() else 0.0
                ),
                "p99_4_log_norm_positive": (
                    float(np.nanpercentile(values[positive], 99.4)) if positive.any() else 0.0
                ),
            }
        )
    pd.DataFrame(summary_rows).to_csv(norm_summary_path, index=False)

    inventory_rows: list[dict[str, object]] = []
    locus_by_gene = loci.set_index("gene_name")
    for _, rank_row in top.iterrows():
        gene = str(rank_row["gene"])
        gene_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", gene)
        out_png = (
            FIG_DIR
            / f"top20_svp_{int(rank_row['svp_rank']):02d}_{gene_slug}_probe_transcript_panel.png"
        )
        # Assemble per-gene slices for both the optional PNG and the inventory row.
        gene_features = pd.concat(
            [
                gencode[gencode["gene_name"].astype(str).eq(gene)],
                pacbio[pacbio["gene_name"].astype(str).eq(gene)],
            ],
            ignore_index=True,
        )
        gene_mappings = mappings[mappings["target_name"].astype(str).eq(gene)].copy()
        gene_compat = compat_all[compat_all["gene"].astype(str).eq(gene)].copy()
        if render_figures:
            make_gene_figure(
                gene,
                rank_row,
                coords,
                norm_df,
                locus_by_gene.loc[gene],
                gene_features,
                gene_mappings,
                gene_compat,
                out_png,
            )
        inventory_rows.append(
            {
                "svp_rank": int(rank_row["svp_rank"]),
                "gene": gene,
                "figure": str(out_png),
                "n_codeword_features": int(
                    probe_info.loc[probe_info["target_name"].eq(gene), "feature"].nunique()
                ),
                "n_probe_oligos": int(probe_info["target_name"].eq(gene).sum()),
                "n_mapped_probe_oligos": int(
                    gene_mappings.loc[
                        gene_mappings["mapping_status"].astype(str).str.startswith("mapped"),
                        "probe_instance_id",
                    ].nunique()
                ),
                "n_mapping_rows_including_ties": int(
                    gene_mappings["mapping_status"].astype(str).str.startswith("mapped").sum()
                ),
                "n_gencode_transcripts": int(gencode_tx["gene_name"].astype(str).eq(gene).sum()),
                "n_pacbio_transcripts": int(pacbio_tx["gene_name"].astype(str).eq(gene).sum()),
                "n_compatibility_groups": (
                    int(gene_compat["compatibility_group"].nunique())
                    if not gene_compat.empty
                    else 0
                ),
            }
        )
        if render_figures:
            log(f"wrote {project_path(out_png)}")

    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(inventory_path, sep="\t", index=False)
    log(f"wrote {project_path(inventory_path)}")
    return inventory


def render_top20_figures(
    *,
    force: bool = False,
    n_top: int = N_TOP,
    genes: list[str] | None = None,
    svp_csv: Path = SVP_CSV,
    out_dir: Path = OUT_DIR,
    fig_dir: Path = FIG_DIR,
    cache_dir: Path = SVP_PROBE_RESULT_DIR,
) -> pd.DataFrame:
    """Render top-SVP per-gene inspection PNGs from workflow-owned caches."""
    return prepare_top20_panel_inputs(
        force=force,
        n_top=n_top,
        genes=genes,
        svp_csv=svp_csv,
        out_dir=out_dir,
        fig_dir=fig_dir,
        cache_dir=cache_dir,
        render_figures=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render top-SVP probe/transcript inspection panels from cached all-SVP probe mappings."
    )
    parser.add_argument("--n-top", type=int, default=N_TOP)
    parser.add_argument(
        "--genes",
        nargs="*",
        default=None,
        help="Optional explicit gene list replacing the top-N SVP list.",
    )
    parser.add_argument("--svp-csv", type=Path, default=SVP_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--cache-dir", type=Path, default=SVP_PROBE_RESULT_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute renderer-side summaries and figures.",
    )
    parser.add_argument(
        "--render", action="store_true", help="Also render per-gene inspection PNGs."
    )
    args = parser.parse_args()
    prepare_top20_panel_inputs(
        force=args.force,
        n_top=args.n_top,
        genes=args.genes,
        svp_csv=args.svp_csv,
        out_dir=args.out_dir,
        fig_dir=args.fig_dir,
        cache_dir=args.cache_dir,
        render_figures=args.render,
    )


if __name__ == "__main__":
    main()
