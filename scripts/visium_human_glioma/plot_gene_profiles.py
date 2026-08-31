"""Standalone glioma gene-profile panels from cached IDH-status spectra."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.collections as mcollections
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import sparse

from scripts.paper_utils import (
    FIGURE_DPI,
    FONT_CONFIG,
    format_pvalue,
    manuscript_size,
    parse_gene_list,
)
from scripts.paper_utils import (
    configure_paper_matplotlib as _configure_paper_matplotlib,
)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.visium_human_glioma"

from .config import (  # noqa: E402
    FIGURE_DIR,
    GROUP_COLORS,
    GROUP_LABELS,
    GROUP_ORDER,
    REAL_TESTS_FILE,
    SPECTRA_FILE,
)
from .utils import (  # noqa: E402
    load_normalized_samples_from_metadata,
    project_relative,
)

FONT_MIN = FONT_CONFIG["text"]
FONT_TICK = FONT_CONFIG["tick"]
FONT_LABEL = FONT_CONFIG["label"]
FONT_TITLE = FONT_CONFIG["title"]
EPS = 1e-12
VISIUM_SPOT_DIAMETER_UM = 55.0
SPATIAL_SPOT_RADIUS_SCALE = 1.45
REPRESENTATIVE_SAMPLES_PER_GROUP = 2


def configure_matplotlib() -> None:
    """Apply the compact manuscript-scale Matplotlib style."""
    _configure_paper_matplotlib(
        fonts=FONT_CONFIG,
        overrides={
            "pdf.compression": 9,
            "figure.constrained_layout.use": False,
        },
    )


def short_sample_id(sample_id: str) -> str:
    """Return the compact sample label used in figure panels."""
    return sample_id.replace("IDHm_", "").replace("GBM_", "")


def load_spectra_cache(path: Path = SPECTRA_FILE) -> dict[str, Any]:
    """Load radial spectra plus sample metadata from the workflow cache."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run workflow.py first.")
    with np.load(path, allow_pickle=True) as data:
        required = {
            "spectra",
            "freq_edges",
            "sample_ids",
            "groups",
            "h5ad_paths",
            "genes",
        }
        missing = required - set(data.files)
        if missing:
            raise FileNotFoundError(
                "Spectra cache uses an obsolete schema; rerun workflow.py. "
                f"Missing keys: {sorted(missing)}"
            )
        return {
            "spectra": np.asarray(data["spectra"], dtype=float),
            "freq_edges": np.asarray(data["freq_edges"], dtype=float),
            "sample_ids": data["sample_ids"].astype(str).tolist(),
            "groups": data["groups"].astype(str).tolist(),
            "h5ad_paths": data["h5ad_paths"].astype(str).tolist(),
            "genes": data["genes"].astype(str).tolist(),
        }


def metadata_from_spectra(spectra_data: dict[str, Any]) -> pd.DataFrame:
    """Build sample metadata from the spectra cache."""
    required = {"sample_ids", "groups", "h5ad_paths"}
    missing = required - set(spectra_data)
    if missing:
        raise FileNotFoundError(
            "Missing sample metadata in spectra cache; rerun workflow.py. "
            f"Missing keys: {sorted(missing)}"
        )
    return pd.DataFrame(
        {
            "sample_id": list(map(str, spectra_data["sample_ids"])),
            "group": list(map(str, spectra_data["groups"])),
            "h5ad_path": list(map(str, spectra_data["h5ad_paths"])),
        }
    )


def gene_values(adata: Any, gene: str) -> np.ndarray:
    """Return log-normalized expression values for one gene."""
    if gene not in adata.var_names:
        return np.zeros(adata.n_obs, dtype=float)
    x = adata[:, gene].X
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=float).ravel()


def plot_glioma_spatial(
    ax: Any,
    adata: Any,
    gene: str,
    title: str | None = None,
    *,
    vmin: float = 0.0,
    vmax: float | None = None,
    spot_radius_scale: float = SPATIAL_SPOT_RADIUS_SCALE,
    view_span_um: float | None = None,
) -> Any:
    """Draw one glioma Visium map as physical-radius filled spots."""
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    vals = gene_values(adata, gene)
    radius = (VISIUM_SPOT_DIAMETER_UM / 2.0) * spot_radius_scale
    spots = [mpatches.Circle((x, y), radius=radius) for x, y in coords]
    sc = mcollections.PatchCollection(
        spots,
        array=vals,
        cmap="magma",
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        linewidths=0,
        edgecolors="none",
        antialiased=True,
        rasterized=True,
    )
    ax.add_collection(sc)
    if view_span_um is None:
        xlim = (coords[:, 0].min() - radius, coords[:, 0].max() + radius)
        ylim = (coords[:, 1].max() + radius, coords[:, 1].min() - radius)
    else:
        span = max(
            float(view_span_um),
            float(np.ptp(coords[:, 0])) + 2.0 * radius,
            float(np.ptp(coords[:, 1])) + 2.0 * radius,
        )
        half = 0.5 * span
        center_x = 0.5 * (coords[:, 0].min() + coords[:, 0].max())
        center_y = 0.5 * (coords[:, 1].min() + coords[:, 1].max())
        xlim = (center_x - half, center_x + half)
        ylim = (center_y + half, center_y - half)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=FONT_MIN, linespacing=0.9, pad=1.0, y=1.01)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return sc


def _shared_spatial_span_um(samples: Sequence[Any]) -> float:
    """Return one square view span so selected maps use identical spot size."""
    radius = (VISIUM_SPOT_DIAMETER_UM / 2.0) * SPATIAL_SPOT_RADIUS_SCALE
    spans = []
    for adata in samples:
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
        spans.extend(
            [
                float(np.ptp(coords[:, 0])) + 2.0 * radius,
                float(np.ptp(coords[:, 1])) + 2.0 * radius,
            ]
        )
    return max(spans) * 1.02


def _gene_vmax(samples: Sequence[Any], gene: str) -> float:
    vals = np.concatenate([gene_values(a, gene) for a in samples])
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 1.0
    return max(float(np.quantile(finite, 0.99)), 1e-6)


def _freq_axis(freq_edges: np.ndarray, spectra: np.ndarray) -> tuple[np.ndarray, str]:
    k = spectra.shape[-1]
    if freq_edges.size != k + 1:
        raise ValueError(
            f"Expected {k + 1} frequency edges for {k} spectral bins; "
            f"found {freq_edges.size}. Rerun workflow.py."
        )
    return 0.5 * (freq_edges[:-1] + freq_edges[1:]), "Spatial frequency (cycles / um)"


def add_length_scale_axis(
    ax: Any,
    ticks: Sequence[float] = (1 / 1000, 1 / 500),
    labels: Sequence[str] = ("1 mm", "500 um"),
) -> Any:
    """Add physical length scales above a radial frequency plot."""
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(ticks)
    tick_labels = sec.set_xticklabels(labels, fontsize=FONT_MIN)
    if len(tick_labels) >= 2:
        tick_labels[0].set_horizontalalignment("right")
        tick_labels[-1].set_horizontalalignment("left")
    sec.tick_params(width=0.5, length=2, pad=0.5)
    for tick in ticks:
        ax.axvline(tick, color="0.72", lw=0.5, ls=":", zorder=0)
    return sec


class GliomaGeneProfilePlotter:
    """Load cached glioma analysis data and draw gene-profile panels."""

    def __init__(
        self,
        genes: Sequence[str],
        *,
        samples: dict[str, Any] | None = None,
        meta: pd.DataFrame | None = None,
        spectra_data: dict[str, Any] | None = None,
        real_tests: pd.DataFrame | None = None,
    ) -> None:
        """Reuse cached spectra and optional in-memory samples for selected genes."""
        selected_genes = parse_gene_list(genes)
        if not selected_genes:
            raise ValueError("No genes selected.")
        self.spectra_data = load_spectra_cache() if spectra_data is None else spectra_data
        tests = real_tests.copy() if real_tests is not None else pd.read_csv(REAL_TESTS_FILE)
        missing = [gene for gene in selected_genes if gene not in set(self.spectra_data["genes"])]
        if missing:
            raise KeyError(f"Genes missing from cached glioma spectra: {missing}")
        missing_tests = [
            gene for gene in selected_genes if gene not in set(tests["Feature"].astype(str))
        ]
        if missing_tests:
            raise KeyError(f"Genes missing from cached glioma tests: {missing_tests}")

        if (samples is None) != (meta is None):
            raise ValueError("samples and meta must be provided together")
        if samples is None:
            meta = metadata_from_spectra(self.spectra_data)
            samples, meta = load_normalized_samples_from_metadata(meta)
        self.samples = samples
        self.meta = meta.copy()
        self.real_by_gene = tests.set_index("Feature")

    def representative_sample_ids(
        self,
        gene: str,
        *,
        n_per_group: int = REPRESENTATIVE_SAMPLES_PER_GROUP,
    ) -> list[str]:
        """Choose median-like expression samples per IDH-status group."""
        out: list[str] = []
        for group in GROUP_ORDER:
            ids = self.meta.loc[self.meta["group"] == group, "sample_id"].astype(str).tolist()
            if not ids:
                raise KeyError(f"No samples found for group {group}")
            means = np.asarray([np.nanmean(gene_values(self.samples[sid], gene)) for sid in ids])
            positive = means[means > 0]
            target = float(np.median(positive)) if positive.size else float(np.median(means))
            order = np.lexsort((np.asarray(ids, dtype=object), np.abs(means - target)))
            out.extend(ids[i] for i in order[:n_per_group])
        return out

    def validate_sample_ids(self, sample_ids: Sequence[str] | None, gene: str) -> list[str]:
        """Return caller-selected sample IDs or per-group representatives."""
        out = list(sample_ids) if sample_ids else self.representative_sample_ids(gene)
        missing = [sid for sid in out if sid not in self.samples]
        if missing:
            raise KeyError(f"Sample IDs missing from cached glioma samples: {missing}")
        return out

    def plot_spatial(
        self,
        ax: Any,
        gene: str,
        sample_id: str,
        *,
        vmax: float,
        view_span_um: float | None = None,
    ) -> Any:
        """Draw one spatial expression heatmap."""
        adata = self.samples[sample_id]
        group = self.meta.set_index("sample_id").loc[sample_id, "group"]
        sample_label = short_sample_id(sample_id).replace("bulk", "")
        return plot_glioma_spatial(
            ax,
            adata,
            gene,
            f"{group}\n{sample_label}",
            vmax=vmax,
            view_span_um=view_span_um,
        )

    def plot_profiles(
        self,
        raw_ax: Any,
        l1_ax: Any,
        gene: str,
        *,
        show_legend: bool = True,
    ) -> None:
        """Plot raw spectra and the L1-normalized group-mean log2 fold change."""
        gi = self.spectra_data["genes"].index(gene)
        spectra = np.asarray(self.spectra_data["spectra"], dtype=float)
        x, xlabel = _freq_axis(np.asarray(self.spectra_data["freq_edges"], dtype=float), spectra)
        id_to_group = self.meta.set_index("sample_id")["group"].astype(str).to_dict()
        groups = np.asarray(
            [id_to_group[sid] for sid in self.spectra_data["sample_ids"]], dtype=object
        )
        raw_by_group: dict[str, np.ndarray] = {}
        for group in GROUP_ORDER:
            idx = np.flatnonzero(groups == group)
            y = np.maximum(spectra[idx, gi, :], EPS)
            raw_by_group[group] = y
            color = GROUP_COLORS[group]
            for row in y:
                raw_ax.plot(x, row, color=color, alpha=0.18, lw=0.5)
            raw_ax.plot(
                x,
                y.mean(axis=0),
                color=color,
                lw=1.55,
                label=f"{GROUP_LABELS[group]} (n={len(idx)})",
            )
        raw_ax.set_yscale("log")
        finite = np.concatenate(list(raw_by_group.values()))
        finite = finite[np.isfinite(finite) & (finite > 0)]
        if finite.size:
            upper = float(finite.max())
            # Keep rare near-zero profiles from compressing the visible group curves.
            raw_ax.set_ylim(max(float(finite.min()), upper * 1e-6), upper * 1.15)
        raw_ax.set_xlabel(xlabel, labelpad=1)
        raw_ax.set_ylabel("P(k)", labelpad=1)
        raw_ax.set_xlim(x[0], x[-1])
        add_length_scale_axis(raw_ax)
        if show_legend:
            raw_ax.legend(
                frameon=False,
                loc="upper right",
                handlelength=1.0,
                handletextpad=0.35,
                labelspacing=0.2,
                borderaxespad=0.2,
            )

        l1_means = {
            group: (y / np.maximum(y.sum(axis=1, keepdims=True), EPS)).mean(axis=0)
            for group, y in raw_by_group.items()
        }
        lower_group, upper_group = GROUP_ORDER
        log2fc = np.log2(
            np.maximum(l1_means[upper_group], EPS) / np.maximum(l1_means[lower_group], EPS)
        )
        l1_ax.axhline(0, color="0.25", lw=0.55)
        l1_ax.fill_between(
            x,
            0,
            np.maximum(log2fc, 0),
            color=GROUP_COLORS[upper_group],
            alpha=0.35,
            lw=0,
        )
        l1_ax.fill_between(
            x,
            0,
            np.minimum(log2fc, 0),
            color=GROUP_COLORS[lower_group],
            alpha=0.28,
            lw=0,
        )
        l1_ax.plot(x, log2fc, color="0.15", lw=1.05)
        ymax = max(0.25, float(np.nanmax(np.abs(log2fc)))) * 1.18
        l1_ax.set_ylim(-ymax, ymax)
        l1_ax.set_xlim(x[0], x[-1])
        l1_ax.set_xlabel(xlabel, labelpad=1)
        l1_ax.set_ylabel("log2 FC per bin", labelpad=1)
        add_length_scale_axis(l1_ax)
        if show_legend:
            l1_ax.legend(
                handles=[
                    mpatches.Patch(
                        facecolor=GROUP_COLORS[upper_group],
                        alpha=0.45,
                        edgecolor="none",
                        label=f"Higher in {GROUP_LABELS[upper_group]}",
                    ),
                    mpatches.Patch(
                        facecolor=GROUP_COLORS[lower_group],
                        alpha=0.45,
                        edgecolor="none",
                        label=f"Higher in {GROUP_LABELS[lower_group]}",
                    ),
                ],
                frameon=False,
                loc="upper right",
                handlelength=1.0,
                handletextpad=0.35,
                labelspacing=0.2,
                borderaxespad=0.2,
            )

        for ax in (raw_ax, l1_ax):
            ax.grid(color="0.90", lw=0.5, alpha=0.7)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    def gene_pvalue_label(self, gene: str) -> str:
        """Return compact DE/DF/DF-norm p-value text for one gene."""
        row = self.real_by_gene.loc[gene]
        return (
            f"p-values: DE={format_pvalue(row['P_DE'])} | "
            f"DF={format_pvalue(row['P_DF'])} | "
            f"DF-norm={format_pvalue(row['P_DF_norm'])}"
        )


def draw_gene_profile_panel(
    fig: Any,
    spec: Any,
    *,
    plotter: GliomaGeneProfilePlotter,
    gene: str,
    sample_ids: Sequence[str] | None = None,
    narrative_label: str | None = None,
    show_legend: bool = True,
) -> None:
    """Draw spatial maps above raw and L1-normalized radial-profile panels."""
    selected = plotter.validate_sample_ids(sample_ids, gene)
    gs = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=spec,
        height_ratios=[0.92, 1.08],
        hspace=0.58,
    )
    map_gs = gridspec.GridSpecFromSubplotSpec(
        1,
        len(selected) + 1,
        subplot_spec=gs[0, 0],
        width_ratios=[1.0] * len(selected) + [0.055],
        wspace=0.055,
    )

    selected_samples = [plotter.samples[sid] for sid in selected]
    vmax = _gene_vmax(selected_samples, gene)
    view_span_um = _shared_spatial_span_um(selected_samples)
    last_sc = None
    for j, sid in enumerate(selected):
        ax = fig.add_subplot(map_gs[0, j])
        last_sc = plotter.plot_spatial(ax, gene, sid, vmax=vmax, view_span_um=view_span_um)
    cax = fig.add_subplot(map_gs[0, len(selected)])
    cbar = fig.colorbar(last_sc, cax=cax)
    cbar.ax.tick_params(labelsize=FONT_MIN, width=0.5, length=1.8, pad=1.2)
    cbar.ax.set_title("Log1p", fontsize=FONT_MIN, pad=0.8)

    profile_gs = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=gs[1, 0],
        wspace=0.34,
    )
    ax_raw = fig.add_subplot(profile_gs[0, 0])
    ax_l1 = fig.add_subplot(profile_gs[0, 1], sharex=ax_raw)
    plotter.plot_profiles(
        ax_raw,
        ax_l1,
        gene,
        show_legend=show_legend,
    )
    label = narrative_label or "spatial expression pattern"
    title_ax = fig.add_subplot(gs[1, 0], frameon=False)
    title_ax.patch.set_visible(False)
    title_ax.set_xticks([])
    title_ax.set_yticks([])
    title_ax.set_zorder(5)
    for spine in title_ax.spines.values():
        spine.set_visible(False)
    title_ax.set_title(
        rf"$\mathit{{{gene}}}$: {label}" + f"\n{plotter.gene_pvalue_label(gene)}",
        fontsize=FONT_TITLE,
        fontstyle="normal",
        pad=13,
        y=1.02,
    )


def make_gene_profile_page(
    genes: Sequence[str],
    *,
    plotter: GliomaGeneProfilePlotter,
    sample_ids: Sequence[str] | None,
    page_title: str | None = None,
) -> Any:
    """Build one optionally titled PDF page with half-page-width gene panels."""
    n_cols = 2
    n_rows = int(math.ceil(len(genes) / n_cols))
    height_mm = max(100.0, 68.0 * n_rows + 18.0)
    fig = plt.figure(figsize=manuscript_size(height_mm))
    outer = gridspec.GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        left=0.055,
        right=0.972,
        top=0.88,
        bottom=0.12,
        hspace=0.42,
        wspace=0.26,
    )
    if page_title:
        fig.suptitle(page_title, fontsize=FONT_TITLE, y=0.965)
    for i, gene in enumerate(genes):
        row, col = divmod(i, n_cols)
        draw_gene_profile_panel(
            fig,
            outer[row, col],
            plotter=plotter,
            gene=gene,
            sample_ids=sample_ids,
            show_legend=True,
        )
    return fig


def generate_pdf(
    genes: Sequence[str],
    *,
    output: Path,
    sample_ids: Sequence[str] | None = None,
    genes_per_page: int = 4,
    page_titles: Sequence[str] | None = None,
) -> None:
    """Generate a multi-page glioma gene-profile PDF for explicit genes."""
    genes = parse_gene_list(genes)
    if not genes:
        raise ValueError("No genes selected.")
    if not 1 <= genes_per_page <= 6:
        raise ValueError("genes_per_page must be between 1 and 6.")
    configure_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter = GliomaGeneProfilePlotter(genes)
    page_starts = list(range(0, len(genes), genes_per_page))
    if page_titles is not None and len(page_titles) != len(page_starts):
        raise ValueError(
            f"Expected {len(page_starts)} page titles for {len(genes)} genes; "
            f"received {len(page_titles)}."
        )
    with PdfPages(output) as pdf:
        for page_index, start in enumerate(page_starts):
            subset = genes[start : start + genes_per_page]
            fig = make_gene_profile_page(
                subset,
                plotter=plotter,
                sample_ids=sample_ids,
                page_title=page_titles[page_index] if page_titles is not None else None,
            )
            pdf.savefig(fig, bbox_inches=None, dpi=FIGURE_DPI)
            plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse standalone plotter CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate glioma gene-profile PDF with spatial maps, raw radial profiles, "
            "and L1-normalized log2 fold-change profiles."
        )
    )
    parser.add_argument("--genes", nargs="+", required=True, help="Selected genes.")
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Optional sample IDs for spatial maps; defaults to two representatives per IDH group.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGURE_DIR / "idh_selected_genes.pdf",
        help="Output multi-page PDF path.",
    )
    parser.add_argument(
        "--genes-per-page",
        type=int,
        default=4,
        help="Number of gene panels per PDF page; maximum 6.",
    )
    parser.add_argument(
        "--page-titles",
        nargs="*",
        default=None,
        help="Optional title for each PDF page, in page order.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    genes = parse_gene_list(args.genes)
    generate_pdf(
        genes,
        output=args.output,
        sample_ids=args.sample_ids,
        genes_per_page=args.genes_per_page,
        page_titles=args.page_titles,
    )
    print(project_relative(args.output))
    print("genes:", ", ".join(genes))


if __name__ == "__main__":
    main()
