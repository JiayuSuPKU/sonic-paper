"""Plot brain-development spatial maps and radial-profile examples."""

from __future__ import annotations

import argparse
import sys
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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.comparator_benchmark"

from .config import (  # noqa: E402
    BRAIN_DEV_REFERENCE_BINNING,
    BRAIN_DEV_REFERENCE_FEATURE_MODE,
    BRAIN_DEV_REFERENCE_INPUT,
    BRAIN_DEV_REFERENCE_N_RADIAL_BINS,
    BRAIN_DEV_SPOT_DIAMETER_UM,
    BRAIN_DEV_TESTS_FILE,
    BRAIN_DEV_TIMEPOINT_ORDER,
    EPS,
    RESULT_DIR,
)
from .plot_common import (  # noqa: E402
    configure_paper_matplotlib,
    gene_values,
)
from .plot_common import (
    gene_vmax as common_gene_vmax,
)
from .utils import (  # noqa: E402
    add_length_scale_axis,
    fit_all_samples,
    format_pvalue,
    l1_normalize_rows,
    load_panel,
    parse_gene_list,
    radial_frequency_axis,
    unique,
)

BRAIN_DEV_TIME_COLORS = dict(
    zip(
        BRAIN_DEV_TIMEPOINT_ORDER,
        mpl.colormaps["viridis"](np.linspace(0.10, 0.88, len(BRAIN_DEV_TIMEPOINT_ORDER))),
        strict=True,
    )
)
SPATIAL_SPOT_RADIUS_SCALE = 1.5


def _load_brain_dev_stats() -> pd.DataFrame:
    """Load wide brain-dev p-values for per-gene profile annotations."""
    tests = pd.read_csv(RESULT_DIR / BRAIN_DEV_TESTS_FILE)
    base = tests[
        tests["panel"].eq("brain_dev")
        & tests["input"].eq(BRAIN_DEV_REFERENCE_INPUT)
        & tests["binning"].eq(BRAIN_DEV_REFERENCE_BINNING)
        & tests["requested_n_radial_bins"].eq(BRAIN_DEV_REFERENCE_N_RADIAL_BINS)
    ].copy()

    # The reference feature mode is 2D; for visualization we also need radial-mode pvalues.
    ref = base[base["feature_mode"].eq(BRAIN_DEV_REFERENCE_FEATURE_MODE)].copy()
    radial = base[base["feature_mode"].eq("radial")].copy()
    if ref.empty:
        raise RuntimeError("brain_dev_tests.csv has no reference brain_dev rows")
    if radial.empty:
        raise RuntimeError("brain_dev_tests.csv has no radial brain_dev rows")

    wide = ref.pivot_table(
        index="Feature", columns="test", values="P_value", aggfunc="first"
    ).rename(columns={col: f"P_{col}" for col in ref["test"].unique()})
    radial_wide = radial.pivot_table(
        index="Feature", columns="test", values="P_value", aggfunc="first"
    ).rename(columns={col: f"P_{col}_radial" for col in radial["test"].unique()})
    return wide.join(radial_wide, how="left")


def representative_brain_dev_paths(
    samples: dict[Path, Any],
    meta: pd.DataFrame,
    *,
    timepoints: tuple[str, ...] = BRAIN_DEV_TIMEPOINT_ORDER,
) -> list[Path]:
    """Pick the lowest-replicate anterior sample for each requested timepoint."""
    available = set(samples)
    sub = meta[meta["plane"].eq("ANTERIOR")].sort_values(["timepoint", "replicate"])
    out: list[Path] = []
    for timepoint in timepoints:
        match = sub[sub["timepoint"].eq(timepoint)]
        paths = [Path(p) for p in match["h5ad_path"] if Path(p) in available]
        if not paths:
            raise KeyError(f"No anterior brain-dev sample found for {timepoint}")
        out.append(paths[0])
    return out


def plot_brain_spatial(
    ax,
    adata: Any,
    gene: str,
    title: str | None = None,
    *,
    vmin: float = 0.0,
    vmax: float | None = None,
    spot_radius_scale: float = SPATIAL_SPOT_RADIUS_SCALE,
):
    """Draw one brain-dev spatial map as high-resolution filled spots."""
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    vals = gene_values(adata, gene)
    radius = (BRAIN_DEV_SPOT_DIAMETER_UM / 2.0) * spot_radius_scale
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
    ax.set_xlim(coords[:, 0].min() - radius, coords[:, 0].max() + radius)
    ax.set_ylim(coords[:, 1].min() - radius, coords[:, 1].max() + radius)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=5.5, pad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return sc


def brain_spectra_by_timepoint(
    spectra: np.ndarray,
    gene_index: int,
    paths_order: list[Path],
    timepoint_by_path: dict[Path, str],
    *,
    eps: float = EPS,
) -> dict[str, np.ndarray]:
    """Return fitted radial spectra for one gene grouped by postnatal timepoint."""
    gene_spectra = np.maximum(np.asarray(spectra[:, gene_index, :], dtype=float), eps)
    grouped: dict[str, list[np.ndarray]] = {}
    for i, path in enumerate(paths_order):
        grouped.setdefault(timepoint_by_path[path], []).append(gene_spectra[i])
    return {timepoint: np.vstack(rows) for timepoint, rows in grouped.items()}


def plot_brain_raw_profiles(
    ax,
    grouped: dict[str, np.ndarray],
    frequency: np.ndarray,
    *,
    show_legend: bool = True,
) -> None:
    """Plot per-sample raw radial power curves plus timepoint means."""
    for timepoint in BRAIN_DEV_TIMEPOINT_ORDER:
        if timepoint not in grouped:
            continue
        arr = grouped[timepoint]
        color = BRAIN_DEV_TIME_COLORS[timepoint]
        for row in arr:
            ax.plot(frequency, row, color=color, alpha=0.24, lw=0.5)
        ax.plot(
            frequency,
            arr.mean(axis=0),
            color=color,
            lw=1.2,
            label=f"{timepoint} (n={arr.shape[0]})",
        )
    ax.set_yscale("log")
    ax.set_ylabel("Radial power", labelpad=1)
    ax.tick_params(labelbottom=False)
    add_length_scale_axis(ax)
    ax.grid(color="0.90", lw=0.5)
    ax.set_xlim(frequency[0], frequency[-1])
    if show_legend:
        ax.legend(
            frameon=False,
            loc="upper right",
            fontsize=5,
            ncol=2,
            handlelength=1.0,
            columnspacing=0.7,
        )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def brain_l1_log2fc_matrix(
    grouped: dict[str, np.ndarray],
    *,
    eps: float = EPS,
) -> tuple[np.ndarray, list[str]]:
    """Return timepoint mean L1-normalized radial-profile log2 changes."""
    all_shape: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for timepoint in BRAIN_DEV_TIMEPOINT_ORDER:
        if timepoint not in grouped:
            continue
        shape = l1_normalize_rows(grouped[timepoint], eps)
        all_shape.append(shape)
        rows.append(shape.mean(axis=0))
        labels.append(timepoint)
    all_ref = np.maximum(np.vstack(all_shape).mean(axis=0), eps)
    mat = np.log2(np.maximum(np.vstack(rows), eps) / all_ref)
    return mat, labels


def plot_brain_l1_log2fc_heatmap(
    ax,
    cax,
    grouped: dict[str, np.ndarray],
    frequency: np.ndarray,
):
    """Plot timepoint mean L1-normalized radial-profile log2 change."""
    mat, labels = brain_l1_log2fc_matrix(grouped)
    vmax_abs = max(0.75, float(np.nanquantile(np.abs(mat), 0.98)))
    im = ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vmax_abs,
        vmax=vmax_abs,
        extent=[frequency[0], frequency[-1], len(labels) - 0.5, -0.5],
    )
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("Radial frequency (cycles / um)")
    ax.set_ylabel("Timepoint", labelpad=1)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.set_title("Log2FC", fontsize=5, pad=2)
    cbar.ax.tick_params(labelsize=5, width=0.4, length=2, pad=1)
    return im


class BrainDevProfilePlotter:
    """Load brain-dev data, fit requested genes, and draw profile rows."""

    def __init__(self, genes: list[str]) -> None:
        """Fit radial spectra for exactly the requested brain-dev genes."""
        self.samples, self.meta, _sample_sets = load_panel("brain_dev")
        present = set.intersection(*(set(adata.var_names) for adata in self.samples.values()))
        self.genes = unique(genes)
        missing = [gene for gene in self.genes if gene not in present]
        if missing:
            raise KeyError(f"Genes missing from loaded brain-dev samples: {missing}")

        self.comparator, self.paths_order = fit_all_samples(
            self.samples,
            self.genes,
            "radial",
            n_radial_bins=BRAIN_DEV_REFERENCE_N_RADIAL_BINS,
        )
        self.frequency = radial_frequency_axis(self.comparator)

        self.meta = self.meta.copy()
        self.meta["_path"] = self.meta["h5ad_path"].map(Path)
        self.meta_by_path = self.meta.set_index("_path").to_dict(orient="index")
        self.timepoint_by_path = {
            path: str(self.meta_by_path[path]["timepoint"]) for path in self.paths_order
        }

    def representative_paths(self) -> list[Path]:
        """Pick the lowest-replicate anterior sample for each timepoint."""
        return representative_brain_dev_paths(self.samples, self.meta)

    def gene_vmax(self, gene: str, paths: list[Path], q: float = 0.99) -> float:
        """Shared upper color limit for one gene across representative maps."""
        return common_gene_vmax((self.samples[path] for path in paths), gene, q=q)

    def plot_spatial(self, ax, gene: str, path: Path, vmax: float):
        """Draw one high-resolution spatial expression map."""
        return plot_brain_spatial(
            ax,
            self.samples[path],
            gene,
            self.timepoint_by_path[path],
            vmax=vmax,
        )

    def spectra_by_timepoint(self, gene: str) -> dict[str, np.ndarray]:
        """Return fitted raw radial spectra grouped by postnatal timepoint."""
        gi = self.genes.index(gene)
        return brain_spectra_by_timepoint(
            np.asarray(self.comparator.spectra_, dtype=float),
            gi,
            self.paths_order,
            self.timepoint_by_path,
        )

    def plot_raw_profiles(self, ax, gene: str) -> None:
        """Plot per-sample raw radial power curves plus timepoint means."""
        plot_brain_raw_profiles(ax, self.spectra_by_timepoint(gene), self.frequency)

    def plot_l1_log2fc_heatmap(self, ax, cax, gene: str):
        """Plot timepoint mean L1-normalized radial-profile log2 change."""
        return plot_brain_l1_log2fc_heatmap(
            ax,
            cax,
            self.spectra_by_timepoint(gene),
            self.frequency,
        )


def _draw_gene_row(
    fig,
    spec,
    *,
    plotter: BrainDevProfilePlotter,
    stats: pd.DataFrame,
    gene: str,
) -> None:
    """Draw a 50:50 spatial-map/profile row for one gene."""
    row = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=spec,
        width_ratios=[1.0, 1.0],
        wspace=0.24,
    )
    left = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=row[0],
        width_ratios=[1.0, 0.035],
        wspace=0.035,
    )
    maps = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=left[0], wspace=0.05, hspace=0.08)
    cax_maps = fig.add_subplot(left[1])

    rep_paths = plotter.representative_paths()
    vmax = plotter.gene_vmax(gene, rep_paths)
    last_sc = None
    for i, path in enumerate(rep_paths):
        ax = fig.add_subplot(maps[i // 3, i % 3])
        last_sc = plotter.plot_spatial(ax, gene, path, vmax)
        if i == 1:
            ax.text(
                0.5,
                1.16,
                rf"$\mathit{{{gene}}}$ spatial expression",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=6.5,
                clip_on=False,
            )
    cbar = fig.colorbar(last_sc, cax=cax_maps)
    cbar.outline.set_linewidth(0.45)
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.tick_params(labelsize=5, width=0.4, length=2, pad=1)
    cbar.ax.set_title("Log1p", fontsize=5.5, pad=2)

    right = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=row[1],
        width_ratios=[1.0, 0.035],
        height_ratios=[1.18, 0.62],
        hspace=0.35,
        wspace=0.035,
    )
    ax_profile = fig.add_subplot(right[0, 0])
    ax_spacer = fig.add_subplot(right[0, 1])
    ax_spacer.axis("off")
    ax_heat = fig.add_subplot(right[1, 0], sharex=ax_profile)
    cax_heat = fig.add_subplot(right[1, 1])

    gene_stats = stats.loc[gene]
    plotter.plot_raw_profiles(ax_profile, gene)
    ax_profile.set_title(
        "Raw-count radial profile | "
        f"DF p: 2D={format_pvalue(gene_stats.P_DF)}; "
        f"radial={format_pvalue(gene_stats.P_DF_radial)}",
        fontsize=6,
        pad=2,
    )
    plotter.plot_l1_log2fc_heatmap(ax_heat, cax_heat, gene)
    ax_heat.set_title(
        "L1-normalized profile change | "
        f"DF-norm p: 2D={format_pvalue(gene_stats.P_DF_norm)}; "
        f"radial={format_pvalue(gene_stats.P_DF_norm_radial)}",
        fontsize=6,
        pad=2,
    )


def _make_page(
    genes: list[str],
    *,
    plotter: BrainDevProfilePlotter,
    stats: pd.DataFrame,
    rows_per_page: int,
):
    """Build one PDF page containing up to ``rows_per_page`` gene rows."""
    height_mm = max(82.0, 55.0 * rows_per_page + 10.0)
    fig = plt.figure(figsize=(184 / 25.4, height_mm / 25.4))
    outer = gridspec.GridSpec(
        len(genes),
        1,
        figure=fig,
        left=0.045,
        right=0.97,
        top=0.94,
        bottom=0.06,
        hspace=0.55,
    )
    for i, gene in enumerate(genes):
        _draw_gene_row(fig, outer[i], plotter=plotter, stats=stats, gene=gene)
    return fig


def generate_pdf(genes: list[str], *, output: Path, rows_per_page: int) -> None:
    """Generate a multi-page brain-dev gene-profile PDF for explicit genes."""
    genes = unique(genes)
    if not genes:
        raise ValueError("No genes selected.")
    configure_paper_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)

    stats = _load_brain_dev_stats()
    missing_stats = [gene for gene in genes if gene not in stats.index]
    if missing_stats:
        raise KeyError(f"Genes missing from brain_dev_tests.csv: {missing_stats}")

    plotter = BrainDevProfilePlotter(genes)
    with PdfPages(output) as pdf:
        for start in range(0, len(genes), rows_per_page):
            subset = genes[start : start + rows_per_page]
            fig = _make_page(
                subset,
                plotter=plotter,
                stats=stats,
                rows_per_page=rows_per_page,
            )
            pdf.savefig(fig, bbox_inches=None, dpi=600)
            plt.close(fig)


def build_gene_list(args: argparse.Namespace) -> list[str]:
    """Parse and de-duplicate CLI gene arguments."""
    genes = unique(parse_gene_list(args.genes))
    if not genes:
        raise ValueError("No genes selected. Pass --genes.")
    return genes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse standalone plotter CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate brain-dev gene-profile PDF with six timepoint maps, "
            "raw radial profiles, and DF-norm profile-change heatmaps."
        )
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        required=True,
        help="Selected genes as space- or comma-separated values.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output PDF path.")
    parser.add_argument(
        "--rows-per-page",
        type=int,
        default=2,
        help="Number of gene rows per PDF page; maximum 4.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    if args.rows_per_page < 1:
        raise ValueError("--rows-per-page must be >= 1")
    if args.rows_per_page > 4:
        raise ValueError("--rows-per-page must be <= 4")
    genes = build_gene_list(args)
    generate_pdf(genes, output=args.output, rows_per_page=args.rows_per_page)
    print(args.output)
    print("genes:", ", ".join(genes))


if __name__ == "__main__":
    main()
