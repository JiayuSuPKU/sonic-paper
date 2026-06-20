"""Inspect AD/PIG spatial maps, raw spectra, and DF-norm profile shifts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.comparator_benchmark"

from .config import (
    AD_PIG_FULL_TESTS_FILE,
    AD_PIG_REFERENCE_BINNING,
    AD_PIG_REFERENCE_FEATURE_MODE,
    AD_PIG_REFERENCE_INPUT,
    AD_PIG_REFERENCE_N_RADIAL_BINS,
    EPS,
    RESULT_DIR,
)
from .plot_common import (
    configure_paper_matplotlib,
)
from .plot_common import (
    gene_values as common_gene_values,
)
from .plot_common import (
    gene_vmax as common_gene_vmax,
)
from .utils import (
    fit_all_samples,
    format_pvalue,
    l1_normalize_rows,
    load_panel,
    parse_gene_list,
    unique,
)

AD_MAP_LABELS = ("WT", "AppNL-G-F")
WT_COLOR = "#4d4d4d"
AD_COLOR = "#e69f00"


def _load_ad_pig_stats() -> pd.DataFrame:
    """Load reference-scenario AD/PIG p-values used for row annotations."""
    tests = pd.read_csv(RESULT_DIR / AD_PIG_FULL_TESTS_FILE)
    # Profile PDFs always annotate the manuscript reference scenario:
    # raw counts, linear radial bins, radial features, configured 30 bins.
    ref = tests[
        tests["panel"].eq("ad_pig")
        & tests["input"].eq(AD_PIG_REFERENCE_INPUT)
        & tests["binning"].eq(AD_PIG_REFERENCE_BINNING)
        & tests["feature_mode"].eq(AD_PIG_REFERENCE_FEATURE_MODE)
        & tests["n_radial_bins"].eq(AD_PIG_REFERENCE_N_RADIAL_BINS)
    ].copy()
    if ref.empty:
        raise RuntimeError(
            "Reference AD/PIG scenario is missing from ad_pig_full_tests.csv"
        )
    wide = ref.pivot_table(
        index="Feature", columns="test", values="P_value", aggfunc="first"
    )
    wide = wide.rename(columns={col: f"P_{col}" for col in wide.columns})
    return wide


class AdPigProfilePlotter:
    """Fit the requested AD/PIG genes and expose plotting helpers."""

    def __init__(self, genes: list[str]) -> None:
        """Load AD/PIG samples and fit radial spectra for exactly ``genes``."""
        self.samples, _meta, sample_sets = load_panel("ad_pig")
        present = set.intersection(
            *(set(adata.var_names) for adata in self.samples.values())
        )
        self.genes = unique(genes)
        missing = [gene for gene in self.genes if gene not in present]
        if missing:
            raise KeyError(f"Genes missing from loaded AD/PIG samples: {missing}")

        self.real_set = next(s for s in sample_sets if s.null_type == "real")
        # The standalone plotter is argument-driven: it fits only the requested
        # genes so custom profile PDFs do not depend on hidden panel choices.
        self.comparator, self.paths_order = fit_all_samples(
            self.samples, self.genes, "radial"
        )
        path_to_idx = {p: i for i, p in enumerate(self.paths_order)}
        self.wt_idx = np.asarray(
            [path_to_idx[p] for p in self.real_set.a_paths if p in path_to_idx],
            dtype=int,
        )
        self.ad_idx = np.asarray(
            [path_to_idx[p] for p in self.real_set.b_paths if p in path_to_idx],
            dtype=int,
        )
        self.wt_path = self.real_set.a_paths[0]
        self.ad_path = self.real_set.b_paths[0]

    def gene_values(self, gene: str, path: Path) -> np.ndarray:
        """Return log1p expression values for one gene/sample map."""
        return common_gene_values(self.samples[path], gene)

    def gene_vmax(self, gene: str, q: float = 0.99) -> float:
        """Shared upper color limit for the WT/AppNL-G-F maps of one gene."""
        return common_gene_vmax(
            (self.samples[self.wt_path], self.samples[self.ad_path]), gene, q=q
        )

    def plot_spatial(self, ax, gene: str, path: Path, title: str, vmax: float):
        """Draw one spatial expression map with a caller-provided color scale."""
        coords = np.asarray(self.samples[path].obsm["spatial"])
        vals = self.gene_values(gene, path)
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=vals,
            s=2.0,
            cmap="magma",
            vmin=0,
            vmax=vmax,
            linewidths=0,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=6, pad=1.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return sc

    def profile_arrays(self, gene: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return raw group spectra and DF-norm log2 fold-change profile."""
        gi = self.genes.index(gene)
        raw_wt = self.comparator.spectra_[self.wt_idx, gi, :]
        raw_ad = self.comparator.spectra_[self.ad_idx, gi, :]
        wt_l1 = l1_normalize_rows(raw_wt, EPS)
        ad_l1 = l1_normalize_rows(raw_ad, EPS)
        log2fc = np.log2((ad_l1.mean(axis=0) + EPS) / (wt_l1.mean(axis=0) + EPS))
        return raw_wt, raw_ad, log2fc

    def plot_raw_profile(self, ax, gene: str) -> None:
        """Plot per-sample raw radial spectra plus group means."""
        raw_wt, raw_ad, _log2fc = self.profile_arrays(gene)
        x = np.arange(raw_wt.shape[-1])
        for y, label, color in (
            (raw_wt, AD_MAP_LABELS[0], WT_COLOR),
            (raw_ad, AD_MAP_LABELS[1], AD_COLOR),
        ):
            for row in y:
                ax.plot(x, row, color=color, alpha=0.20, lw=0.50)
            ax.plot(
                x,
                y.mean(axis=0),
                color=color,
                lw=1.35,
                label=f"{label} (n={y.shape[0]})",
            )
        ax.legend(
            frameon=True, loc="upper right", fontsize=5, borderpad=0.2, handlelength=1.4
        )
        ax.set_yscale("log")
        ax.set_xlabel("Radial bin")
        ax.set_ylabel("P(k)")
        ax.set_title("Raw radial profile", fontsize=6.5, pad=2)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    def plot_l1_log2fc(self, ax, gene: str) -> None:
        """Plot AppNL-G-F/WT log2 fold change after per-sample L1 normalization."""
        _raw_wt, _raw_ad, log2fc = self.profile_arrays(gene)
        x = np.arange(log2fc.size)
        ax.axhline(0, color="0.25", lw=0.7)
        ax.fill_between(
            x,
            0,
            log2fc,
            where=log2fc >= 0,
            interpolate=True,
            color=AD_COLOR,
            alpha=0.35,
            lw=0,
        )
        ax.fill_between(
            x,
            0,
            log2fc,
            where=log2fc < 0,
            interpolate=True,
            color=WT_COLOR,
            alpha=0.35,
            lw=0,
        )
        ax.plot(x, log2fc, color="0.15", lw=1.05)
        ymax = max(0.35, float(np.nanmax(np.abs(log2fc))) * 1.12)
        ax.set_ylim(-ymax, ymax)
        ax.set_xlabel("Radial bin")
        ax.set_ylabel("log2 FC (App/WT)")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)


def _add_gene_row(
    fig,
    spec,
    *,
    plotter: AdPigProfilePlotter,
    gene: str,
    real: pd.DataFrame,
) -> None:
    """Add the spatial maps, raw profile, and DF-norm profile for one gene."""
    # Each gene row is a 3-part layout: spatial maps + colorbar, raw power
    # profile, and L1-normalized profile fold change.
    row_gs = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=spec,
        width_ratios=[1.58, 1.25, 1.32],
        wspace=0.48,
    )
    spatial_gs = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=row_gs[0],
        width_ratios=[1.0, 1.0, 0.042],
        wspace=0.035,
    )

    vmax = plotter.gene_vmax(gene)
    ax_wt = fig.add_subplot(spatial_gs[0])
    ax_ad = fig.add_subplot(spatial_gs[1])
    cax = fig.add_subplot(spatial_gs[2])
    ax_raw = fig.add_subplot(row_gs[1])
    ax_fc = fig.add_subplot(row_gs[2])

    plotter.plot_spatial(
        ax_wt,
        gene,
        plotter.wt_path,
        f"$\\it{{{gene}}}$" + f"\n{AD_MAP_LABELS[0]}",
        vmax,
    )
    sc = plotter.plot_spatial(ax_ad, gene, plotter.ad_path, AD_MAP_LABELS[1], vmax)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.ax.tick_params(labelsize=5, width=0.4, length=2)
    cbar.ax.set_title("Log1p", fontsize=5.5, pad=1)

    # The two curve panels reuse the fitted spectra, so no extra comparator
    # work is done while paging through genes.
    plotter.plot_raw_profile(ax_raw, gene)
    plotter.plot_l1_log2fc(ax_fc, gene)

    stats = real.loc[gene]
    ax_fc.set_title(
        "DF-norm log2FC\n"
        f"DE {format_pvalue(stats.P_DE)} | DF {format_pvalue(stats.P_DF)} | "
        f"DF-norm {format_pvalue(stats.P_DF_norm)}",
        fontsize=6,
        pad=2,
    )


def _make_page(
    subset: list[str],
    *,
    plotter: AdPigProfilePlotter,
    real: pd.DataFrame,
    rows_per_page: int,
):
    """Build one PDF page containing up to ``rows_per_page`` gene rows."""
    height_mm = max(70.0, 38.5 * rows_per_page + 8.0)
    fig = plt.figure(figsize=(185 / 25.4, height_mm / 25.4))
    outer = gridspec.GridSpec(
        len(subset),
        1,
        figure=fig,
        left=0.05,
        right=0.95,
        top=0.95,
        bottom=0.05,
        hspace=0.6,
    )
    for i, gene in enumerate(subset):
        _add_gene_row(
            fig,
            outer[i],
            plotter=plotter,
            gene=gene,
            real=real,
        )
    return fig


def generate_pdf(
    genes: list[str],
    *,
    output: Path,
    rows_per_page: int,
) -> None:
    """Generate a multi-page AD/PIG gene-profile PDF for explicit genes."""
    genes = unique(genes)
    if not genes:
        raise ValueError("No genes selected.")
    configure_paper_matplotlib()
    output.parent.mkdir(parents=True, exist_ok=True)
    real = _load_ad_pig_stats()
    missing_stats = [gene for gene in genes if gene not in real.index]
    if missing_stats:
        raise KeyError(
            f"Genes missing from cached AD/PIG real-test table: {missing_stats}"
        )
    # Fit once per requested gene list, then reuse the plotter across pages.
    plotter = AdPigProfilePlotter(genes)

    with PdfPages(output) as pdf:
        # Paginate by gene rows; page geometry stays fixed for manuscript-scale
        # reading and stable visual comparison across profile PDFs.
        for start in range(0, len(genes), rows_per_page):
            subset = genes[start : start + rows_per_page]
            fig = _make_page(
                subset,
                plotter=plotter,
                real=real,
                rows_per_page=rows_per_page,
            )
            pdf.savefig(fig, bbox_inches=None)
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
        description="Generate AD/PIG gene-profile PDF with spatial maps, raw spectra, and DF-norm log2FC."
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        required=True,
        help="Selected genes as space- or comma-separated values.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output multi-page PDF path.",
    )
    parser.add_argument(
        "--rows-per-page",
        type=int,
        default=6,
        help="Number of gene rows per PDF page; maximum 6.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    if args.rows_per_page < 1:
        raise ValueError("--rows-per-page must be >= 1")
    if args.rows_per_page > 6:
        raise ValueError("--rows-per-page must be <= 6")
    genes = build_gene_list(args)
    generate_pdf(
        genes,
        output=args.output,
        rows_per_page=args.rows_per_page,
    )
    print(args.output)
    print("genes:", ", ".join(genes))


if __name__ == "__main__":
    main()
