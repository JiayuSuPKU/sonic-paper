"""Shared runtime and manuscript-figure helpers for paper analyses."""

from __future__ import annotations

import json
import os
import resource
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTERPROFILER_SCRIPT = PROJECT_ROOT / "scripts" / "run_clusterprofiler.R"

MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 185.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / MM_PER_INCH
FIGURE_DPI = 300

FONT_CONFIG: dict[str, float] = {
    "font": 6.0,
    "title": 7.0,
    "label": 6.0,
    "tick": 5.5,
    "text": 5.0,
    "legend": 5.5,
}
ARIAL_MATHTEXT: dict[str, Any] = {
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.bfit": "Arial:bold:italic",
    "mathtext.cal": "Arial:italic",
    "mathtext.sf": "Arial",
    "mathtext.tt": "Arial",
    "mathtext.default": "it",
    "mathtext.fallback": None,
}
COMPUTER_MODERN_MATHTEXT = {
    "mathtext.fontset": "cm",
    "mathtext.default": "it",
    "mathtext.fallback": "cm",
}

# Okabe-Ito colors provide one named, colorblind-safe base palette. Analyses
# keep their own semantic mappings because the represented groups differ.
OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}


def log(message: str) -> None:
    """Print a timestamped progress message."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def ensure_directories(*paths: Path) -> None:
    """Create output directories and their parents."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def project_relative(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    """Return a project-relative display path when possible."""
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def resolve_cached_data_path(
    cached_path: str | Path,
    data_root: Path,
    *,
    subdirectory: str = "adata",
) -> Path:
    """Resolve a cached path, rebasing stale absolute paths by filename."""
    path = Path(cached_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(data_root / path)
    candidates.append(data_root / subdirectory / path.name)
    return next((p.resolve() for p in candidates if p.exists()), candidates[-1].resolve())


def save_csv(
    frame: Any,
    path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    index: bool = False,
) -> None:
    """Write a table and log its project-relative path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)
    log(f"wrote {project_relative(path, project_root)} ({frame.shape[0]:,} rows)")


def save_json(
    data: Mapping[str, Any],
    path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> None:
    """Write deterministic indented JSON and log its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2, sort_keys=True) + "\n")
    log(f"wrote {project_relative(path, project_root)}")


def normalize_gene_symbol(value: object) -> str:
    """Normalize gene symbols for case-insensitive joins."""
    return str(value).strip().casefold()


def max_rss_gb() -> float:
    """Return peak process memory in GiB on macOS and Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    bytes_per_gib = 1024**3 if sys.platform == "darwin" else 1024**2
    return rss / bytes_per_gib


def time_call(label: str, fn: Callable[[], Any]) -> tuple[Any, float, float]:
    """Run a callable and return its value, elapsed seconds, and peak GiB."""
    log(f"START {label}")
    started = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - started
    rss = max_rss_gb()
    log(f"END {label}: {elapsed:.2f}s, max_rss={rss:.2f} GB")
    return value, elapsed, rss


def paper_style(
    fonts: Mapping[str, float] = FONT_CONFIG,
    *,
    font_family: str | Sequence[str] = ("Arial", "DejaVu Sans"),
    figure_dpi: int | None = None,
    savefig_dpi: int | None = FIGURE_DPI,
    hide_spines: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Matplotlib rcParams for manuscript-scale figures."""
    families = [font_family] if isinstance(font_family, str) else list(font_family)
    style: dict[str, Any] = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": families,
        "font.size": fonts["font"],
        **ARIAL_MATHTEXT,
        "axes.titlesize": fonts["title"],
        "axes.labelsize": fonts["label"],
        "xtick.labelsize": fonts["tick"],
        "ytick.labelsize": fonts["tick"],
        "legend.fontsize": fonts["legend"],
        "legend.title_fontsize": fonts["legend"],
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    }
    if figure_dpi is not None:
        style["figure.dpi"] = figure_dpi
    if savefig_dpi is not None:
        style["savefig.dpi"] = savefig_dpi
    if hide_spines:
        style.update({"axes.spines.top": False, "axes.spines.right": False})
    if overrides:
        style.update(overrides)
    return style


def configure_paper_matplotlib(**kwargs: Any) -> None:
    """Apply paper_style to Matplotlib's global configuration."""
    import matplotlib as mpl

    mpl.rcParams.update(paper_style(**kwargs))


def manuscript_size(
    height_mm: float,
    width_mm: float = FIGURE_WIDTH_MM,
) -> tuple[float, float]:
    """Convert manuscript figure dimensions from millimetres to inches."""
    return width_mm / MM_PER_INCH, height_mm / MM_PER_INCH


def neglog10(values: Any, *, floor: float = 1e-300) -> np.ndarray:
    """Return finite, clipped -log10 p-values."""
    return -np.log10(np.clip(np.asarray(values, dtype=float), floor, 1.0))


def format_pvalue(value: float) -> str:
    """Format a p-value for compact figure annotations."""
    pvalue = float(value)
    if not np.isfinite(pvalue):
        return "NA"
    if pvalue <= 0:
        return "<1e-300"
    if pvalue < 1e-3:
        exponent = int(np.floor(np.log10(pvalue)))
        return f"{pvalue / 10**exponent:.1f}e{exponent}"
    return f"{pvalue:.3f}".rstrip("0").rstrip(".")


def parse_gene_list(values: Sequence[str] | None) -> list[str]:
    """Parse separated CLI gene values while preserving first-seen order."""
    genes: list[str] = []
    for value in values or ():
        text = str(value)
        for separator in ",;|":
            text = text.replace(separator, " ")
        genes.extend(text.split())
    return list(dict.fromkeys(genes))


def clusterprofiler_command(default_conda_env: str = "sonic_paper") -> list[str]:
    """Resolve Rscript from an override, active environment, Conda, or PATH."""
    if configured := os.environ.get("CLUSTERPROFILER_RSCRIPT"):
        return [configured]
    active = Path(sys.prefix) / "bin" / "Rscript"
    if active.exists():
        return [str(active)]
    if conda := shutil.which("conda"):
        env_name = os.environ.get("CLUSTERPROFILER_CONDA_ENV", default_conda_env)
        return [conda, "run", "-n", env_name, "Rscript"]
    if rscript := shutil.which("Rscript"):
        return [rscript]
    raise FileNotFoundError("Could not find Rscript in the active environment or PATH.")
