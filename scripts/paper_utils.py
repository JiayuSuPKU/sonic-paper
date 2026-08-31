"""Shared runtime and manuscript-figure helpers for paper analyses."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTERPROFILER_SCRIPT = PROJECT_ROOT / "scripts" / "run_clusterprofiler.R"
CACHE_MANIFEST_FILENAME = "cache_manifest.json"
CACHE_MANIFEST_SCHEMA_VERSION = 1
WORKFLOW_MANIFEST_FILENAME = "workflow_manifest.json"
WORKFLOW_MANIFEST_SCHEMA_VERSION = 2

_CACHE_ACTIVITY: dict[Path, dict[str, set[str]]] = {}

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


def installed_version(distribution: str) -> str:
    """Return an installed distribution version without making cache checks fail."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def path_fingerprint(
    path: str | Path,
    *,
    recursive: bool = False,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Return lightweight path metadata suitable for a cache signature."""
    path = Path(path).expanduser()
    fingerprint: dict[str, Any] = {
        "path": project_relative(path, project_root),
        "exists": path.exists(),
    }
    if not path.exists():
        return fingerprint
    stat = path.stat()
    fingerprint.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if path.is_dir():
        children = (
            [child for child in sorted(path.rglob("*")) if child.is_file()]
            if recursive
            else [
                path / name
                for name in (".zmetadata", ".zgroup", "zarr.json")
                if (path / name).is_file()
            ]
        )
        if children:
            fingerprint["files"] = [
                {
                    "path": str(child.relative_to(path)),
                    "size": child.stat().st_size,
                    "mtime_ns": child.stat().st_mtime_ns,
                }
                for child in children
            ]
    return fingerprint


def cache_signature(data: Mapping[str, Any]) -> str:
    """Hash canonical cache inputs and parameters."""
    payload = json.dumps(dict(data), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _manifest_output(path: str | Path, project_root: Path) -> str:
    return project_relative(Path(path), project_root)


def _load_cache_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"schema_version": CACHE_MANIFEST_SCHEMA_VERSION, "entries": {}}
    if manifest.get("schema_version") != CACHE_MANIFEST_SCHEMA_VERSION or not isinstance(
        manifest.get("entries"), dict
    ):
        return {"schema_version": CACHE_MANIFEST_SCHEMA_VERSION, "entries": {}}
    return manifest


def _record_cache_activity(result_dir: Path, section: str, status: str) -> None:
    path = (result_dir / CACHE_MANIFEST_FILENAME).resolve()
    activity = _CACHE_ACTIVITY.setdefault(path, {"computed": set(), "reused": set()})
    activity[status].add(section)
    if status == "computed":
        activity["reused"].discard(section)


def _consume_cache_status(result_dir: Path) -> str:
    path = (result_dir / CACHE_MANIFEST_FILENAME).resolve()
    activity = _CACHE_ACTIVITY.pop(path, None)
    if not activity:
        return "unknown"
    computed = bool(activity["computed"])
    reused = bool(activity["reused"])
    if computed and reused:
        return "mixed"
    return "computed" if computed else "reused"


def cache_entry_is_current(
    result_dir: Path,
    section: str,
    signature: str,
    *,
    outputs: Sequence[str | Path] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> bool:
    """Return whether one manifest section and all of its outputs are current."""
    manifest = _load_cache_manifest(result_dir / CACHE_MANIFEST_FILENAME)
    entry = manifest["entries"].get(section, {})
    owned = entry.get("outputs")
    if entry.get("signature") != signature or not isinstance(owned, list) or not owned:
        return False
    if outputs is not None and owned != [_manifest_output(path, project_root) for path in outputs]:
        return False
    for output in owned:
        path = Path(output)
        if not path.is_absolute():
            path = project_root / path
        if not path.exists() or (path.is_file() and path.stat().st_size == 0):
            return False
    _record_cache_activity(result_dir, section, "reused")
    return True


def update_cache_entry(
    result_dir: Path,
    section: str,
    signature: str,
    outputs: Sequence[str | Path],
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Atomically update one minimal section of an analysis cache manifest."""
    path = result_dir / CACHE_MANIFEST_FILENAME
    manifest = _load_cache_manifest(path)
    manifest["entries"][section] = {
        "signature": signature,
        "outputs": [_manifest_output(output, project_root) for output in outputs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    _record_cache_activity(result_dir, section, "computed")
    log(f"wrote {project_relative(path, project_root)} [{section}]")
    return path


def write_workflow_manifest(
    analysis: str,
    result_dir: Path,
    *,
    started_at: float,
    parameters: Mapping[str, Any],
    outputs: Mapping[str, Any],
    summary: Mapping[str, Any] | None = None,
    cache_status: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Write audit-only workflow metadata, separate from cache signatures.

    Cache validators and visualization notebooks must not use this manifest to
    decide whether analysis outputs need refreshing.
    """
    path = result_dir / WORKFLOW_MANIFEST_FILENAME
    inferred_status = _consume_cache_status(result_dir)
    cache_status = cache_status or inferred_status
    if cache_status not in {"computed", "mixed", "reused", "unknown"}:
        raise ValueError(f"Unknown cache status: {cache_status!r}")
    invocation = {
        "cache_status": cache_status,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - started_at, 6),
        "peak_rss_gib": round(max_rss_gb(), 6),
    }
    try:
        previous = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        previous = {}
    if not isinstance(previous, dict):
        previous = {}
    previous_runtime = previous.get("runtime", {})
    previous_compute = (
        previous_runtime.get("last_compute") if isinstance(previous_runtime, dict) else None
    )
    manifest: dict[str, Any] = {
        "schema_version": WORKFLOW_MANIFEST_SCHEMA_VERSION,
        "analysis": analysis,
        "parameters": dict(parameters),
        "outputs": dict(outputs),
        "runtime": {"last_invocation": invocation},
    }
    if cache_status in {"computed", "mixed"}:
        manifest["runtime"]["last_compute"] = invocation
    elif isinstance(previous_compute, dict):
        manifest["runtime"]["last_compute"] = previous_compute
    if summary is not None:
        manifest["summary"] = dict(summary)
    save_json(manifest, path, project_root=project_root)
    return path


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
