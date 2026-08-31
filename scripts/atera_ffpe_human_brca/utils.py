from __future__ import annotations

import gzip
import hashlib
import shutil
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scripts.paper_utils import (
    ensure_directories,
    log,
    max_rss_gb,
    normalize_gene_symbol,
    project_relative,
    time_call,
)
from scripts.paper_utils import (
    save_csv as _save_csv,
)
from scripts.paper_utils import (
    save_json as _save_json,
)

from .config import (
    ATERA_DATA_DIR,
    FIGURE_DIR,
    PACBIO_DATA_DIR,
    PROJECT_ROOT,
    RBP_GENES,
    REMAP_CACHE_DIR,
    REMAP_NR_HG38_BED,
    REMAP_NR_HG38_URLS,
    REMAP_TF_TARGETS_HG38,
    RESULT_DIR,
    SCRIPT_DIR,
    SDATA_ZARR,
    TCGA_ANALYSIS_DIR,
    TCGA_DATA_DIR,
)

__all__ = ["log", "max_rss_gb", "normalize_gene_symbol", "time_call"]


def ensure_dirs() -> None:
    ensure_directories(
        RESULT_DIR,
        FIGURE_DIR,
        SCRIPT_DIR,
        ATERA_DATA_DIR,
        TCGA_DATA_DIR,
        TCGA_ANALYSIS_DIR,
        PACBIO_DATA_DIR,
    )


def project_path(path: Path) -> str:
    return project_relative(path, PROJECT_ROOT)


def save_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    _save_csv(df, path, project_root=PROJECT_ROOT, index=index)


def save_json(data: dict[str, Any], path: Path) -> None:
    _save_json(data, path, project_root=PROJECT_ROOT)


def stable_hash_values(values: Iterable[object]) -> str:
    """Return a stable SHA256 hash for an unordered collection of identity values."""
    payload = "\n".join(sorted(map(str, values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def selected_to_dense(x: Any) -> np.ndarray:
    # Spatial plotting helpers expect dense vectors after selecting a few columns.
    if sp.issparse(x):
        return x.toarray()
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def load_sdata():
    import spatialdata as sd

    if not SDATA_ZARR.exists():
        raise FileNotFoundError(f"Missing SpatialData store: {SDATA_ZARR}")
    return sd.read_zarr(SDATA_ZARR)


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    # Benjamini-Hochberg over finite p-values while preserving NaNs in place.
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = ranked.size
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    restored = np.empty_like(q)
    restored[order] = q
    out[valid] = restored
    return out


def _open_text(path: Path):
    """Open plain or gzipped text files in UTF-8 text mode."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _looks_like_remap_target(token: str) -> bool:
    """Return whether a ReMap BED token looks like a TF target symbol."""
    token = token.strip()
    if not token or len(token) > 48:
        return False
    if token.startswith(("GSE", "SRX", "ENC", "ERR", "DRX")):
        return False
    if token.replace("_", "").replace("-", "").isdigit():
        return False
    if "," in token or ":" in token:
        return False
    return any(char.isalpha() for char in token)


def remap_target_from_fields(fields: list[str]) -> str | None:
    """Extract a TF symbol from ReMap BED fields."""
    # Newer ReMap BEDs carry the target symbol in field 10; older/name-only
    # records require parsing the peak name defensively.
    if len(fields) >= 10 and _looks_like_remap_target(fields[9]):
        return fields[9].strip()
    if len(fields) < 4:
        return None
    name = fields[3].replace("|", ".").replace(":", ".").replace(";", ".")
    parts = [part for part in name.split(".") if _looks_like_remap_target(part)]
    return parts[0] if parts else None


def ensure_remap_hg38_bed() -> Path:
    """Return the local ReMap 2022 human hg38 non-redundant BED."""
    if REMAP_NR_HG38_BED.exists():
        return REMAP_NR_HG38_BED

    REMAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    # Try the primary ReMap URL first, then the Zenodo mirror before failing.
    for url in REMAP_NR_HG38_URLS:
        try:
            log(f"downloading ReMap hg38 BED: {url}")
            with urllib.request.urlopen(url, timeout=120) as response:
                with NamedTemporaryFile("wb", delete=False, dir=str(REMAP_CACHE_DIR)) as tmp:
                    shutil.copyfileobj(response, tmp)
                    tmp_path = Path(tmp.name)
            tmp_path.replace(REMAP_NR_HG38_BED)
            return REMAP_NR_HG38_BED
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            log(f"ReMap hg38 download failed from {url}: {exc}")
    raise FileNotFoundError(
        f"Missing ReMap BED {REMAP_NR_HG38_BED}; download failed from configured URLs."
    ) from last_error


def remap_tf_symbols(*, force: bool = False) -> list[str]:
    """Return ReMap human TF symbols, parsing and caching the hg38 BED once."""
    if REMAP_TF_TARGETS_HG38.exists() and not force:
        df = pd.read_csv(REMAP_TF_TARGETS_HG38)
        return sorted(df["tf_symbol"].dropna().astype(str).unique())

    bed = ensure_remap_hg38_bed()
    targets: set[str] = set()
    # Only the TF symbol list is needed by DU; peak-level support is parsed in
    # the ANKRD30A-specific module for local TSS windows.
    with _open_text(bed) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            target = remap_target_from_fields(line.rstrip("\n").split("\t"))
            if target:
                targets.add(target)

    out = pd.DataFrame({"tf_symbol": sorted(targets)})
    out["tf_key"] = out["tf_symbol"].map(normalize_gene_symbol)
    save_csv(out, REMAP_TF_TARGETS_HG38)
    return out["tf_symbol"].tolist()


def rbp_symbols() -> list[str]:
    """Return configured POSTAR3 CLIP-backed RBP symbols."""
    return sorted(RBP_GENES)


def regulator_gene_sets() -> dict[str, list[str]]:
    """Return configured RBP and ReMap TF gene sets for DU."""
    # DU tests the union of POSTAR3 RBPs and ReMap TFs in one design matrix.
    return {
        "rbp": rbp_symbols(),
        "tf_chip": remap_tf_symbols(),
    }


def regulator_class_keys() -> dict[str, set[str]]:
    """Return normalized symbol sets keyed by regulator source class."""
    return {
        key: {normalize_gene_symbol(gene) for gene in genes}
        for key, genes in regulator_gene_sets().items()
    }
