"""Download Stereo-seq MOSTA mouse-embryo per-section files from CNGBdb.

Source: Chen, Ao, et al. "Spatiotemporal transcriptomic atlas of mouse organogenesis
using DNA nanoball-patterned arrays." Cell 185.10 (2022): 1777-1792.

The MOSTA dataset is hosted on CNGBdb accession STDS0000058, where the
``<stage>_<chip>S<slice>.MOSTA.h5ad`` files under ``.../stomics/`` are the processed +
domain/cell-type-annotated MOSTA products.

Preprocessing recovers the physical MOSTA bin50 lattice from ``obs.index`` and
writes 25 µm SpatialData zarr stores.

Idempotent download pipeline:

1. Select section IDs from ``config.MOUSE.ALL_SECTIONS``. ``--representative``
   uses ``config.MOUSE.REPRESENTATIVE_SECTIONS`` instead; ``--sections`` filters
   the configured IDs by exact match, substring, or shell-style glob.
2. For each selected public URL, read the remote byte count via ``HEAD`` or a
   one-byte ``Range`` request. Existing local files with matching sizes are
   skipped; files with unavailable remote size are kept but logged unverified.
3. Download missing or stale files to ``<name>.part`` with HTTP ``Range`` resume,
   verify the final byte count when known, then atomically rename into place.
   Downloads are parallelised across sections (``--workers N``, default 6).
4. Merge completed/kept rows into ``metadata.csv``, retain only the current
   configured panel, and record repo-relative ``h5ad_path`` plus ``source_url``.

Usage::

    # default: every section listed in config.MOUSE.ALL_SECTIONS (~87 GB)
    python -m scripts.embryo_xspecies.download_mosta_data

    # one representative section per configured timepoint (~20 GB)
    python -m scripts.embryo_xspecies.download_mosta_data --representative

    # a matched subset (exact names, substrings, or shell-style globs)
    python -m scripts.embryo_xspecies.download_mosta_data --sections E11.5_E1S1 'E14.5_*'
"""

from __future__ import annotations

import argparse
import logging
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as CFG
from .config import MOUSE as C
from .config import ensure_species_output_dirs
from .download_utils import download, matches_section, remote_size, repo_relative_path

logger = logging.getLogger("embryo_xspecies.download")


def _metadata_path(path: Path) -> str:
    return repo_relative_path(path)


def _configured_sections(*, representative: bool = False) -> list[tuple[str, str]]:
    sections_by_stage = C.REPRESENTATIVE_SECTIONS if representative else C.ALL_SECTIONS
    out: list[tuple[str, str]] = []
    for tp in C.KEPT_TIMEPOINTS:
        out.extend((tp, sid) for sid in sections_by_stage[tp])
    return out


def _resolve_sections(
    patterns: list[str] | None, *, representative: bool
) -> list[tuple[str, str]]:
    sections = _configured_sections(
        representative=False if patterns else representative
    )
    if not patterns:
        return sections
    return [(tp, sid) for tp, sid in sections if matches_section(sid, patterns)]


def download_is_current() -> bool:
    """Return whether every configured MOSTA h5ad and metadata row is present."""
    if not C.META_CSV.exists():
        return False
    metadata = pd.read_csv(C.META_CSV)
    required = {"section_id", "timepoint", "h5ad_path", "source_url", "bin_size_um"}
    if missing := required - set(metadata):
        raise ValueError(f"{C.META_CSV} is missing columns: {sorted(missing)}")

    configured = _configured_sections()
    selected = metadata[metadata["section_id"].isin(sid for _, sid in configured)]
    if selected["section_id"].duplicated().any() or len(selected) != len(configured):
        return False
    rows = selected.set_index("section_id")
    for timepoint, section_id in configured:
        if section_id not in rows.index:
            return False
        row = rows.loc[section_id]
        path = Path(str(row["h5ad_path"]))
        path = path if path.is_absolute() else CFG.PROJECT_ROOT / path
        expected_path = C.RAW_DIR / f"{section_id}.MOSTA.h5ad"
        expected_url = f"{C.MOSTA_BASE_URL}/{section_id}.MOSTA.h5ad"
        if (
            str(row["timepoint"]) != timepoint
            or path != expected_path
            or str(row["source_url"]) != expected_url
            or not np.isclose(float(row["bin_size_um"]), CFG.BIN_SIZE_UM)
            or not path.is_file()
            or path.stat().st_size == 0
        ):
            return False
    return True


def _fetch_one(
    tp: str,
    sid: str,
    *,
    force: bool,
    log_every: float = 15.0,
    max_attempts: int = 8,
) -> tuple[str, str, Path, int, str]:
    """Fetch one section. Returns ``(tp, sid, dest, new_bytes, status)``.

    ``status`` is one of ``"present"`` / ``"present-unverified"`` / ``"downloaded"``
    / ``"skip-404"`` / ``"skip-empty"`` / ``"skip-error"``. ``new_bytes`` is the
    number of bytes pulled this run (0 if the file was already present).
    """
    url = f"{C.MOSTA_BASE_URL}/{sid}.MOSTA.h5ad"
    dest = C.RAW_DIR / f"{sid}.MOSTA.h5ad"
    remote = remote_size(url)  # may be None if the server rejects/throttles HEAD
    if (
        dest.exists()
        and not force
        and remote is not None
        and dest.stat().st_size == remote
    ):
        logger.info("[%s] already present (%.1f GB) — skip.", sid, remote / 1e9)
        return tp, sid, dest, 0, "present"
    if dest.exists() and not force and remote is None:
        logger.warning(
            "[%s] already present (%.1f GB, remote size unavailable — not verified).",
            sid,
            dest.stat().st_size / 1e9,
        )
        return tp, sid, dest, 0, "present-unverified"
    size_note = f"{remote / 1e9:.1f} GB" if remote is not None else "unknown size"
    part_now = C.RAW_DIR / f"{sid}.MOSTA.h5ad.part"
    resume_note = (
        f" (resuming from {part_now.stat().st_size / 1e6:.1f} MB)"
        if part_now.exists() and part_now.stat().st_size > 0
        else ""
    )
    logger.info("[%s] downloading (%s)%s …", sid, size_note, resume_note)
    try:
        n = download(
            url,
            dest,
            tag=sid,
            log_every=log_every,
            expected_size=remote,
            max_attempts=max_attempts,
        )
    except urllib.error.HTTPError as exc:
        logger.warning("[%s] %s — skipping.", sid, exc)
        return tp, sid, dest, 0, "skip-404"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.error("[%s] download failed: %s", sid, exc)
        return tp, sid, dest, 0, "skip-error"
    if n == 0:
        if dest.exists() and dest.stat().st_size > 0:
            logger.info(
                "[%s] completed from existing partial file (%.1f GB).",
                sid,
                dest.stat().st_size / 1e9,
            )
            return tp, sid, dest, 0, "present"
        logger.warning(
            "[%s] downloaded 0 bytes — server returned empty; skipping.", sid
        )
        dest.unlink(missing_ok=True)
        return tp, sid, dest, 0, "skip-empty"
    logger.info("[%s] done (%.1f GB).", sid, n / 1e9)
    return tp, sid, dest, n, "downloaded"


def run(
    *,
    sections: list[str] | None = None,
    representative: bool = False,
    force: bool = False,
    workers: int = 6,
    max_attempts: int = 8,
) -> int:
    """Download selected MOSTA sections and update the local metadata cache."""
    ensure_species_output_dirs(C)
    selected_sections = _resolve_sections(sections, representative=representative)
    if not selected_sections:
        logger.error("No sections selected — check --sections patterns.")
        return 1

    workers = max(1, int(workers))
    logger.info(
        "MOSTA download: %d section(s) → %s  (workers=%d)",
        len(selected_sections),
        C.RAW_DIR,
        workers,
    )
    rows: list[dict] = []
    total_bytes = 0
    t0 = time.time()
    max_attempts = max(1, int(max_attempts))
    if workers == 1:
        results = [
            _fetch_one(tp, sid, force=force, max_attempts=max_attempts)
            for tp, sid in selected_sections
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    _fetch_one, tp, sid, force=force, max_attempts=max_attempts
                ): (
                    tp,
                    sid,
                )
                for tp, sid in selected_sections
            }
            results = []
            for fut in as_completed(futs):
                results.append(fut.result())
    for tp, sid, dest, n, status in results:
        total_bytes += n
        if status.startswith("skip-"):
            continue
        rows.append(
            {
                "section_id": sid,
                "timepoint": tp,
                "day": C.STAGE_TO_DAY[tp],
                "h5ad_path": _metadata_path(dest),
                "source_url": f"{C.MOSTA_BASE_URL}/{sid}.MOSTA.h5ad",
                "bin_size_um": CFG.BIN_SIZE_UM,
            }
        )
    failed = any(status.startswith("skip-") for *_, status in results)
    elapsed = time.time() - t0
    if total_bytes > 0:
        logger.info(
            "Aggregate throughput this run: %.1f GB in %.1f min (%.2f MB/s).",
            total_bytes / 1e9,
            elapsed / 60,
            total_bytes / 1e6 / max(elapsed, 1e-9),
        )

    if not rows:
        logger.error("Nothing downloaded — metadata.csv not written.")
        return 1
    new = pd.DataFrame(rows)
    # Merge into any existing metadata.csv rather than overwriting it. Rows for
    # sections handled this run take priority.
    if C.META_CSV.exists():
        old = pd.read_csv(C.META_CSV)
        old = old[~old["section_id"].isin(new["section_id"])]
        new = pd.concat([old, new], ignore_index=True)
    configured = {section_id for _, section_id in _configured_sections()}
    new = new[new["section_id"].isin(configured)]
    meta = new.sort_values(["day", "section_id"]).reset_index(drop=True)
    meta.to_csv(C.META_CSV, index=False)
    logger.info(
        "Wrote %s (%d rows total). New bytes this run: %.1f GB.",
        C.META_CSV,
        len(meta),
        total_bytes / 1e9,
    )
    return int(failed)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sections",
        nargs="+",
        default=None,
        metavar="PATTERN",
        help="Download configured section IDs matching any exact name, substring, or glob.",
    )
    p.add_argument(
        "--representative", action="store_true", help="Download one section per stage."
    )
    p.add_argument(
        "--force", action="store_true", help="Re-download even if present + size-sane."
    )
    p.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel download workers (default 6).",
    )
    p.add_argument("--max-attempts", type=int, default=8)
    args = p.parse_args(argv)
    return run(
        sections=args.sections,
        representative=args.representative,
        force=args.force,
        workers=args.workers,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    raise SystemExit(main())
