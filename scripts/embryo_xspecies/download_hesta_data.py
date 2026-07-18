"""Download Stereo-seq HESTA human-embryo per-section h5ad files from GEO.

Source: Pan, Jiexue, et al. "Spatiotemporal transcriptome atlas of human embryos
after gastrulation." Nature (2026): 1-11.

The HESTA dataset is hosted on GEO **GSE326326** and CNGB **STT0000025**.
Each GEO sample ships three supplementary files: a multi-GB ``.gef``,
the H&E ``.tif.gz``, and the processed per-section ``*_HESTA.h5ad`` files
(hundreds of MB to ~11 GB each). Only the h5ad files are downloaded here.

The section catalog is discovered at run time from GEO's machine-readable
brief dump (``config.HUMAN.GEO_BRIEF_URL``) — all 77 samples, their titles
(``"CS12-13, Embryo1, Section1"``), and the per-file FTP URLs. By default the
script downloads the section IDs listed in ``config.HUMAN.ALL_SECTIONS``.

Idempotent download pipeline:

1. Discover the current GEO catalog to map section IDs to public h5ad URLs.
2. Select section IDs from ``config.HUMAN.ALL_SECTIONS``. ``--representative``
   uses ``config.HUMAN.REPRESENTATIVE_SECTIONS`` instead; ``--sections`` filters
   the configured IDs by exact match, substring, or shell-style glob.
3. For each selected URL, read the remote byte count via ``HEAD`` or a one-byte
   ``Range`` request. Existing local files with matching sizes are skipped;
   files with unavailable remote size are kept but logged unverified.
4. Download missing or stale files to ``<name>.part`` with HTTP ``Range`` resume,
   verify the final byte count when known, then atomically rename into place.
   Downloads are parallelised across sections (``--workers N``, default 6).
5. Merge completed/kept rows into ``metadata.csv`` with repo-relative
   ``h5ad_path`` and explicit ``source_url``.

Usage::

    # default: every section listed in config.HUMAN.ALL_SECTIONS (21 sections, ~63 GB)
    python -m scripts.embryo_xspecies.download_hesta_data

    # one representative section per configured stage
    python -m scripts.embryo_xspecies.download_hesta_data --representative

    # a matched subset (exact names, substrings, or shell-style globs)
    python -m scripts.embryo_xspecies.download_hesta_data --sections CS17 'CS23_E1S*'
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import re
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import pandas as pd

from . import config as CFG
from .config import HUMAN as C
from .config import ensure_species_output_dirs
from .download_utils import download, open_url, remote_size, repo_relative_path

logger = logging.getLogger("embryo_xspecies.download")

# title like "CS12-13, Embryo1, Section1"
_TITLE_RE = re.compile(r"^(CS[\dA-Za-z-]+),\s*Embryo(\d+),\s*Section(\d+)\s*$")
# the per-section processed h5ad supplementary line
_H5AD_RE = re.compile(
    r"!Sample_supplementary_file_\d+\s*=\s*(\S*_HESTA\.h5ad)\s*$", re.M
)


def _metadata_path(section_id: str) -> str:
    return repo_relative_path(C.RAW_DIR / f"{section_id}_HESTA.h5ad")


@dataclass(frozen=True)
class _Section:
    gsm: str
    carnegie_stage: str  # e.g. "CS12-13"
    embryo: int
    section: int
    section_id: str  # e.g. "CS12-13_E1S1"
    url: str  # https URL to the *_HESTA.h5ad

    @property
    def day(self) -> float:
        return C.SECTION_TO_DAY.get(
            self.section_id, C.CS_TO_DAY.get(self.carnegie_stage, float("nan"))
        )


def _https(url: str) -> str:
    """Normalise an ``ftp://ftp.ncbi...`` GEO URL to ``https://``."""
    return re.sub(r"^ftp://", "https://", url.strip())


def _discover_catalog(*, timeout: int = 60) -> list[_Section]:
    """Parse GEO's brief sample dump into the full per-section catalog."""
    try:
        with open_url(C.GEO_BRIEF_URL, timeout=timeout) as resp:
            txt = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.error("Could not fetch GEO brief dump (%s): %s", C.GEO_BRIEF_URL, exc)
        return []
    out: list[_Section] = []
    for block in re.split(r"^\^SAMPLE = ", txt, flags=re.M)[1:]:
        gsm = block.splitlines()[0].strip()
        mt = re.search(r"!Sample_title\s*=\s*(.+)", block)
        mh = _H5AD_RE.search(block)
        if not (mt and mh):
            continue
        tm = _TITLE_RE.match(mt.group(1).strip())
        if not tm:
            logger.warning(
                "[%s] unparsable title %r — skipping.", gsm, mt.group(1).strip()
            )
            continue
        cs, emb, sec = tm.group(1), int(tm.group(2)), int(tm.group(3))
        url = _https(mh.group(1))
        # section_id from the filename's "<CS>_E<e>S<s>" segment (matches mouse style)
        fn = url.rsplit("/", 1)[-1]
        mid = re.search(r"_(CS[\dA-Za-z-]+_E\d+S\d+)_HESTA\.h5ad$", fn)
        section_id = mid.group(1) if mid else f"{cs}_E{emb}S{sec}"
        out.append(_Section(gsm, cs, emb, sec, section_id, url))
    return out


def _configured_section_ids(*, representative: bool = False) -> list[str]:
    sections_by_stage = C.REPRESENTATIVE_SECTIONS if representative else C.ALL_SECTIONS
    out: list[str] = []
    for cs in C.KEPT_TIMEPOINTS:
        out.extend(sections_by_stage.get(cs, []))
    return out


def _matches(section_id: str, patterns: list[str]) -> bool:
    return any(
        p == section_id or p in section_id or fnmatch.fnmatchcase(section_id, p)
        for p in patterns
    )


def _select(
    catalog: list[_Section],
    *,
    patterns: list[str] | None,
    representative: bool,
) -> list[_Section]:
    """Pick configured sections, one per stage, or sections matching patterns."""
    want = set(
        _configured_section_ids(representative=False if patterns else representative)
    )
    pool = sorted(
        [s for s in catalog if s.section_id in want],
        key=lambda s: (s.day, s.embryo, s.section),
    )
    missing = sorted(want - {s.section_id for s in pool})
    if missing:
        logger.warning(
            "Configured HESTA section ids missing from GEO catalog: %s", missing
        )
    if patterns:
        return [s for s in pool if _matches(s.section_id, patterns)]
    return pool


def _fetch_one(
    s: _Section, *, force: bool, max_attempts: int
) -> tuple[_Section, int, str]:
    """Fetch one section's h5ad. Returns ``(section, new_bytes, status)``."""
    dest = C.RAW_DIR / f"{s.section_id}_HESTA.h5ad"
    remote = remote_size(s.url)
    if (
        dest.exists()
        and not force
        and remote is not None
        and dest.stat().st_size == remote
    ):
        logger.info(
            "[%s] already present (%.0f MB) — skip.", s.section_id, remote / 1e6
        )
        return s, 0, "present"
    if dest.exists() and not force and remote is None:
        logger.warning(
            "[%s] already present (remote size unavailable — not verified).",
            s.section_id,
        )
        return s, 0, "present-unverified"
    size_note = f"{remote / 1e6:.0f} MB" if remote is not None else "unknown size"
    logger.info("[%s] downloading (%s) …", s.section_id, size_note)
    try:
        n = download(
            s.url,
            dest,
            tag=s.section_id,
            expected_size=remote,
            max_attempts=max_attempts,
        )
    except urllib.error.HTTPError as exc:
        logger.warning("[%s] %s — skipping.", s.section_id, exc)
        return s, 0, "skip-404"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.error("[%s] download failed: %s", s.section_id, exc)
        return s, 0, "skip-error"
    if n == 0 and not dest.exists():
        logger.warning("[%s] 0 bytes — skipping.", s.section_id)
        return s, 0, "skip-empty"
    logger.info("[%s] done (%.0f MB).", s.section_id, (dest.stat().st_size) / 1e6)
    return s, n, "downloaded"


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
        help="Download section IDs matching any exact name, substring, or glob.",
    )
    p.add_argument(
        "--representative", action="store_true", help="Download one section per stage."
    )
    p.add_argument(
        "--force", action="store_true", help="Re-download even if present + size-sane."
    )
    p.add_argument(
        "--workers", type=int, default=6, help="Parallel download workers (default 6)."
    )
    p.add_argument(
        "--max-attempts", type=int, default=8, help="Per-section retry budget."
    )
    args = p.parse_args(argv)

    ensure_species_output_dirs(C)
    catalog = _discover_catalog()
    if not catalog:
        logger.error("Empty catalog — could not discover sections from GEO.")
        return 1
    logger.info(
        "Discovered %d HESTA sections across %d stages.", len(catalog), len(C.CS_TO_DAY)
    )
    chosen = _select(
        catalog,
        patterns=args.sections,
        representative=args.representative,
    )
    if not chosen:
        logger.error("No sections selected — check --sections patterns.")
        return 1
    workers = max(1, int(args.workers))
    max_attempts = max(1, int(args.max_attempts))
    logger.info(
        "Selected %d section(s) → %s  (workers=%d)", len(chosen), C.RAW_DIR, workers
    )

    t0 = time.time()
    if workers == 1:
        results = [
            _fetch_one(s, force=args.force, max_attempts=max_attempts) for s in chosen
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_fetch_one, s, force=args.force, max_attempts=max_attempts): s
                for s in chosen
            }
            results = [fut.result() for fut in as_completed(futs)]

    rows: list[dict] = []
    total_bytes = 0
    for s, n, status in results:
        total_bytes += n
        if status.startswith("skip-"):
            continue
        rows.append(
            {
                "section_id": s.section_id,
                "timepoint": s.carnegie_stage,
                "day": s.day,
                "gsm": s.gsm,
                "embryo": s.embryo,
                "section": s.section,
                "h5ad_path": _metadata_path(s.section_id),
                "source_url": s.url,
                "bin_size_um": CFG.BIN_SIZE_UM,
            }
        )
    elapsed = time.time() - t0
    if not rows:
        logger.error("Nothing downloaded — metadata.csv not written.")
        return 1
    new = pd.DataFrame(rows)
    # Merge into any existing metadata.csv rather than overwriting it, so a
    # subset run does not wipe rows for sections fetched by earlier runs. Rows
    # for sections handled this run take priority.
    if C.META_CSV.exists():
        try:
            old = pd.read_csv(C.META_CSV)
            old = old[~old["section_id"].isin(new["section_id"])]
            new = pd.concat([old, new], ignore_index=True)
        except (OSError, ValueError, KeyError) as exc:
            logger.warning(
                "Could not merge existing %s (%s) — writing fresh.", C.META_CSV, exc
            )
    stage_rank = {tp: i for i, tp in enumerate(C.KEPT_TIMEPOINTS)}
    meta = (
        new.assign(_stage_rank=new["timepoint"].map(stage_rank).fillna(len(stage_rank)))
        .sort_values(["_stage_rank", "section_id"])
        .drop(columns="_stage_rank")
        .reset_index(drop=True)
    )
    meta.to_csv(C.META_CSV, index=False)
    logger.info(
        "Wrote %s (%d rows total). New bytes this run: %.2f GB in %.1f min.",
        C.META_CSV,
        len(meta),
        total_bytes / 1e9,
        elapsed / 60,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
