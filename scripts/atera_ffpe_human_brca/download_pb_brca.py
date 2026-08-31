"""Download + cache BRCA PacBio long-read transcript resources.

Source: Veiga, Diogo FT, et al. "A comprehensive long-read isoform analysis
platform and sequencing resource for breast cancer." Science Advances 8.3 (2022): eabg6711.

Processed data are available from the Jackson Laboratory BRCA-LRseq-pipeline
GitHub repository ``TheJacksonLaboratory/BRCA-LRseq-pipeline``.
The workflow uses the QC-pass PacBio transcript structures and SQANTI transcript annotations,
plus the full-length read count table used to prioritize transcripts detected in ER+
long-read samples.

Cached structure:

- ``<data_root>/pacbio_brca_lrseq/PacBio_Breast_Cancer_all_QC_pass_transcripts_unique.gff.gz``
  stores transcript and exon structures for QC-pass PacBio isoforms.
- ``<data_root>/pacbio_brca_lrseq/Sqanti_annotation_QC_pass_transcripts_unique.txt.gz``
  stores SQANTI transcript classification and sample-level annotations.
- ``<data_root>/pacbio_brca_lrseq/BRCA_superPBID_counts.tsv.zip`` stores
  compressed per-sample full-length counts; this script extracts
  ``BRCA_superPBID_counts.tsv`` next to it for fast downstream loading.

Idempotent pipeline:

1. Downloads each source file into ``data/atera_ffpe_human_brca/pacbio_brca_lrseq/``.
2. Extracts the FL count TSV from the downloaded ZIP archive.
3. Skips existing downloads and extracted files unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "scripts.atera_ffpe_human_brca"

from .config import (
    PACBIO_DATA_DIR,
    PACBIO_FL_COUNTS_TSV,
    PACBIO_FL_COUNTS_URL,
    PACBIO_FL_COUNTS_ZIP,
    PACBIO_GFF,
    PACBIO_GFF_URL,
    SQANTI_TXT,
    SQANTI_TXT_URL,
)

DOWNLOADS = (
    (PACBIO_GFF_URL, PACBIO_GFF),
    (SQANTI_TXT_URL, SQANTI_TXT),
    (PACBIO_FL_COUNTS_URL, PACBIO_FL_COUNTS_ZIP),
)


def download(url: str, path: Path, *, force: bool = False) -> None:
    if path.exists() and not force:
        print(f"exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    # Download to a .part file first so interrupted runs do not look complete.
    request = urllib.request.Request(url, headers={"User-Agent": "sonic-paper"})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(path)
    print(f"downloaded: {path}")


def extract_fl_counts(*, force: bool = False) -> None:
    if PACBIO_FL_COUNTS_TSV.exists() and not force:
        print(f"exists: {PACBIO_FL_COUNTS_TSV}")
        return
    with zipfile.ZipFile(PACBIO_FL_COUNTS_ZIP) as archive:
        # The upstream archive currently contains a single TSV, but selecting by
        # suffix keeps this robust to harmless wrapper directory changes.
        members = [name for name in archive.namelist() if name.endswith(".tsv")]
        if not members:
            raise ValueError(f"No TSV file found in {PACBIO_FL_COUNTS_ZIP}")
        with archive.open(members[0]) as source, PACBIO_FL_COUNTS_TSV.open(
            "wb"
        ) as target:
            shutil.copyfileobj(source, target)
    print(f"extracted: {PACBIO_FL_COUNTS_TSV}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download BRCA PacBio long-read transcript files."
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    PACBIO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Keep downloads idempotent and then ensure the uncompressed count table exists.
    for url, path in DOWNLOADS:
        download(url, path, force=args.force)
    extract_fl_counts(force=args.force)


if __name__ == "__main__":
    main()
