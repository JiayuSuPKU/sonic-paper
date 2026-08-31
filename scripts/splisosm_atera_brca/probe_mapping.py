"""Map Atera SVP probe sequences to hg38 and transcript annotations.

Source data:
    - Atera whole-transcriptome probe definitions from ``config.PROBE_CSV``.
    - hg38 genome FASTA plus ``.fai`` index from ``config.HG38_FASTA``.
    - GENCODE v49 gene/transcript annotation from ``config.GENCODE_GTF``.
    - PacBio BRCA QC-pass transcript structures and SQANTI annotation from
      ``config.PACBIO_GFF`` and ``config.SQANTI_TXT``.

The workflow passes all SVP genes with ``SV FDR < 0.01`` into
``prepare_svp_probe_mapping_cache``. For each requested gene, probes are first
searched against spliced GENCODE transcript sequences, so exon-junction probes
are projected back to split genomic segments instead of being treated as local
mismatches. If no transcript-space exact hit exists, the mapper falls back to a
padded GENCODE gene-locus exact search and finally to best local exon-near
mismatches. The resulting genomic probe segments are overlapped with both
GENCODE and PacBio exons to classify transcript compatibility.

Coordinate conventions:

- Input GTF/GFF positions are 1-based closed intervals.
- Internal overlap/search intervals are 0-based half-open intervals.
- Cached tables store both forms where useful:
  ``start_0based``/``end_0based`` and ``start_1based``/``end_1based``.
- Split probes are represented as semicolon-delimited ``segments_0based`` and
  ``segments_1based`` strings, with one row per tied mapping location.

Mapping status values:

- ``mapped_exact_spliced_transcript``: exact hit in a spliced GENCODE transcript;
  this is preferred because it correctly recovers exon-junction probes.
- ``mapped_exact_gene_locus``: exact hit in the padded genomic gene locus when no
  spliced transcript hit was found.
- ``mapped_best_gene_exon_mismatch``: best exon-near local mismatch within
  ``APPROX_MAX_MISMATCHES`` mismatches, used only as a last resort.
- ``unmapped_no_gencode_gene_locus`` / ``unmapped_gene_locus``: no usable mapping.

Main cached outputs:

- ``svp_probe_mappings.csv``: probe oligo to genomic segment mappings.
- ``svp_probe_gencode_overlaps.csv`` and ``svp_probe_pacbio_overlaps.csv``:
  per-probe overlap summaries against transcript exons.
- ``svp_gencode_transcript_probe_compatibility.csv`` and
  ``svp_pacbio_transcript_probe_compatibility.csv``: per-transcript probe
  compatibility classes.
- ``svp_probe_cache_inventory.tsv``: per-gene mapping completeness summary.
- ``svp_probe_cache_metadata.json``: request/input fingerprint used to reject
  stale cache reuse.

Idempotent pipeline:

1. Read probe definitions for the requested SVP genes.
2. Read matching GENCODE, PacBio, and SQANTI transcript annotations.
3. Map probe oligo sequences to spliced transcripts and hg38 gene loci.
4. Cache probe mappings, GENCODE/PacBio exon overlaps, transcript
   compatibility tables, and a per-gene inventory under
   ``results/splisosm_atera_brca/svp_probe_transcript_cache/``.

Reruns skip all work when every expected cache file exists, unless ``force`` is
set. This module owns the all-SVP mapping cache; plotting scripts should only
read its outputs.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    GENCODE_GTF,
    HG38_FASTA,
    PACBIO_GFF,
    PROBE_CSV,
    SQANTI_TXT,
    SVP_PROBE_RESULT_DIR,
)
from .utils import (
    log,
    normalize_gene_symbol,
    project_path,
    save_json,
    stable_hash_values,
)

# Exact genomic fallback uses a padded gene locus. The first-pass mapping still
# happens in transcript space, so this padding only affects probes not explained
# by annotated spliced GENCODE transcripts.
GENE_PAD = 25_000

# Approximate hits are intentionally rare: they are restricted to exon-near
# intervals and are emitted with mismatch counts so downstream plots can flag
# non-perfect mappings instead of silently treating them as exact probes.
APPROX_MAX_MISMATCHES = 5
CACHE_SCHEMA_VERSION = 2


def open_text(path: Path):
    """Open plain or gzipped annotation text in text mode."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return path.open()


def parse_attrs(attr_text: str) -> dict[str, str]:
    """Parse GTF/GFF attributes into ``key -> pipe-joined values``.

    GENCODE GTF uses ``key "value";`` attributes, while the PacBio GFF mostly
    uses ``key=value``. This helper accepts both forms and joins duplicate keys
    with ``|`` so no source attribute is dropped during parsing.
    """
    vals: dict[str, list[str]] = {}
    for item in attr_text.rstrip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        if " " in item:
            key, value = item.split(" ", 1)
        elif "=" in item:
            key, value = item.split("=", 1)
        else:
            continue
        value = value.strip().strip('"')
        vals.setdefault(key, []).append(value)
    return {key: "|".join(value) for key, value in vals.items()}


def revcomp(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1].upper()


def mismatch_count(a: str, b: str) -> int:
    """Count position-wise mismatches for equal-length strings."""
    return sum(x != y for x, y in zip(a.upper(), b.upper(), strict=False))


def opposite_strand(strand: str) -> str:
    """Flip a genomic strand symbol."""
    return "-" if strand == "+" else "+"


def format_segments(segments: list[tuple[str, int, int]]) -> str:
    """Serialize genomic segments as ``chrom:start0-end0`` chunks."""
    return ";".join(f"{chrom}:{int(start)}-{int(end)}" for chrom, start, end in segments)


def parse_segments(row: pd.Series | dict[str, object]) -> list[tuple[str, int, int]]:
    """Return one or more 0-based half-open genomic segments for a mapped probe row.

    ``segments_0based`` is preferred because it preserves exon-junction probes.
    The scalar chrom/start/end columns are used only for legacy/single-segment
    rows.
    """
    raw = str(row.get("segments_0based", "") or "")
    segments: list[tuple[str, int, int]] = []
    if raw and raw.lower() != "nan":
        for chunk in raw.split(";"):
            if not chunk:
                continue
            chrom, coords = chunk.split(":", 1)
            start, end = coords.split("-", 1)
            segments.append((chrom, int(float(start)), int(float(end))))
    if segments:
        return segments
    if pd.notna(row.get("start_0based", np.nan)) and pd.notna(row.get("end_0based", np.nan)):
        return [
            (
                str(row.get("chrom", "")),
                int(float(row["start_0based"])),
                int(float(row["end_0based"])),
            )
        ]
    return []


def _file_fingerprint(path: Path) -> dict[str, object]:
    """Return enough file metadata to reject stale mapping caches."""
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _expected_cache_metadata(genes: Iterable[str]) -> dict[str, object]:
    """Describe the probe-mapping request and source inputs for cache reuse."""
    gene_keys = sorted({normalize_gene_symbol(gene) for gene in genes if str(gene)})
    input_files = {
        "probe_csv": _file_fingerprint(PROBE_CSV),
        "gencode_gtf": _file_fingerprint(GENCODE_GTF),
        "pacbio_gff": _file_fingerprint(PACBIO_GFF),
        "sqanti_txt": _file_fingerprint(SQANTI_TXT),
        "hg38_fasta": _file_fingerprint(HG38_FASTA),
        "hg38_fasta_fai": _file_fingerprint(Path(str(HG38_FASTA) + ".fai")),
    }
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "requested_gene_keys": gene_keys,
        "requested_gene_key_hash": stable_hash_values(gene_keys),
        "n_requested_genes": len(gene_keys),
        "mapping_parameters": {
            "gene_pad": GENE_PAD,
            "approx_max_mismatches": APPROX_MAX_MISMATCHES,
            "coordinate_system": "0-based half-open internal; 1-based closed exported",
        },
        "input_files": input_files,
    }


def _cache_metadata_matches(path: Path, expected: dict[str, object]) -> bool:
    """Return true only when an existing cache was built from the same request."""
    if not path.exists():
        return False
    observed = json.loads(path.read_text())
    return (
        observed.get("schema_version") == expected.get("schema_version")
        and observed.get("requested_gene_key_hash") == expected.get("requested_gene_key_hash")
        and observed.get("mapping_parameters") == expected.get("mapping_parameters")
        and observed.get("input_files") == expected.get("input_files")
    )


def _segment_payload(segments: list[tuple[str, int, int]]) -> dict[str, object]:
    """Create the common mapping-column payload for one probe hit.

    The bounding start/end columns make simple tabular summaries easy, while
    ``segments_*`` preserve the true split structure needed for exon-junction
    probe plots.
    """
    segments = sorted(segments, key=lambda item: (item[0], int(item[1]), int(item[2])))
    starts = [start for _chrom, start, _end in segments]
    ends = [end for _chrom, _start, end in segments]
    return {
        "chrom": segments[0][0] if segments else np.nan,
        "start_0based": min(starts) if starts else np.nan,
        "end_0based": max(ends) if ends else np.nan,
        "start_1based": min(starts) + 1 if starts else np.nan,
        "end_1based": max(ends) if ends else np.nan,
        "segment_count": len(segments),
        "segments_0based": format_segments(segments),
        "segments_1based": ";".join(f"{chrom}:{start + 1}-{end}" for chrom, start, end in segments),
        "spans_exon_junction": len(segments) > 1,
    }


class IndexedFasta:
    """Minimal random-access FASTA reader backed by a samtools-style ``.fai`` index."""

    def __init__(self, fasta: Path):
        self.fasta = fasta
        fai = Path(str(fasta) + ".fai")
        if not fai.exists():
            raise FileNotFoundError(f"Missing FASTA index: {fai}")
        rows = []
        # Read the .fai index once so fetch() can jump directly to genomic bytes.
        with fai.open() as handle:
            for line in handle:
                chrom, length, offset, line_bases, line_width = line.rstrip("\n").split("\t")[:5]
                rows.append(
                    {
                        "chrom": chrom,
                        "length": int(length),
                        "offset": int(offset),
                        "line_bases": int(line_bases),
                        "line_width": int(line_width),
                    }
                )
        self.index = {row["chrom"]: row for row in rows}
        self._handle = fasta.open("rb")

    def close(self) -> None:
        """Close the underlying FASTA file handle."""
        self._handle.close()

    def fetch(self, chrom: str, start0: int, end0: int) -> str:
        """Fetch a 0-based half-open genomic sequence interval as uppercase DNA."""
        if chrom not in self.index:
            raise KeyError(f"{chrom} not present in {self.fasta}")
        info = self.index[chrom]
        start0 = max(0, int(start0))
        end0 = min(int(end0), int(info["length"]))
        if end0 <= start0:
            return ""
        line_bases = int(info["line_bases"])
        line_width = int(info["line_width"])
        offset = int(info["offset"])
        # FASTA offsets include newline width, so byte coordinates differ from
        # genomic coordinates after every wrapped sequence line.
        byte_start = offset + (start0 // line_bases) * line_width + (start0 % line_bases)
        byte_end = offset + (end0 // line_bases) * line_width + (end0 % line_bases)
        self._handle.seek(byte_start)
        data = self._handle.read(byte_end - byte_start + line_width)
        return data.replace(b"\n", b"").replace(b"\r", b"")[: end0 - start0].decode("ascii").upper()


def read_probe_info(path: Path, genes: Iterable[str]) -> pd.DataFrame:
    """Read Atera probe oligos for selected genes and add stable display IDs.

    Atera ``codeword`` identifies the spatial feature, but some codewords are
    implemented by multiple oligos. The ``probe_oligo_label`` and
    ``probe_instance_id`` columns keep those oligos distinct for mapping audits
    while leaving the feature-level codeword unchanged.
    """
    genes = set(genes)
    probe = pd.read_csv(path, keep_default_na=False)
    probe = probe[probe["target_name"].astype(str).isin(genes)].copy()
    probe["codeword"] = probe["codeword"].astype(int)
    probe["feature"] = probe["target_name"].astype(str) + "|" + probe["codeword"].astype(str)
    # Some codewords have multiple oligos; suffix labels (a/b/c) make split
    # probe mappings inspectable without changing the feature identity.
    probe["probe_ordinal"] = probe.groupby(["target_name", "codeword"]).cumcount() + 1
    probe["probe_n_for_codeword"] = probe.groupby(["target_name", "codeword"])[
        "codeword"
    ].transform("size")
    probe["probe_oligo_label"] = probe.apply(
        lambda row: (
            str(row["codeword"])
            if int(row["probe_n_for_codeword"]) == 1
            else f"{int(row['codeword'])}{chr(ord('a') + int(row['probe_ordinal']) - 1)}"
        ),
        axis=1,
    )
    probe["probe_instance_id"] = (
        probe["target_name"].astype(str)
        + "|"
        + probe["codeword"].astype(str)
        + "|"
        + probe["probe_ordinal"].astype(str)
    )
    return probe.reset_index(drop=True)


def read_gencode_subset(path: Path, genes: Iterable[str]) -> pd.DataFrame:
    """Read all GENCODE rows for selected gene symbols with normalized coordinates.

    Gene, transcript, exon, CDS, and UTR rows are retained because different
    downstream steps need different feature levels: loci come from gene rows,
    spliced sequence models from transcript/exon rows, and plotting can reuse
    the richer feature table.
    """
    genes = set(genes)
    rows: list[dict[str, object]] = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            attrs = parse_attrs(fields[8])
            gene_name = attrs.get("gene_name", "")
            if gene_name not in genes:
                continue
            chrom, source, feature, start, end, score, strand, frame, _ = fields
            rows.append(
                {
                    "source_name": "GENCODE v49",
                    "chrom": chrom,
                    "source": source,
                    "feature": feature,
                    "start_1based": int(start),
                    "end_1based": int(end),
                    "start_0based": int(start) - 1,
                    "end_0based": int(end),
                    "score": score,
                    "strand": strand,
                    "frame": frame,
                    **attrs,
                }
            )
    return pd.DataFrame(rows)


def read_pacbio_subset(path: Path, genes: Iterable[str]) -> pd.DataFrame:
    """Read all PacBio QC-pass GFF rows for selected gene symbols.

    PacBio rows are not used to discover probe genomic coordinates. They are
    used after mapping, as an independent long-read transcript model for
    compatibility and ANKRD30A TSS interpretation.
    """
    genes = set(genes)
    rows: list[dict[str, object]] = []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            attrs = parse_attrs(fields[8])
            gene_name = attrs.get("gene_name", "")
            if gene_name not in genes:
                continue
            chrom, source, feature, start, end, score, strand, frame, _ = fields
            rows.append(
                {
                    "source_name": "PacBio QC pass",
                    "chrom": chrom,
                    "source": source,
                    "feature": feature,
                    "start_1based": int(start),
                    "end_1based": int(end),
                    "start_0based": int(start) - 1,
                    "end_0based": int(end),
                    "score": score,
                    "strand": strand,
                    "frame": frame,
                    **attrs,
                }
            )
    return pd.DataFrame(rows)


def read_sqanti_subset(path: Path, transcript_ids: Iterable[str]) -> pd.DataFrame:
    """Read SQANTI rows for PacBio transcript IDs used by the selected genes.

    SQANTI contributes transcript-level metadata such as structural category,
    coding status, ORF length, and full-length read support. Missing SQANTI rows
    do not block probe mapping.
    """
    transcript_ids = set(map(str, transcript_ids))
    if not transcript_ids:
        return pd.DataFrame()
    rows = []
    with open_text(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        idx_iso = header.index("isoform")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == len(header) and fields[idx_iso] in transcript_ids:
                rows.append(dict(zip(header, fields, strict=False)))
    out = pd.DataFrame(rows)
    for col in [
        "FL_counts",
        "FL_samples",
        "ORF_length",
        "CDS_length",
        "length",
        "exons",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def gene_loci(gencode: pd.DataFrame) -> pd.DataFrame:
    """Return one explicit GENCODE ``gene`` locus row per selected gene."""
    genes = gencode[gencode["feature"].eq("gene")].copy()
    return genes[
        ["gene_name", "gene_id", "chrom", "strand", "start_1based", "end_1based"]
    ].drop_duplicates()


def find_all(seq: str, query: str) -> list[int]:
    """Return all overlapping start offsets for ``query`` in ``seq``."""
    hits: list[int] = []
    start = 0
    while True:
        idx = seq.find(query, start)
        if idx < 0:
            break
        hits.append(idx)
        start = idx + 1
    return hits


def merge_intervals(intervals: list[tuple[int, int]], gap: int = 0) -> list[tuple[int, int]]:
    """Merge sorted 0-based intervals separated by at most ``gap`` bases."""
    if not intervals:
        return []
    intervals = sorted((int(s), int(e)) for s, e in intervals if int(e) > int(s))
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def transcript_models(
    gencode: pd.DataFrame, fasta: IndexedFasta
) -> dict[str, list[dict[str, object]]]:
    """Build spliced GENCODE transcript sequences and transcript-to-genome maps.

    Each returned model stores the spliced transcript sequence in transcript
    orientation plus an ``exon_maps`` list that converts transcript offsets back
    into genomic intervals. That back-projection is what lets an exact
    transcript-space probe hit become two genomic segments when the probe spans
    an exon junction.
    """
    if gencode.empty or "feature" not in gencode:
        return {}
    transcripts = gencode[gencode["feature"].eq("transcript")].copy()
    exons = gencode[gencode["feature"].eq("exon")].copy()
    # Grouping by gene lets map_probes_to_gene_loci search only transcripts for
    # the probe's intended target gene instead of scanning all selected models.
    models_by_gene: dict[str, list[dict[str, object]]] = {}
    for _, tx in transcripts.iterrows():
        txid = str(tx.get("transcript_id", ""))
        if not txid:
            continue
        tx_exons = exons[exons["transcript_id"].astype(str).eq(txid)].copy()
        if tx_exons.empty:
            continue
        strand = str(tx.get("strand", "+"))
        # Negative-strand transcript sequence is still stored 5' to 3', so exons
        # are traversed in reverse genomic order and reverse-complemented.
        tx_exons = tx_exons.sort_values("start_0based", ascending=strand != "-")
        pieces: list[str] = []
        exon_maps: list[dict[str, int | str]] = []
        cursor = 0
        for _, exon in tx_exons.iterrows():
            chrom = str(exon["chrom"])
            start0 = int(exon["start_0based"])
            end0 = int(exon["end_0based"])
            seq = fasta.fetch(chrom, start0, end0)
            if strand == "-":
                seq = revcomp(seq)
            length = len(seq)
            pieces.append(seq)
            exon_maps.append(
                {
                    "chrom": chrom,
                    "genomic_start0": start0,
                    "genomic_end0": end0,
                    "tx_start": cursor,
                    "tx_end": cursor + length,
                }
            )
            cursor += length
        gene = str(tx.get("gene_name", ""))
        models_by_gene.setdefault(gene, []).append(
            {
                "gene_name": gene,
                "transcript_id": txid,
                "transcript_name": str(tx.get("transcript_name", txid)),
                "strand": strand,
                "sequence": "".join(pieces).upper(),
                "exon_maps": exon_maps,
            }
        )
    return models_by_gene


def project_transcript_interval(
    model: dict[str, object], start: int, end: int
) -> list[tuple[str, int, int]]:
    """Project a transcript-space interval onto one or more genomic segments.

    Parameters ``start`` and ``end`` are transcript-coordinate offsets in the
    spliced sequence. Returned intervals are genomic 0-based half-open segments.
    """
    strand = str(model["strand"])
    segments: list[tuple[str, int, int]] = []
    for exon in model["exon_maps"]:  # type: ignore[index]
        tx_start = int(exon["tx_start"])
        tx_end = int(exon["tx_end"])
        ov_start = max(start, tx_start)
        ov_end = min(end, tx_end)
        if ov_end <= ov_start:
            continue
        offset_start = ov_start - tx_start
        offset_end = ov_end - tx_start
        chrom = str(exon["chrom"])
        genomic_start = int(exon["genomic_start0"])
        genomic_end = int(exon["genomic_end0"])
        # Transcript-space intervals that cross exon junctions become multiple
        # genomic boxes, preserving split-probe structure for plotting.
        if strand == "-":
            seg_start = genomic_end - offset_end
            seg_end = genomic_end - offset_start
        else:
            seg_start = genomic_start + offset_start
            seg_end = genomic_start + offset_end
        segments.append((chrom, seg_start, seg_end))
    return segments


def exact_spliced_transcript_matches(
    models: list[dict[str, object]],
    binder: str,
) -> list[dict[str, object]]:
    """Find exact probe matches in spliced transcript sequences.

    The probe sequence and its reverse complement are both tested. Identical
    genomic segment/strand hits shared by multiple transcripts are collapsed into
    one row with pipe-delimited transcript IDs/names.
    """
    hits_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    for model in models:
        sequence = str(model["sequence"])
        tx_strand = str(model["strand"])
        for query_kind, query, query_type in [
            ("binder", binder, "binder_exact_on_spliced_transcript"),
            (
                "revcomp",
                revcomp(binder),
                "binder_reverse_complement_exact_on_spliced_transcript",
            ),
        ]:
            mapped_strand = tx_strand if query_kind == "binder" else opposite_strand(tx_strand)
            for offset in find_all(sequence, query):
                segments = project_transcript_interval(model, offset, offset + len(query))
                segments = sorted(segments, key=lambda item: (item[0], int(item[1]), int(item[2])))
                if not segments:
                    continue
                # Collapse identical genomic hits that are shared by multiple
                # transcript isoforms, retaining all hit transcript IDs.
                key = (
                    tuple(segments),
                    mapped_strand,
                    query_type,
                    query,
                )
                row = hits_by_key.setdefault(
                    key,
                    {
                        **_segment_payload(segments),
                        "mapped_strand": mapped_strand,
                        "query_type": query_type,
                        "match_sequence": query,
                        "approx_mismatches": 0,
                        "transcript_hit_ids": [],
                        "transcript_hit_names": [],
                    },
                )
                row["transcript_hit_ids"].append(str(model["transcript_id"]))  # type: ignore[index]
                row["transcript_hit_names"].append(str(model["transcript_name"]))  # type: ignore[index]

    out: list[dict[str, object]] = []
    for row in hits_by_key.values():
        row["transcript_hit_ids"] = "|".join(sorted(set(row["transcript_hit_ids"])))  # type: ignore[arg-type]
        row["transcript_hit_names"] = "|".join(sorted(set(row["transcript_hit_names"])))  # type: ignore[arg-type]
        row["n_transcript_hits"] = (
            len(str(row["transcript_hit_ids"]).split("|")) if row["transcript_hit_ids"] else 0
        )
        out.append(row)
    return sorted(
        out,
        key=lambda row: (
            int(row["start_0based"]),
            int(row["end_0based"]),
            str(row["transcript_hit_ids"]),
        ),
    )


def best_approx_matches_in_intervals(
    fasta: IndexedFasta,
    chrom: str,
    intervals: list[tuple[int, int]],
    binder: str,
    *,
    max_mismatches: int,
) -> list[dict[str, object]]:
    """Find best local mismatch hits within candidate intervals.

    This is intentionally not a genome-wide aligner. It scans only exon-near
    intervals for the selected gene and returns all ties at the lowest mismatch
    count not exceeding ``max_mismatches``. The fallback exists mainly to make
    imperfect probe definitions auditable; exact transcript/genomic hits should
    dominate normal workflow outputs.
    """
    best_dist = len(binder) + 1
    best: list[dict[str, object]] = []
    # Approximate matching is a last resort and is restricted to exon-near
    # intervals to avoid spurious best hits elsewhere in a padded gene locus.
    for interval_start, interval_end in intervals:
        fetch_start = max(0, int(interval_start))
        fetch_end = int(interval_end)
        region = fasta.fetch(chrom, fetch_start, fetch_end)
        if len(region) < len(binder):
            continue
        for strand, query, query_type in [
            ("+", binder, "binder_best_mismatch_on_gene_locus"),
            (
                "-",
                revcomp(binder),
                "binder_reverse_complement_best_mismatch_on_gene_locus",
            ),
        ]:
            qlen = len(query)
            for offset in range(0, len(region) - qlen + 1):
                ref = region[offset : offset + qlen]
                dist = mismatch_count(ref, query)
                if dist > max_mismatches or dist > best_dist:
                    continue
                segments = [(chrom, fetch_start + offset, fetch_start + offset + qlen)]
                row = {
                    **_segment_payload(segments),
                    "mapped_strand": strand,
                    "query_type": query_type,
                    "match_sequence": ref,
                    "approx_mismatches": dist,
                    "transcript_hit_ids": "",
                    "transcript_hit_names": "",
                    "n_transcript_hits": 0,
                }
                if dist < best_dist:
                    best_dist = dist
                    best = [row]
                else:
                    best.append(row)
    return best


def map_probes_to_gene_loci(
    probes: pd.DataFrame,
    loci: pd.DataFrame,
    gencode: pd.DataFrame,
    fasta: IndexedFasta,
    *,
    max_mismatches: int = APPROX_MAX_MISMATCHES,
) -> pd.DataFrame:
    """Map probe oligos to genomic coordinates for selected gene loci.

    Mapping is deterministic and deliberately conservative:

    1. Exact spliced GENCODE transcript search, preserving exon-junction splits.
    2. Exact padded genomic gene-locus search if transcript search has no hit.
    3. Best exon-near mismatch search if no exact hit exists.

    Each tied hit is emitted as a separate row with ``approx_tie_index`` and
    ``approx_tie_count``. Exact hits have ``approx_mismatches == 0``. Rows keep
    both probe-level identity columns and mapping-level status columns so the
    cache can be used for quality control, transcript overlap, and plotting.
    """
    rows: list[dict[str, object]] = []
    locus_map = loci.set_index("gene_name").to_dict("index")
    transcript_models_by_gene = transcript_models(gencode, fasta)
    exon_intervals_by_gene: dict[str, list[tuple[int, int]]] = {}
    # Precompute padded exon intervals for the approximate-match fallback.
    for gene, gdf in gencode[gencode["feature"].eq("exon")].groupby("gene_name"):
        exon_intervals_by_gene[gene] = merge_intervals(
            [
                (int(row["start_0based"]) - 25, int(row["end_0based"]) + 25)
                for _, row in gdf.iterrows()
            ],
            gap=50,
        )

    for _, probe in probes.iterrows():
        gene = str(probe["target_name"])
        if gene not in locus_map:
            rows.append({**probe.to_dict(), "mapping_status": "unmapped_no_gencode_gene_locus"})
            continue
        locus = locus_map[gene]
        chrom = str(locus["chrom"])
        start0 = max(0, int(locus["start_1based"]) - 1 - GENE_PAD)
        end0 = int(locus["end_1based"]) + GENE_PAD
        # The padded locus is used only for genomic exact/fallback searches; the
        # preferred transcript-space search is built from annotated exons above.
        region = fasta.fetch(chrom, start0, end0)
        binder = str(probe["binder"]).upper()
        # Search order: exact spliced transcript hit, exact genomic hit, then
        # best exon-near mismatch if no exact location exists.
        hits = exact_spliced_transcript_matches(transcript_models_by_gene.get(gene, []), binder)
        mapping_status = "mapped_exact_spliced_transcript"
        if not hits:
            for strand, query, query_type in [
                ("+", binder, "binder_exact_on_gene_locus"),
                ("-", revcomp(binder), "binder_reverse_complement_exact_on_gene_locus"),
            ]:
                for offset in find_all(region, query):
                    segments = [(chrom, start0 + offset, start0 + offset + len(query))]
                    hits.append(
                        {
                            **_segment_payload(segments),
                            "mapped_strand": strand,
                            "query_type": query_type,
                            "match_sequence": query,
                            "approx_mismatches": 0,
                            "transcript_hit_ids": "",
                            "transcript_hit_names": "",
                            "n_transcript_hits": 0,
                        }
                    )
            mapping_status = "mapped_exact_gene_locus"
        if not hits:
            approx_intervals = exon_intervals_by_gene.get(gene) or [(start0, end0)]
            hits = best_approx_matches_in_intervals(
                fasta,
                chrom,
                approx_intervals,
                binder,
                max_mismatches=max_mismatches,
            )
            mapping_status = "mapped_best_gene_exon_mismatch" if hits else "unmapped_gene_locus"
        if not hits:
            # No exact evidence anywhere in transcript or locus space; emit a
            # fully populated unmapped row so downstream schemas stay stable.
            rows.append(
                {
                    **probe.to_dict(),
                    "mapping_status": mapping_status,
                    "gene_locus_chrom": chrom,
                    "gene_locus_start_1based": int(locus["start_1based"]),
                    "gene_locus_end_1based": int(locus["end_1based"]),
                    "chrom": chrom,
                    "start_0based": np.nan,
                    "end_0based": np.nan,
                    "start_1based": np.nan,
                    "end_1based": np.nan,
                    "mapped_strand": np.nan,
                    "query_type": np.nan,
                    "match_sequence": np.nan,
                    "approx_mismatches": np.nan,
                    "approx_tie_index": np.nan,
                    "approx_tie_count": np.nan,
                    "segment_count": 0,
                    "segments_0based": "",
                    "segments_1based": "",
                    "spans_exon_junction": False,
                    "transcript_hit_ids": "",
                    "transcript_hit_names": "",
                    "n_transcript_hits": 0,
                }
            )
            continue
        for idx, hit in enumerate(hits, start=1):
            # Multiple exact loci or equal-best approximate loci are retained as
            # separate rows rather than arbitrarily choosing one.
            rows.append(
                {
                    **probe.to_dict(),
                    "mapping_status": mapping_status,
                    "gene_locus_chrom": chrom,
                    "gene_locus_start_1based": int(locus["start_1based"]),
                    "gene_locus_end_1based": int(locus["end_1based"]),
                    **hit,
                    "approx_tie_index": idx,
                    "approx_tie_count": len(hits),
                }
            )
    return pd.DataFrame(rows)


def probe_feature_overlap(
    mappings: pd.DataFrame, features: pd.DataFrame, source_label: str
) -> pd.DataFrame:
    """Summarize which transcripts have exon overlap with each mapped probe oligo.

    This is a per-probe view. For split probes, all segments must overlap exons
    from the same transcript for that transcript to be counted as an overlap.
    The strict same-transcript rule prevents an exon-junction probe from being
    incorrectly explained by separate isoforms that each cover only one side.
    """
    rows: list[dict[str, object]] = []
    if features.empty or "feature" not in features:
        exons = pd.DataFrame()
    else:
        exons = features[features["feature"].eq("exon")].copy()
    for _, probe in mappings.iterrows():
        base = probe.to_dict()
        if not str(probe.get("mapping_status", "")).startswith("mapped"):
            rows.append(
                {
                    **base,
                    "annotation_source": source_label,
                    "overlap_status": "unmapped",
                    "overlap_transcript_ids": "",
                    "overlap_transcript_names": "",
                    "overlap_exon_numbers": "",
                    "n_overlap_transcripts": 0,
                }
            )
            continue
        segments = parse_segments(probe)
        gene_exons = (
            exons[exons["gene_name"].astype(str).eq(str(probe["target_name"]))].copy()
            if not exons.empty
            else exons
        )
        hit_parts = []
        if not gene_exons.empty:
            for _txid, tx_exons in gene_exons.groupby(gene_exons["transcript_id"].astype(str)):
                # Require every segment of a split probe to overlap the same transcript.
                segment_hits = []
                for chrom, start0, end0 in segments:
                    ov = tx_exons[
                        (tx_exons["chrom"].astype(str).eq(chrom))
                        & (tx_exons["start_0based"].astype(int) < end0)
                        & (tx_exons["end_0based"].astype(int) > start0)
                    ]
                    if ov.empty:
                        segment_hits = []
                        break
                    segment_hits.append(ov)
                if segment_hits:
                    hit_parts.append(pd.concat(segment_hits, ignore_index=True))
        hits = pd.concat(hit_parts, ignore_index=True) if hit_parts else pd.DataFrame()
        if hits.empty:
            transcript_ids: list[str] = []
            transcript_names: list[str] = []
            exon_numbers: list[str] = []
        else:
            transcript_ids = sorted(hits["transcript_id"].dropna().astype(str).unique().tolist())
            transcript_names = sorted(
                hits.get("transcript_name", hits["transcript_id"])
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            exon_numbers = sorted(
                hits.get("exon_number", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
            )
        rows.append(
            {
                **base,
                "annotation_source": source_label,
                "overlap_status": ("overlaps_exon" if transcript_ids else "mapped_no_exon_overlap"),
                "overlap_transcript_ids": "|".join(transcript_ids),
                "overlap_transcript_names": "|".join(transcript_names),
                "overlap_exon_numbers": "|".join(exon_numbers),
                "n_overlap_transcripts": len(transcript_ids),
            }
        )
    return pd.DataFrame(rows)


def transcript_compatibility(  # noqa: C901
    transcripts: pd.DataFrame,
    exons: pd.DataFrame,
    probe_mappings: pd.DataFrame,
    *,
    source_label: str,
    sqanti: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify each transcript by the set of Atera codewords it can explain.

    Compatibility is exon-overlap based: a mapped probe contributes to a
    transcript only when every segment of the probe overlaps that transcript's
    exon chain. The resulting ``compatibility_group`` is a ``+``-joined codeword
    signature such as ``45213+50762`` or ``no_probe_overlap``. This table is the
    source for transcript/probe interpretation; it is not used to quantify TCGA
    probe-region counts.
    """
    rows: list[dict[str, object]] = []
    if transcripts.empty:
        return pd.DataFrame()
    sqanti_by_iso = {}
    if sqanti is not None and not sqanti.empty:
        sqanti_by_iso = sqanti.set_index("isoform").to_dict("index")
    mapped = probe_mappings[
        probe_mappings["mapping_status"].astype(str).str.startswith("mapped")
    ].copy()
    if exons.empty or "transcript_id" not in exons:
        exons = pd.DataFrame()
    for _, tx in transcripts.iterrows():
        txid = str(tx["transcript_id"])
        tx_exons = (
            exons[exons["transcript_id"].astype(str).eq(txid)].copy() if not exons.empty else exons
        )
        hits = []
        for _, probe in mapped[
            mapped["target_name"].astype(str).eq(str(tx["gene_name"]))
        ].iterrows():
            if tx_exons.empty:
                continue
            segments = parse_segments(probe)
            if not segments:
                continue
            # A transcript is probe-compatible only if each mapped segment hits
            # one of its exons.
            segment_overlaps = []
            for chrom, start0, end0 in segments:
                ov = tx_exons[
                    (tx_exons["chrom"].astype(str).eq(chrom))
                    & (tx_exons["start_0based"].astype(int) < end0)
                    & (tx_exons["end_0based"].astype(int) > start0)
                ]
                if ov.empty:
                    segment_overlaps = []
                    break
                segment_overlaps.append(ov)
            if not segment_overlaps:
                continue
            hits.append(
                {
                    "codeword": int(probe["codeword"]),
                    "feature": str(probe["feature"]),
                    "probe_oligo_label": str(probe["probe_oligo_label"]),
                    "mapping_status": str(probe["mapping_status"]),
                    "approx_mismatches": probe.get("approx_mismatches", np.nan),
                }
            )
        codewords = sorted({str(hit["codeword"]) for hit in hits}, key=lambda x: int(x))
        labels = sorted({hit["probe_oligo_label"] for hit in hits})
        # The compatibility signature intentionally ignores oligo suffixes and
        # keeps only codeword-level support, matching the spatial feature table.
        signature = "+".join(codewords) if codewords else "no_probe_overlap"
        row = {
            "annotation_source": source_label,
            "gene": str(tx["gene_name"]),
            "transcript_id": txid,
            "transcript_name": str(tx.get("transcript_name", txid)),
            "chrom": str(tx["chrom"]),
            "strand": str(tx["strand"]),
            "start_1based": int(tx["start_1based"]),
            "end_1based": int(tx["end_1based"]),
            "n_exons": int(tx_exons.shape[0]),
            "transcript_type": str(tx.get("transcript_type", tx.get("gene_type", ""))),
            "tag": str(tx.get("tag", "")),
            "structural_category": str(tx.get("structural_category", "")),
            "associated_transcript": str(tx.get("associated_transcript", "")),
            "compatible_codewords": "|".join(codewords),
            "compatible_probe_oligos": "|".join(labels),
            "compatibility_group": signature,
            "n_compatible_codewords": len(codewords),
            "exon_chain": ";".join(
                f"{int(row['start_1based'])}-{int(row['end_1based'])}"
                for _, row in tx_exons.sort_values("start_1based").iterrows()
            ),
        }
        if txid in sqanti_by_iso:
            for key in [
                "structural_category",
                "associated_transcript",
                "subcategory",
                "coding",
                "ORF_length",
                "CDS_length",
                "FL_counts",
                "FL_samples",
            ]:
                row[f"sqanti_{key}"] = sqanti_by_iso[txid].get(key, np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    counts = (
        out.groupby(["annotation_source", "gene", "compatibility_group"], dropna=False)
        .size()
        .rename("n_transcripts_in_compatibility_group")
        .reset_index()
    )
    return out.merge(counts, on=["annotation_source", "gene", "compatibility_group"], how="left")


def _svp_probe_cache_paths(out_dir: Path) -> dict[str, Path]:
    """Return the fixed all-SVP probe mapping cache file layout."""
    return {
        "mapping": out_dir / "svp_probe_mappings.csv",
        "gencode_overlaps": out_dir / "svp_probe_gencode_overlaps.csv",
        "pacbio_overlaps": out_dir / "svp_probe_pacbio_overlaps.csv",
        "gencode_compatibility": out_dir / "svp_gencode_transcript_probe_compatibility.csv",
        "pacbio_compatibility": out_dir / "svp_pacbio_transcript_probe_compatibility.csv",
        "inventory": out_dir / "svp_probe_cache_inventory.tsv",
        "metadata": out_dir / "svp_probe_cache_metadata.json",
    }


def _table_cache_paths(paths: dict[str, Path]) -> dict[str, Path]:
    """Return cache paths for tabular outputs only."""
    return {key: path for key, path in paths.items() if key != "metadata"}


def _build_svp_probe_cache_inventory(
    genes: list[str],
    probe_info: pd.DataFrame,
    mappings: pd.DataFrame,
    gencode: pd.DataFrame,
    pacbio: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize probe mapping completeness by requested SVP gene."""
    inventory_rows = []
    gencode_tx = gencode[gencode["feature"].eq("transcript")]
    pacbio_tx = pacbio[pacbio["feature"].eq("transcript")]
    for gene in genes:
        gene_mappings = mappings[mappings["target_name"].astype(str).eq(gene)]
        mapped = gene_mappings["mapping_status"].astype(str).str.startswith("mapped")
        inventory_rows.append(
            {
                "gene": gene,
                "n_codeword_features": int(
                    probe_info.loc[probe_info["target_name"].eq(gene), "feature"].nunique()
                ),
                "n_probe_oligos": int(probe_info["target_name"].eq(gene).sum()),
                "n_mapped_probe_oligos": int(
                    gene_mappings.loc[mapped, "probe_instance_id"].nunique()
                ),
                "n_mapping_rows_including_ties": int(mapped.sum()),
                "n_gencode_transcripts": int(gencode_tx["gene_name"].astype(str).eq(gene).sum()),
                "n_pacbio_transcripts": int(pacbio_tx["gene_name"].astype(str).eq(gene).sum()),
            }
        )
    return pd.DataFrame(inventory_rows)


def _build_svp_probe_mapping_tables(genes: list[str]) -> dict[str, pd.DataFrame]:
    """Compute all tables in the workflow-owned SVP probe mapping cache."""
    probe_info = read_probe_info(PROBE_CSV, genes)
    gencode = read_gencode_subset(GENCODE_GTF, genes)
    pacbio = read_pacbio_subset(PACBIO_GFF, genes)

    # SQANTI is optional metadata for PacBio transcript compatibility rows; it is
    # not needed to map probe sequences.
    pacbio_txids = (
        pacbio.loc[pacbio["feature"].eq("transcript"), "transcript_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    sqanti = read_sqanti_subset(SQANTI_TXT, pacbio_txids)
    loci = gene_loci(gencode)
    missing_loci = sorted(set(genes) - set(loci["gene_name"].astype(str)))
    if missing_loci:
        raise ValueError(f"Missing GENCODE gene loci for: {missing_loci}")

    # Probe locations are resolved by sequence search in each GENCODE gene locus,
    # then overlapped with both GENCODE and PacBio transcript structures.
    fasta = IndexedFasta(HG38_FASTA)
    try:
        mappings = map_probes_to_gene_loci(probe_info, loci, gencode, fasta)
    finally:
        fasta.close()

    # Overlap tables answer "which transcripts overlap this probe?" while
    # compatibility tables answer "which probes can this transcript explain?"
    gencode_overlaps = probe_feature_overlap(mappings, gencode, "GENCODE v49")
    pacbio_overlaps = probe_feature_overlap(mappings, pacbio, "PacBio QC pass")
    gencode_compat = transcript_compatibility(
        gencode[gencode["feature"].eq("transcript")].copy(),
        gencode[gencode["feature"].eq("exon")].copy(),
        mappings,
        source_label="GENCODE v49",
    )
    pacbio_compat = transcript_compatibility(
        pacbio[pacbio["feature"].eq("transcript")].copy(),
        pacbio[pacbio["feature"].eq("exon")].copy(),
        mappings,
        source_label="PacBio QC pass",
        sqanti=sqanti,
    )
    inventory = _build_svp_probe_cache_inventory(genes, probe_info, mappings, gencode, pacbio)

    return {
        "mapping": mappings,
        "gencode_overlaps": gencode_overlaps,
        "pacbio_overlaps": pacbio_overlaps,
        "gencode_compatibility": gencode_compat,
        "pacbio_compatibility": pacbio_compat,
        "inventory": inventory,
    }


def _write_svp_probe_mapping_tables(
    tables: dict[str, pd.DataFrame], paths: dict[str, Path]
) -> None:
    """Write all generated SVP probe mapping tables using the canonical filenames."""
    for key, table in tables.items():
        path = paths[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        if key == "inventory":
            table.to_csv(path, sep="\t", index=False)
            log(f"wrote {project_path(path)}")
        else:
            table.to_csv(path, index=False)


def prepare_svp_probe_mapping_cache(
    genes: Iterable[str],
    *,
    force: bool = False,
    out_dir: Path = SVP_PROBE_RESULT_DIR,
) -> dict[str, Path]:
    """Map Atera probes and transcript overlaps for all SVP genes selected by SV FDR.

    The returned dictionary names every cache path expected by downstream
    renderers. Existing complete caches are reused unless ``force`` is true.

    This is the only workflow-owned entry point in the module. Standalone
    renderers and the ANKRD30A notebook should read the CSV/TSV files it writes
    rather than recomputing genome mapping on the fly.
    """
    genes = list(dict.fromkeys(map(str, genes)))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _svp_probe_cache_paths(out_dir)
    expected_metadata = _expected_cache_metadata(genes)
    table_paths = _table_cache_paths(paths)
    if (
        all(path.exists() for path in table_paths.values())
        and _cache_metadata_matches(paths["metadata"], expected_metadata)
        and not force
    ):
        log(f"SVP probe mapping cache exists: {paths['mapping']}")
        return paths
    if not genes:
        raise ValueError("SVP probe mapping requires at least one SVP target gene")

    tables = _build_svp_probe_mapping_tables(genes)
    _write_svp_probe_mapping_tables(tables, paths)
    save_json(expected_metadata, paths["metadata"])
    return paths
