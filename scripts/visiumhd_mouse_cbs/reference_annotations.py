from __future__ import annotations

import gzip
import re
import shutil
import urllib.error
import urllib.request
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DATASETS,
    REMAP_INTERVAL_DIR,
    REMAP_NR_MM39_BED,
    REMAP_NR_MM39_URLS,
    REMAP_TF_TARGETS,
    TF_CHIP_SOURCE,
)
from .utils import (
    load_sdata,
    log,
    normalize_gene_symbol,
    resolve_table_config,
    save_csv,
)

_REFERENCE_FEATURE_CACHE: dict[tuple[str, ...], dict[str, Any]] = {}


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _extract_attr(attr_text: object, key: str) -> str | None:
    match = re.search(rf'{re.escape(key)}[= ]"?([^";]+)', str(attr_text))
    return match.group(1).strip().strip('"') if match else None


def _normalize_tx_id(value: object) -> str:
    token = str(value).strip().split("|")[0].split(";")[0].split(" ")[0].split(",")[0]
    token = token.replace("ENSTMUST", "ENSMUST").replace("ENSMUT", "ENSMUST")
    token = token.split(":", 1)[1] if token.startswith(("transcript:", "rna:")) else token
    match = re.match(r"^(ENSMUST\d+)\.\d+$", token)
    return match.group(1) if match else token


def _as_patterns(values: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def _matches_any(values: list[object] | tuple[object, ...], patterns: list[str] | None) -> bool:
    if patterns is None:
        return True
    if not patterns:
        return False
    candidates = [str(value) for value in values if value is not None and pd.notna(value)]
    return any(
        fnmatchcase(candidate.casefold(), pattern.casefold())
        for candidate in candidates
        for pattern in patterns
    )


def _bed12_blocks(row: pd.Series) -> list[tuple[int, int]]:
    starts = [int(x) for x in str(row["blockStarts"]).strip(",").split(",") if x]
    sizes = [int(x) for x in str(row["blockSizes"]).strip(",").split(",") if x]
    return [
        (int(row["start"]) + start, int(row["start"]) + start + size)
        for start, size in zip(starts, sizes, strict=False)
    ]


def _gtf_row(tx_key: str, info: dict[str, str], blocks: list[tuple[int, int]]) -> dict[str, Any]:
    return {
        "group": "Reference transcripts",
        "tx_id": tx_key,
        "tx_name": info["tx_name"],
        "label": f"{tx_key} | {info['tx_name']}",
        "blocks": sorted(blocks),
    }


def _store_gtf_feature(
    fields: list[str],
    meta: dict[str, dict[str, str]],
    exons: dict[str, list[tuple[int, int]]],
) -> None:
    attrs = fields[8]
    tx_id = _extract_attr(attrs, "transcript_id")
    if not tx_id:
        return
    tx_key = _normalize_tx_id(tx_id)
    meta.setdefault(
        tx_key,
        {
            "tx_name": _extract_attr(attrs, "transcript_name") or tx_key,
            "gene_name": _extract_attr(attrs, "gene_name") or "",
            "gene_id": _normalize_tx_id(_extract_attr(attrs, "gene_id") or ""),
        },
    )
    if fields[2] == "exon":
        exons.setdefault(tx_key, []).append((int(fields[3]), int(fields[4])))


def _index_rows_by_gene(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_gene: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for gene in row.get("gene_names", ()):
            if gene:
                by_gene.setdefault(normalize_gene_symbol(gene), []).append(row)
    for gene_key, gene_rows in by_gene.items():
        by_gene[gene_key] = sorted(
            gene_rows, key=lambda r: (min(start for start, _ in r["blocks"]), r["label"])
        )
    return by_gene


def _parse_reference_gtf(path: Path) -> dict[str, Any]:
    meta: dict[str, dict[str, str]] = {}
    exons: dict[str, list[tuple[int, int]]] = {}
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 9:
                _store_gtf_feature(fields, meta, exons)

    rows = []
    by_tx = {}
    for tx_key, blocks in exons.items():
        if tx_key not in meta:
            continue
        row = _gtf_row(tx_key, meta[tx_key], blocks)
        row["gene_names"] = (meta[tx_key]["gene_name"], meta[tx_key]["gene_id"])
        rows.append(row)
        by_tx[tx_key] = row
    return {"by_tx": by_tx, "by_gene": _index_rows_by_gene(rows)}


def _parse_bed12(path: Path, mode: str) -> list[dict[str, Any]]:
    bed_cols = [
        "chrom",
        "start",
        "end",
        "name",
        "score",
        "strand",
        "thickStart",
        "thickEnd",
        "itemRgb",
        "blockCount",
        "blockSizes",
        "blockStarts",
    ]
    bed = pd.read_csv(path, sep="\t", header=None, names=bed_cols, usecols=list(range(12)))
    rows = []
    for _, row in bed.iterrows():
        name = str(row["name"])
        if mode == "trend_peak_bed12":
            gene_keys = {normalize_gene_symbol(name.split(":", 1)[0])}
            group = "3' TREND peaks"
            label = name
        else:
            gene_keys = {normalize_gene_symbol(part) for part in name.split("|") if part}
            group = "FFPE probes"
            parts = name.split("|")
            label = "|".join(parts[1:]) if parts and parts[0].startswith("ENSMUSG") else name
        rows.append(
            {
                "group": group,
                "name": name,
                "label": label,
                "gene_keys": gene_keys,
                "blocks": _bed12_blocks(row),
            }
        )
    return rows


def _store_ont_feature(
    fields: list[str],
    meta: dict[str, dict[str, str]],
    exons: dict[str, list[tuple[int, int]]],
) -> None:
    attrs = fields[8]
    feature_type = fields[2].lower()
    if feature_type in {"transcript", "mrna"}:
        tx_id = _extract_attr(attrs, "transcript_id") or _extract_attr(attrs, "ID")
        if not tx_id:
            return
        tx_key = _normalize_tx_id(tx_id)
        meta[tx_key] = {
            "tx_id": tx_key,
            "ref_tx_id": _normalize_tx_id(_extract_attr(attrs, "cmp_ref") or tx_key),
            "gene_name": _extract_attr(attrs, "ref_gene_name")
            or _extract_attr(attrs, "gene_name")
            or _extract_attr(attrs, "cmp_ref_gene")
            or "",
            "gene_id": _normalize_tx_id(
                _extract_attr(attrs, "ref_gene_id") or _extract_attr(attrs, "gene_id") or ""
            ),
            "class_code": _extract_attr(attrs, "class_code") or "",
        }
    elif feature_type == "exon":
        tx_id = (
            _extract_attr(attrs, "transcript_id")
            or _extract_attr(attrs, "Parent")
            or _extract_attr(attrs, "ID")
        )
        if tx_id:
            exons.setdefault(_normalize_tx_id(tx_id), []).append((int(fields[3]), int(fields[4])))


def _parse_ont_gff(path: Path) -> dict[str, Any]:
    meta: dict[str, dict[str, str]] = {}
    exons: dict[str, list[tuple[int, int]]] = {}
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 9:
                _store_ont_feature(fields, meta, exons)

    rows = []
    by_input = {}
    by_ref: dict[str, list[dict[str, Any]]] = {}
    for tx_key, info in meta.items():
        blocks = sorted(exons.get(tx_key, []))
        if not blocks:
            continue
        row = {
            **info,
            "label": f"{tx_key} -> {info['ref_tx_id']}",
            "blocks": blocks,
            "n_exons": len(blocks),
            "tx_len": sum(end - start for start, end in blocks),
            "gene_names": (info["gene_name"], info["gene_id"]),
        }
        rows.append(row)
        by_input[tx_key] = row
        by_ref.setdefault(row["ref_tx_id"], []).append(row)
    return {"by_input": by_input, "by_ref": by_ref, "by_gene": _index_rows_by_gene(rows)}


def load_reference_feature_cache(*, force: bool = False) -> dict[str, Any]:
    """Load/cache reference GTF, ONT GFF, 3' TREND BED12, and FFPE probe BED12.

    The returned object is intentionally plain dictionaries/lists so notebook
    cells can inspect it without learning custom classes.
    """
    paths = (
        str(DATASETS["ont_tx"].gtf_file),
        str(DATASETS["ont_tx"].gff_file),
        str(DATASETS["trend_3prime"].bed12_file),
        str(DATASETS["ffpe_probe"].bed12_file),
    )
    if not force and paths in _REFERENCE_FEATURE_CACHE:
        return _REFERENCE_FEATURE_CACHE[paths]

    cache = {
        "reference": _parse_reference_gtf(DATASETS["ont_tx"].gtf_file),
        "ont": _parse_ont_gff(DATASETS["ont_tx"].gff_file),
        "trend": _parse_bed12(DATASETS["trend_3prime"].bed12_file, "trend_peak_bed12"),
        "probe": _parse_bed12(DATASETS["ffpe_probe"].bed12_file, "probe_bed12"),
    }
    _REFERENCE_FEATURE_CACHE[paths] = cache
    return cache


def _best_ont_structure(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose one assembled ONT structure for a reference transcript ID."""
    if not candidates:
        return None
    rank = {"=": 6, "k": 5, "j": 4, "c": 3, "i": 2, "o": 1, "u": 0, "p": -1}
    return sorted(
        candidates,
        key=lambda x: (
            rank.get(str(x.get("class_code", "")), -2),
            int(x.get("n_exons", 0)),
            int(x.get("tx_len", 0)),
        ),
        reverse=True,
    )[0]


def _has_wildcard(patterns: list[str] | None) -> bool:
    return patterns is not None and any("*" in pattern or "?" in pattern for pattern in patterns)


def _spatial_feature_names(
    dataset_key: str,
    gene_name: str,
    *,
    patterns: list[str] | None,
    sdata_cache: dict[str, Any] | None,
    resolution: str,
) -> list[str]:
    """Resolve spatial feature IDs for a gene, preserving explicit pattern order."""
    if sdata_cache is not None and dataset_key in sdata_cache:
        sdata = sdata_cache[dataset_key]["sdata"]
    else:
        sdata = load_sdata(DATASETS[dataset_key])

    table_cfg = resolve_table_config(DATASETS[dataset_key], sdata, resolution)
    adata = sdata.tables[table_cfg["table_name"]]
    gene_key = normalize_gene_symbol(gene_name)
    mask = np.zeros(adata.n_vars, dtype=bool)
    for col in (table_cfg.get("gene_names"), table_cfg.get("group_iso_by")):
        if col and col in adata.var:
            mask |= adata.var[col].astype(str).map(normalize_gene_symbol).eq(gene_key).to_numpy()

    names = adata.var_names[mask].astype(str).tolist()
    names = [name for name in names if _matches_any([name, _normalize_tx_id(name)], patterns)]
    if not names:
        return []

    if patterns is not None and not _has_wildcard(patterns):
        ordered = []
        seen = set()
        for pattern in patterns:
            pattern_tx = _normalize_tx_id(pattern)
            for name in names:
                if name in seen:
                    continue
                if name == pattern or _normalize_tx_id(name) == pattern_tx or pattern in name:
                    ordered.append(name)
                    seen.add(name)
                    break
        return ordered

    idx = adata.var_names.get_indexer(names)
    x = adata[:, idx].layers.get("counts", adata[:, idx].X)
    counts = np.asarray(x.sum(axis=0)).ravel()
    return [names[i] for i in np.argsort(-counts)]


def _bed_row_matches_feature(row: dict[str, Any], feature: str) -> bool:
    feature = str(feature)
    return (
        row["name"] == feature
        or row["label"] == feature
        or row["name"].startswith(feature)
        or feature.startswith(row["name"])
        or feature in row["name"]
        or _matches_any([row["name"], row["label"]], [feature])
    )


def _match_bed_rows(
    source_rows: list[dict[str, Any]],
    feature_names: list[str],
    gene_key: str,
    patterns: list[str] | None,
) -> list[dict[str, Any]]:
    """Match selected TREND/probe spatial features back to BED12 structures."""
    ordered_terms = feature_names or patterns or []
    if ordered_terms:
        rows = []
        seen = set()
        for term in ordered_terms:
            for row in source_rows:
                if row["name"] in seen or gene_key not in row["gene_keys"]:
                    continue
                if _bed_row_matches_feature(row, term):
                    rows.append(row)
                    seen.add(row["name"])
                    if not _has_wildcard([term]):
                        break
        return rows

    return [
        row
        for row in source_rows
        if gene_key in row["gene_keys"] and _matches_any([row["name"], row["label"]], patterns)
    ]


def _select_ont_rows(
    gene_name: str,
    reference_cache: dict[str, Any],
    patterns: list[str] | None,
    sdata_cache: dict[str, Any] | None,
    resolution: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Resolve ONT rows from explicit IDs or spatial ONT features."""
    gene_key = normalize_gene_symbol(gene_name)
    if patterns is None or _has_wildcard(patterns):
        ont_ids = _spatial_feature_names(
            "ont_tx", gene_name, patterns=patterns, sdata_cache=sdata_cache, resolution=resolution
        )
    else:
        ont_ids = patterns

    if not ont_ids and patterns is not None:
        ont_ids = [
            row["ref_tx_id"]
            for row in reference_cache["ont"]["by_gene"].get(gene_key, [])
            if _matches_any([row["tx_id"], row["ref_tx_id"]], patterns)
        ]

    rows = []
    selected_ref = set()
    for tx_id in ont_ids:
        tx_key = _normalize_tx_id(tx_id)
        info = reference_cache["ont"]["by_input"].get(tx_key)
        info = info or _best_ont_structure(reference_cache["ont"]["by_ref"].get(tx_key, []))
        if info is None:
            continue
        ref_key = info["ref_tx_id"]
        ref_row = reference_cache["reference"]["by_tx"].get(ref_key)
        tx_name = ref_row["tx_name"] if ref_row else ref_key
        label = f"{ref_key} | {tx_name}" if tx_key == ref_key else f"{tx_key} -> {ref_key} | {tx_name}"
        rows.append({"group": "ONT transcripts", "label": label, "blocks": info["blocks"]})
        selected_ref.add(ref_key)
    return rows, selected_ref


def _select_bed_rows(
    dataset_key: str,
    gene_name: str,
    reference_cache: dict[str, Any],
    patterns: list[str] | None,
    sdata_cache: dict[str, Any] | None,
    resolution: str,
) -> list[dict[str, Any]]:
    """Select TREND peak or FFPE probe rows for `plot_feature_structure()`."""
    if patterns == []:
        return []

    cache_key = "probe" if dataset_key == "ffpe_probe" else "trend"
    features = _spatial_feature_names(
        dataset_key, gene_name, patterns=patterns, sdata_cache=sdata_cache, resolution=resolution
    )
    match_terms = patterns if patterns is not None and not _has_wildcard(patterns) else features
    return _match_bed_rows(
        reference_cache[cache_key], match_terms, normalize_gene_symbol(gene_name), patterns
    )


def _select_reference_rows(
    gene_key: str,
    reference_cache: dict[str, Any],
    patterns: list[str] | None,
    selected_ref: set[str],
) -> list[dict[str, Any]]:
    """Select reference transcript rows, excluding already displayed ONT refs."""
    rows = [
        row
        for row in reference_cache["reference"]["by_gene"].get(gene_key, [])
        if row["tx_id"] not in selected_ref
    ]
    if patterns is None or _has_wildcard(patterns):
        return [
            row
            for row in rows
            if _matches_any([row["tx_id"], row["tx_name"], row["label"]], patterns)
        ]

    selected = []
    seen = set()
    for pattern in patterns:
        for row in rows:
            if row["tx_id"] in seen:
                continue
            if _matches_any([row["tx_id"], row["tx_name"], row["label"]], [pattern]):
                selected.append(row)
                seen.add(row["tx_id"])
                break
    return selected




def ensure_remap_nr_bed() -> Path:
    """Return the local ReMap non-redundant mm39 BED, downloading if needed."""
    if TF_CHIP_SOURCE != "remap2022":
        raise ValueError(f"Unsupported TF_CHIP_SOURCE: {TF_CHIP_SOURCE}")
    if REMAP_NR_MM39_BED.exists():
        return REMAP_NR_MM39_BED

    REMAP_NR_MM39_BED.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in REMAP_NR_MM39_URLS:
        try:
            log(f"downloading ReMap BED: {url}")
            with urllib.request.urlopen(url, timeout=120) as response:
                with NamedTemporaryFile(
                    "wb", delete=False, dir=str(REMAP_NR_MM39_BED.parent)
                ) as tmp:
                    shutil.copyfileobj(response, tmp)
                    tmp_path = Path(tmp.name)
            tmp_path.replace(REMAP_NR_MM39_BED)
            return REMAP_NR_MM39_BED
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            log(f"ReMap download failed from {url}: {exc}")
    raise FileNotFoundError(
        f"Missing ReMap BED {REMAP_NR_MM39_BED}; download failed from configured URLs."
    ) from last_error


def _open_bed(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _looks_like_target(token: str) -> bool:
    token = token.strip()
    if not token or len(token) > 40:
        return False
    if token.startswith(("GSE", "SRX", "ENC", "ERR", "DRX")):
        return False
    if token.replace("_", "").replace("-", "").isdigit():
        return False
    if "," in token or ":" in token:
        return False
    return any(char.isalpha() for char in token)


def remap_target_from_fields(fields: list[str]) -> str | None:
    """Extract the transcriptional regulator symbol from ReMap BED fields."""
    if len(fields) >= 10 and _looks_like_target(fields[9]):
        return fields[9].strip()

    if len(fields) < 4:
        return None
    name = fields[3].replace("|", ".").replace(":", ".").replace(";", ".")
    parts = [part for part in name.split(".") if _looks_like_target(part)]
    return parts[0] if parts else None


def remap_tf_symbols(*, force: bool = False) -> list[str]:
    """Return ReMap mouse regulator symbols, cached after one streaming parse."""
    if REMAP_TF_TARGETS.exists() and not force:
        df = pd.read_csv(REMAP_TF_TARGETS)
        return sorted(df["tf_symbol"].dropna().astype(str).unique())

    bed_path = ensure_remap_nr_bed()
    targets: set[str] = set()
    with _open_bed(bed_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            target = remap_target_from_fields(line.rstrip("\n").split("\t"))
            if target:
                targets.add(target)

    out = pd.DataFrame({"tf_symbol": sorted(targets)})
    out["tf_key"] = out["tf_symbol"].map(normalize_gene_symbol)
    save_csv(out, REMAP_TF_TARGETS)
    return out["tf_symbol"].tolist()


def _interval_cache_path(chrom: str, start: int, end: int) -> Path:
    safe = f"{chrom}_{int(start)}_{int(end)}".replace(":", "_").replace("-", "_")
    return REMAP_INTERVAL_DIR / f"remap2022_nr_mm39_{safe}.csv"


def _parse_remap_bed_line(line: str) -> tuple[str, int, int, str] | None:
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 3:
        return None
    target = remap_target_from_fields(fields)
    if not target:
        return None
    return fields[0], int(fields[1]), int(fields[2]), target


def _scan_remap_interval(chrom: str, start: int, end: int) -> pd.DataFrame:
    rows = []
    seen_chrom = False
    with _open_bed(ensure_remap_nr_bed()) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parsed = _parse_remap_bed_line(line)
            if parsed is None:
                continue
            row_chrom, row_start, row_end, tf_symbol = parsed
            if row_chrom != chrom:
                if seen_chrom:
                    break
                continue
            seen_chrom = True
            if row_start > end:
                break
            if row_end < start:
                continue
            rows.append(
                {
                    "chrom": chrom,
                    "start": max(row_start, start),
                    "end": min(row_end, end),
                    "tf_symbol": tf_symbol,
                    "tf_key": normalize_gene_symbol(tf_symbol),
                    "source": "ReMap2022 nr mm39",
                }
            )
    return pd.DataFrame(rows)


def remap_binding_sites(
    chrom: str,
    start: int,
    end: int,
    *,
    targets: Iterable[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Return ReMap sites overlapping one interval, optionally filtered by TF symbol."""
    start = int(start)
    end = int(end)
    cache_path = _interval_cache_path(chrom, start, end)

    if cache_path.exists() and not force:
        sites = pd.read_csv(cache_path)
    else:
        sites = _scan_remap_interval(chrom, start, end)
        if not sites.empty:
            try:
                save_csv(sites, cache_path)
            except OSError as exc:
                log(f"could not cache ReMap interval {cache_path}: {exc}")

    if targets is not None and not sites.empty:
        target_keys = {normalize_gene_symbol(target) for target in targets}
        sites = sites[sites["tf_key"].isin(target_keys)].copy()
    return sites
