"""Brain-development benchmark helpers derived from Cell 2026 supplements."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from scripts.paper_utils import (
    cache_entry_is_current,
    cache_signature,
    path_fingerprint,
    source_provenance,
    update_cache_entry,
)

from .config import (
    BRAIN_DEV_ANISOTROPY_FILE,
    BRAIN_DEV_CELL_TYPE_CONTROL_GROUPS,
    BRAIN_DEV_CELL_TYPE_CONTROL_TOP_N,
    BRAIN_DEV_CELL_TYPE_LABELS,
    BRAIN_DEV_CELL_TYPE_MANUAL_GENES,
    BRAIN_DEV_CELL_TYPE_PROGRAM_TOP_N,
    BRAIN_DEV_POSITIVE_CONTROLS_FILE,
    BRAIN_DEV_TABLE_S4_CNMFS,
    RESULT_DIR,
)
from .utils import cache_has_columns, save_csv


def _read_xlsx_values(  # noqa: C901
    path: Path, sheet_index: int = 0
) -> list[list[str]]:
    """Read plain cell values from one worksheet without requiring openpyxl."""
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }

    def _col_to_num(col: str) -> int:
        out = 0
        for char in col:
            out = out * 26 + ord(char) - ord("A") + 1
        return out

    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared.append(
                    "".join(
                        text.text or ""
                        for text in item.iter(
                            "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
                        )
                    )
                )

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        sheets = workbook.find("a:sheets", ns)
        if sheets is None or sheet_index >= len(sheets):
            raise IndexError(f"{path} has no worksheet index {sheet_index}")
        sheet = sheets[sheet_index]
        rid = sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        worksheet_path = "xl/" + relmap[rid].lstrip("/")
        worksheet = ET.fromstring(zf.read(worksheet_path))

        rows: list[list[str]] = []
        max_col = 0
        for row in worksheet.findall(".//a:sheetData/a:row", ns):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", ns):
                match = re.match(r"([A-Z]+)\d+", cell.attrib.get("r", ""))
                if match is None:
                    continue
                col_idx = _col_to_num(match.group(1))
                max_col = max(max_col, col_idx)
                value = cell.find("a:v", ns)
                if value is None:
                    continue
                text = value.text or ""
                if cell.attrib.get("t") == "s":
                    text = shared[int(text)]
                values[col_idx] = text
            if values:
                rows.append([values.get(i, "") for i in range(1, max_col + 1)])
    return rows


def _safe_float(value: Any) -> float:
    """Convert a loose spreadsheet value to float, returning NaN on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _parse_cnmf_gene_cell(value: str) -> tuple[str, float] | None:
    """Parse a Table S4 ranked-gene cell, e.g. ``Pvalb (7.8)``."""
    text = str(value).strip()
    if not text or "(" not in text or not text.endswith(")"):
        return None
    gene, score = text.rsplit("(", maxsplit=1)
    gene = gene.strip()
    cnmf_score = _safe_float(score.rstrip(")"))
    if not gene or not np.isfinite(cnmf_score):
        return None
    return gene, float(cnmf_score)


def _read_table_s4_programs() -> tuple[pd.DataFrame, pd.DataFrame]:  # noqa: C901
    """Return Table S4 program-enrichment and ranked-gene records."""
    if not BRAIN_DEV_TABLE_S4_CNMFS.exists():
        raise FileNotFoundError(
            f"Missing {BRAIN_DEV_TABLE_S4_CNMFS}; copy mmc4.xlsx into "
            "data/brain_dev/supplementary first."
        )
    rows = _read_xlsx_values(BRAIN_DEV_TABLE_S4_CNMFS)
    if len(rows) < 25:
        raise ValueError(f"{BRAIN_DEV_TABLE_S4_CNMFS} is unexpectedly short")

    programs = [str(value).strip() for value in rows[0][1:]]
    enrich_records: list[dict[str, Any]] = []
    # Rows 3-23 hold enrichment scores. The next non-empty section (Row 25)
    # starts with "Program annotation".
    for row in rows[2:]:
        term = str(row[0]).strip() if row else ""
        if term == "Program annotation":
            break
        if not term:
            continue
        for offset, program in enumerate(programs, start=1):
            score = _safe_float(row[offset] if offset < len(row) else "")
            if np.isfinite(score):
                enrich_records.append(
                    {
                        "program": program,
                        "enrichment_term": term,
                        "enrichment_score": float(score),
                    }
                )

    # Row 27 starts the "Ranked genes" section.
    # Each cell in this section has the format "GENE (SCORE)".
    gene_records: list[dict[str, Any]] = []
    rank_start = next(
        (
            idx
            for idx, row in enumerate(rows)
            if row and str(row[0]).strip().startswith("Ranked genes")
        ),
        None,
    )
    if rank_start is None:
        raise ValueError("Table S4 has no ranked-gene section")
    for row in rows[rank_start:]:
        for offset, program in enumerate(programs, start=1):
            parsed = _parse_cnmf_gene_cell(row[offset] if offset < len(row) else "")
            if parsed is None:
                continue
            gene, cnmf_score = parsed
            gene_records.append(
                {
                    "program": program,
                    "gene": gene,
                    "cnmf_score_x1000": cnmf_score,
                }
            )

    return pd.DataFrame(enrich_records), pd.DataFrame(gene_records)


def _read_table_s4_controls() -> pd.DataFrame:
    """Rank Table S4 cNMF genes within cell-type/process control groups."""
    enrich, genes = _read_table_s4_programs()
    manual_stats = (
        genes.groupby("gene", as_index=False)
        .agg(
            mean_score=("cnmf_score_x1000", "mean"),
            max_score=("cnmf_score_x1000", "max"),
            n_programs=("program", "nunique"),
        )
        .assign(
            source_programs="manual_curated",
            source_terms="Manual curated marker",
            max_program_group_score=0.0,
        )
    )
    # Always use manually curated gene list first (also from the original paper)
    manual_rank_by_group = {
        group: {gene: rank for rank, gene in enumerate(dict.fromkeys(markers), start=1)}
        for group, markers in BRAIN_DEV_CELL_TYPE_MANUAL_GENES.items()
    }

    # For each cell-type group, find the top-scoring cNMF programs enriched for that group,
    # then rank genes by their mean score across those programs.
    # Top genes will be added to the positive control gene list.
    records: list[pd.DataFrame] = []
    for control_group, terms in BRAIN_DEV_CELL_TYPE_CONTROL_GROUPS.items():
        group_enrich = enrich[enrich["enrichment_term"].isin(terms)].copy()
        if group_enrich.empty:
            continue
        program_scores = (
            group_enrich.groupby("program", as_index=False)
            .agg(
                program_group_score=("enrichment_score", "max"),
                matched_terms=("enrichment_term", lambda s: "/".join(sorted(set(s)))),
            )
            .sort_values("program_group_score", ascending=False, kind="stable")
            .head(BRAIN_DEV_CELL_TYPE_PROGRAM_TOP_N)
        )
        program_scores = program_scores[program_scores["program_group_score"].gt(0)]
        if program_scores.empty:
            continue
        selected = genes.merge(program_scores, on="program", how="inner")
        ranked = (
            selected.groupby("gene", as_index=False)
            .agg(
                mean_score=("cnmf_score_x1000", "mean"),
                max_score=("cnmf_score_x1000", "max"),
                n_programs=("program", "nunique"),
                source_programs=("program", lambda s: "/".join(sorted(set(s)))),
                source_terms=("matched_terms", lambda s: "/".join(sorted(set(s)))),
                max_program_group_score=("program_group_score", "max"),
            )
            .assign(
                control_group=control_group,
                group_label=BRAIN_DEV_CELL_TYPE_LABELS[control_group],
            )
        )
        manual_rank = manual_rank_by_group.get(control_group, {})
        ranked["manual_rank"] = ranked["gene"].map(manual_rank)
        ranked["manual_control"] = ranked["manual_rank"].notna()
        ranked["manual_rank"] = ranked["manual_rank"].fillna(np.inf)
        missing_manual = [
            gene for gene in manual_rank if gene not in set(ranked["gene"].astype(str))
        ]
        if missing_manual:
            fallback = pd.DataFrame({"gene": missing_manual}).merge(
                manual_stats, on="gene", how="left"
            )
            fallback["source_programs"] = fallback["source_programs"].fillna("manual_curated")
            fallback["source_terms"] = fallback["source_terms"].fillna("Manual curated marker")
            fallback["n_programs"] = fallback["n_programs"].fillna(0).astype(int)
            fallback["max_program_group_score"] = fallback["max_program_group_score"].fillna(0.0)
            fallback["control_group"] = control_group
            fallback["group_label"] = BRAIN_DEV_CELL_TYPE_LABELS[control_group]
            fallback["manual_rank"] = fallback["gene"].map(manual_rank)
            fallback["manual_control"] = True
            ranked = pd.concat([ranked, fallback], ignore_index=True)
        records.append(ranked)
    if not records:
        return pd.DataFrame()

    combined = pd.concat(records, ignore_index=True)
    # Keep each gene in the group where its selected cNMF programs give the
    # highest mean score. This avoids duplicate highlighting in scatter panels.
    best = (
        combined.sort_values(
            [
                "gene",
                "manual_control",
                "manual_rank",
                "mean_score",
                "max_program_group_score",
            ],
            ascending=[True, False, True, False, False],
            kind="stable",
        )
        .groupby("gene", sort=False)
        .head(1)
    )
    best = best.sort_values(
        ["control_group", "manual_control", "manual_rank", "mean_score"],
        ascending=[True, False, True, False],
        kind="stable",
    )
    best["rank_in_group"] = best.groupby("control_group", sort=False).cumcount() + 1
    keep = best["rank_in_group"].le(BRAIN_DEV_CELL_TYPE_CONTROL_TOP_N) | best["manual_control"]
    return best[keep].copy()


def brain_dev_positive_control_table(
    force: bool = False, *, result_dir: Path = RESULT_DIR
) -> pd.DataFrame:
    """Cache paper-derived brain-dev cell-type controls from Table S4."""
    positive_path = result_dir / BRAIN_DEV_POSITIVE_CONTROLS_FILE
    data = {
        "source": source_provenance(__file__),
        "table_s4": path_fingerprint(BRAIN_DEV_TABLE_S4_CNMFS),
    }
    signature = cache_signature(data)
    if not force and cache_entry_is_current(
        result_dir, "brain_dev_controls", signature, outputs=[positive_path]
    ):
        return pd.read_csv(positive_path)

    # Build the positive control table using both a manually curated gene list,
    # and the Table S4, which containsannotated cNMF gene programs
    positive = _read_table_s4_controls()
    positive.insert(0, "panel", "brain_dev")
    positive.insert(1, "source", "Cell 2026 Table S4 cNMF")
    positive.loc[positive["manual_control"].astype(bool), "source"] = "Cell 2026 manual curation"
    save_csv(positive, positive_path)
    update_cache_entry(
        result_dir, "brain_dev_controls", signature, [positive_path], signature_data=data
    )
    return positive


def brain_dev_forced_controls(force: bool = False, *, result_dir: Path = RESULT_DIR) -> list[str]:
    """Return Table S4/manual control genes for panel selection."""
    controls = brain_dev_positive_control_table(force=force, result_dir=result_dir)
    top = controls[
        controls["rank_in_group"].le(BRAIN_DEV_CELL_TYPE_CONTROL_TOP_N) | controls["manual_control"]
    ].copy()
    genes = top.sort_values(["control_group", "rank_in_group"], kind="stable")["gene"].astype(str)
    return list(dict.fromkeys(genes))


def brain_dev_anisotropy_cache_ready(path: Path | None = None) -> bool:
    """Return whether the brain-dev anisotropy cache has the current schema."""
    cache_path = RESULT_DIR / BRAIN_DEV_ANISOTROPY_FILE if path is None else path
    if not cache_has_columns(
        cache_path,
        {
            "panel",
            "sample",
            "group",
            "gene",
            "gene_set",
            "anisotropy",
            "dominant_angle_deg",
        },
    ):
        return False
    try:
        long = pd.read_csv(cache_path, usecols=["panel", "gene_set"])
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False
    return set(long["panel"].astype(str)) == {"brain_dev"} and {
        "marker",
        "non-marker",
    }.issubset(set(long["gene_set"].astype(str)))
