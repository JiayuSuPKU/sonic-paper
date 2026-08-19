"""Build and load the strict human-mouse 1:1 orthology cache.

Pairs come from the MGI ``HOM_MouseHumanSequence.rpt`` report. An explicit
refresh adds protein-coding annotations from local OrgDb packages and writes
``results/embryo_xspecies/comparison/human_mouse_orthologs.tsv``. Analysis
code only reads that cache; it never downloads or repairs it automatically.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

from . import config as C

LOGGER = logging.getLogger("embryo_xspecies.orthology")

ORTHO_TSV = C.COMPARISON_RESULT_DIR / "human_mouse_orthologs.tsv"
MGI_HOM_URL = (
    "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"
)

_HUMAN = "human"
_MOUSE = "mouse, laboratory"
_CACHE_COLUMNS = (
    "human_symbol",
    "mouse_symbol",
    "human_protein_coding",
    "mouse_protein_coding",
)


def _protein_coding_symbols(genes: pd.Series, orgdb: str) -> set[str]:
    """Return symbols annotated as protein-coding by a local OrgDb package."""
    rscript = Path(sys.prefix) / "bin" / "Rscript"
    if not rscript.exists():
        raise FileNotFoundError(f"Missing Rscript in active environment: {rscript}")
    code = """
args <- commandArgs(trailingOnly = TRUE)
suppressPackageStartupMessages(library(args[[1]], character.only = TRUE))
db <- get(args[[1]])
genes <- read.csv(args[[2]], stringsAsFactors = FALSE)$gene
mapped <- suppressMessages(AnnotationDbi::select(
  db, keys = unique(genes), keytype = "SYMBOL", columns = c("SYMBOL", "GENETYPE")
))
mapped <- unique(mapped[!is.na(mapped$GENETYPE) & mapped$GENETYPE == "protein-coding", "SYMBOL", drop = FALSE])
write.csv(mapped, args[[3]], row.names = FALSE)
"""
    with tempfile.TemporaryDirectory(prefix="embryo_orthology_") as temporary:
        root = Path(temporary)
        source = root / "genes.csv"
        output = root / "protein_coding.csv"
        pd.DataFrame({"gene": genes}).to_csv(source, index=False)
        subprocess.run(
            [str(rscript), "-e", code, orgdb, str(source), str(output)],
            cwd=C.PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return set(pd.read_csv(output)["SYMBOL"].astype(str))


def refresh_orthologs() -> pd.DataFrame:
    """Download MGI, build strict 1:1 pairs, annotate, and replace the cache."""
    LOGGER.info("Downloading MGI human-mouse homology report.")
    with urllib.request.urlopen(MGI_HOM_URL, timeout=60) as response:  # noqa: S310
        source = pd.read_csv(response, sep="\t", dtype=str).fillna("")

    key = "DB Class Key"
    organism = "Common Organism Name"
    symbol = "Symbol"
    rows: list[dict[str, str]] = []
    for _, group in source.groupby(key):
        human = group.loc[group[organism].eq(_HUMAN), symbol].tolist()
        mouse = group.loc[group[organism].eq(_MOUSE), symbol].tolist()
        if len(human) == 1 and len(mouse) == 1 and human[0] and mouse[0]:
            rows.append({"human_symbol": human[0], "mouse_symbol": mouse[0]})

    pairs = pd.DataFrame(rows).drop_duplicates()
    pairs = pairs[
        ~pairs["human_symbol"].duplicated(keep=False)
        & ~pairs["mouse_symbol"].duplicated(keep=False)
    ].sort_values("human_symbol", ignore_index=True)
    human_coding = _protein_coding_symbols(pairs["human_symbol"], "org.Hs.eg.db")
    mouse_coding = _protein_coding_symbols(pairs["mouse_symbol"], "org.Mm.eg.db")
    if not human_coding or not mouse_coding:
        raise RuntimeError("OrgDb returned no protein-coding ortholog annotations.")
    pairs["human_protein_coding"] = pairs["human_symbol"].isin(human_coding)
    pairs["mouse_protein_coding"] = pairs["mouse_symbol"].isin(mouse_coding)

    ORTHO_TSV.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(ORTHO_TSV, sep="\t", index=False)
    LOGGER.info("Wrote %s (%d strict 1:1 pairs).", ORTHO_TSV, len(pairs))
    return pairs


def load_orthologs() -> pd.DataFrame:
    """Load and validate the cached strict 1:1 ortholog table."""
    if not ORTHO_TSV.exists():
        raise FileNotFoundError(
            f"Missing {ORTHO_TSV}; run `python -m scripts.embryo_xspecies.orthology`."
        )
    table = pd.read_csv(ORTHO_TSV, sep="\t", dtype=str)
    if tuple(table.columns) != _CACHE_COLUMNS:
        raise ValueError(
            f"{ORTHO_TSV} columns must be {list(_CACHE_COLUMNS)}; rerun "
            "`python -m scripts.embryo_xspecies.orthology`."
        )
    if table.empty or table[list(_CACHE_COLUMNS)].isna().any().any():
        raise ValueError(f"{ORTHO_TSV} is empty or contains missing values.")
    for column in ("human_protein_coding", "mouse_protein_coding"):
        values = table[column].str.lower()
        if not values.isin({"true", "false"}).all():
            raise ValueError(f"{ORTHO_TSV} has invalid values in {column}.")
        table[column] = values.eq("true")
    if (
        table["human_symbol"].duplicated().any()
        or table["mouse_symbol"].duplicated().any()
    ):
        raise ValueError(f"{ORTHO_TSV} is not strict 1:1; refresh the cache.")
    return table


def prepare_orthologs(*, force: bool = False) -> pd.DataFrame:
    """Load the validated cache, downloading and rebuilding it when required."""
    if force or not ORTHO_TSV.exists():
        return refresh_orthologs()
    table = load_orthologs()
    LOGGER.info("Orthology cache is current: %s (%d pairs).", ORTHO_TSV, len(table))
    return table


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    refresh_orthologs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
