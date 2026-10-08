"""Small helpers shared by Q-test analyses and figure notebooks."""

import hashlib
import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from scripts import paper_utils
from scripts.paper_utils import (
    COMPUTER_MODERN_MATHTEXT,
    FIGURE_WIDTH_MM,
    FONT_CONFIG,
    configure_paper_matplotlib,
)

__all__ = ["COMPUTER_MODERN_MATHTEXT", "FIGURE_WIDTH_MM", "FONT_CONFIG"]


def source_provenance(analysis_file: str | Path) -> dict[str, Any]:
    """Fingerprint Q-test Python dependencies, excluding unused R enrichment code."""
    provenance = paper_utils.source_provenance(analysis_file)
    root = Path(analysis_file).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    shared_source = Path(paper_utils.__file__)
    digest.update(shared_source.name.encode())
    digest.update(shared_source.read_bytes())
    provenance["analysis_source_sha256"] = digest.hexdigest()
    return provenance


def cache_is_reusable(
    *outputs: Path,
    force: bool = False,
    inputs: Iterable[Path] = (),
    logger: logging.Logger | None = None,
    label: str = "Cache",
) -> bool:
    """Reuse complete outputs only when they are newer than every input."""
    inputs = tuple(inputs)
    complete = all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    inputs_exist = all(path.exists() for path in inputs)
    fresh = not inputs or (
        inputs_exist
        and complete
        and min(path.stat().st_mtime for path in outputs)
        >= max(path.stat().st_mtime for path in inputs)
    )
    reusable = not force and complete and fresh
    if logger is not None:
        if reusable:
            logger.info("%s: using existing cache", label)
        else:
            invalid_outputs = sum(
                not path.is_file() or path.stat().st_size == 0 for path in outputs
            )
            missing_inputs = sum(not path.exists() for path in inputs)
            if force:
                reason = "--force"
            elif invalid_outputs:
                reason = f"{invalid_outputs}/{len(outputs)} outputs missing or empty"
            elif missing_inputs:
                reason = f"{missing_inputs}/{len(inputs)} inputs missing"
            else:
                reason = "inputs are newer than outputs"
            logger.info("%s: refreshing cache (%s)", label, reason)
    return reusable


def parse_newick(  # noqa: C901
    path: Path,
) -> tuple[str, dict[str, list[str]], dict[str, str], list[str]]:
    """Parse the simple rooted Newick trees used by the lineage analysis."""
    text = path.read_text().strip()
    children: dict[str, list[str]] = defaultdict(list)
    parent: dict[str, str] = {}
    leaves: list[str] = []
    node_counter = position = 0

    def skip_length() -> None:
        nonlocal position
        if position < len(text) and text[position] == ":":
            position += 1
            while position < len(text) and text[position] not in ",);":
                position += 1

    def parse_subtree() -> str:
        nonlocal node_counter, position
        if text[position] == "(":
            position += 1
            node = f"__node_{node_counter}"
            node_counter += 1
            while True:
                child = parse_subtree()
                children[node].append(child)
                parent[child] = node
                if text[position] == ",":
                    position += 1
                    continue
                if text[position] != ")":
                    raise ValueError(f"Malformed Newick near position {position}: {path}")
                position += 1
                while position < len(text) and text[position] not in ":,);":
                    position += 1
                skip_length()
                return node

        start = position
        while position < len(text) and text[position] not in ":,();":
            position += 1
        leaf = text[start:position]
        if not leaf:
            raise ValueError(f"Missing leaf name near position {position}: {path}")
        leaves.append(leaf)
        skip_length()
        return leaf

    root = parse_subtree()
    if position < len(text) and text[position] == ";":
        position += 1
    if position != len(text):
        raise ValueError(f"Unparsed Newick suffix at position {position}: {path}")
    return root, dict(children), parent, leaves


def configure_matplotlib() -> None:
    """Apply the shared compact manuscript style."""
    configure_paper_matplotlib(
        font_family="Arial",
        savefig_dpi=None,
        hide_spines=True,
        overrides={
            "font.sans-serif": ["Arial"],
            "font.serif": ["Arial"],
            "font.monospace": ["Arial"],
            "font.cursive": ["Arial"],
            "font.fantasy": ["Arial"],
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.5,
            "lines.linewidth": 0.75,
            "patch.linewidth": 0.5,
        },
    )
