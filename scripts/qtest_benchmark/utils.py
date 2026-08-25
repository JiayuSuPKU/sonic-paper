"""Small helpers shared by Q-test analyses and figure notebooks."""

import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

FIGURE_WIDTH_MM = 185.0
FONT_CONFIG = {"title": 7.0, "label": 6.5, "tick": 5.5, "text": 5.0, "legend": 5.5}


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
            missing = sum(not path.exists() for path in outputs)
            reason = (
                "--force"
                if force
                else f"{missing}/{len(outputs)} outputs missing or stale"
            )
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
                    raise ValueError(
                        f"Malformed Newick near position {position}: {path}"
                    )
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
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.titlesize": FONT_CONFIG["title"],
            "axes.labelsize": FONT_CONFIG["label"],
            "xtick.labelsize": FONT_CONFIG["tick"],
            "ytick.labelsize": FONT_CONFIG["tick"],
            "legend.fontsize": FONT_CONFIG["legend"],
            "legend.title_fontsize": FONT_CONFIG["text"],
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.5,
            "lines.linewidth": 0.75,
            "patch.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )
