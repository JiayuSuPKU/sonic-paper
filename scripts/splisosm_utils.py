"""Provenance shared by the two SPLISOSM paper analyses."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from scripts import paper_utils
from scripts.paper_utils import _python_source_fingerprint, installed_version


def _source_fingerprint(content: bytes) -> str:
    """Ignore cosmetic edits and log wording, retaining evaluated expressions."""
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "log"
            and node.args
        ):
            message = node.args[0]
            if isinstance(message, ast.Constant) and isinstance(message.value, str):
                node.args[0] = ast.Constant(value="<log message>")
            elif isinstance(message, ast.JoinedStr):
                message.values = [
                    part for part in message.values if not isinstance(part, ast.Constant)
                ]
    return _python_source_fingerprint(ast.unparse(tree).encode())


def _shared_source_fingerprint(content: bytes, names: set[str]) -> str:
    """Include imported shared helpers and their transitive module dependencies."""
    tree = ast.parse(content)
    bindings: dict[str, set[int]] = {}
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound = {node.name}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound = {alias.asname or alias.name.split(".")[0] for alias in node.names}
        else:
            bound = {
                child.id
                for child in ast.walk(node)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
            }
        for name in bound:
            bindings.setdefault(name, set()).add(index)
    pending = set(names)
    selected: set[int] = set()
    while pending:
        for index in bindings.get(pending.pop(), set()) - selected:
            selected.add(index)
            pending.update(
                child.id for child in ast.walk(tree.body[index]) if isinstance(child, ast.Name)
            )
    tree.body = [node for index, node in enumerate(tree.body) if index in selected]
    return _source_fingerprint(ast.unparse(tree).encode())


def source_provenance(analysis_file: str | Path) -> dict:
    """Fingerprint SPLISOSM and relevant analysis code without depending on SONIC."""
    import splisosm

    root = next(p for p in Path(analysis_file).resolve().parents if p.name.startswith("splisosm_"))
    provenance = {"fingerprint_schema": "splisosm-python-ast-v1"}
    for name, directory in (("splisosm", Path(splisosm.__file__).parent), ("analysis", root)):
        digest = hashlib.sha256()
        paths = sorted(directory.rglob("*.py"))
        if name == "analysis":
            paths.append(Path(__file__))
        shared_names: set[str] = set()
        for path in paths:
            content = path.read_bytes()
            digest.update(
                (
                    str(path.relative_to(directory))
                    if path.is_relative_to(directory)
                    else path.name
                ).encode()
            )
            digest.update(_source_fingerprint(content).encode())
            if name == "analysis":
                for node in ast.walk(ast.parse(content)):
                    if isinstance(node, ast.ImportFrom) and node.module == "scripts.paper_utils":
                        if any(alias.name == "*" for alias in node.names):
                            raise ValueError(
                                "Use explicit paper_utils imports for cache provenance"
                            )
                        shared_names.update(alias.name for alias in node.names)
        if name == "analysis":
            digest.update(
                _shared_source_fingerprint(
                    Path(paper_utils.__file__).read_bytes(), shared_names
                ).encode()
            )
        provenance[f"{name}_source_sha256"] = digest.hexdigest()
    provenance["versions"] = {
        name: installed_version(name)
        for name in ("splisosm", "numpy", "scipy", "finufft", "pandas", "anndata", "spatialdata")
    }
    return provenance
