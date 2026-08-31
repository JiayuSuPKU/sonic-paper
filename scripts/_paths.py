"""Helpers for portable paths stored in analysis caches."""

from pathlib import Path


def resolve_cached_data_path(
    cached_path: str | Path,
    data_root: Path,
    *,
    subdirectory: str = "adata",
) -> Path:
    """Resolve a cached path, rebasing stale absolute paths by filename."""
    path = Path(cached_path).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(data_root / path)
    candidates.append(data_root / subdirectory / path.name)
    return next((p.resolve() for p in candidates if p.exists()), candidates[-1].resolve())
