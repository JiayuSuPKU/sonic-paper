"""Small helpers shared by the Q-test cache-producing analyses."""

from pathlib import Path


def cache_is_reusable(*outputs: Path, force: bool = False) -> bool:
    """Return whether every expected output exists and recomputation is disabled."""
    return not force and all(path.exists() for path in outputs)
