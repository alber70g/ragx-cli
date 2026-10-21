"""Static file resolution for GET / and GET /static/<path>, with path-traversal rejection."""

from __future__ import annotations

from pathlib import Path

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}


def resolve(static_root: Path, rel_path: str) -> Path | None:
    """The file `rel_path` resolves to under `static_root`, or None (traversal, missing,
    not a file) — never raises."""
    root = static_root.resolve()
    try:
        target = (root / rel_path).resolve()
    except (OSError, ValueError):
        return None
    if not target.is_relative_to(root) or not target.is_file():
        return None
    return target


def content_type(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix, "application/octet-stream")
