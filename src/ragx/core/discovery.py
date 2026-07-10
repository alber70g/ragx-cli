"""File discovery: walk a corpus root, filter by globs/.gitignore, skip binaries."""

from __future__ import annotations

from pathlib import Path

import pathspec
import xxhash

MAX_FILE_SIZE = 2 * 1024 * 1024
BINARY_SNIFF_BYTES = 8192
ALWAYS_SKIP_DIRS = {".ragx", ".git", "node_modules", "__pycache__", ".venv", "venv"}


def _is_binary(path: Path) -> bool:
    with path.open("rb") as f:
        chunk = f.read(BINARY_SNIFF_BYTES)
    return b"\x00" in chunk


def _load_gitignore_spec(root: Path) -> pathspec.PathSpec | None:
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return None
    lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def discover_files(
    root: Path,
    include: list[str],
    exclude: list[str],
    respect_gitignore: bool = True,
) -> list[str]:
    """Return sorted relative POSIX paths of files matching include/exclude under root.

    Always skips ALWAYS_SKIP_DIRS, hidden directories, binaries (NUL byte in first 8KB),
    and files larger than 2MB. Only a root-level .gitignore is honored (MVP limitation).
    """
    include_spec = pathspec.PathSpec.from_lines("gitwildmatch", include)
    exclude_spec = pathspec.PathSpec.from_lines("gitwildmatch", exclude) if exclude else None
    gitignore_spec = _load_gitignore_spec(root) if respect_gitignore else None

    results: list[str] = []
    for dirpath, dirnames, filenames in root.walk():
        dirnames[:] = [
            d for d in dirnames if d not in ALWAYS_SKIP_DIRS and not d.startswith(".")
        ]
        for name in filenames:
            path = dirpath / name
            rel = path.relative_to(root).as_posix()

            if not include_spec.match_file(rel):
                continue
            if exclude_spec is not None and exclude_spec.match_file(rel):
                continue
            if gitignore_spec is not None and gitignore_spec.match_file(rel):
                continue

            try:
                if path.stat().st_size > MAX_FILE_SIZE:
                    continue
                if _is_binary(path):
                    continue
            except OSError:
                continue

            results.append(rel)

    return sorted(results)


def hash_file(path: Path) -> str:
    """xxhash.xxh64 hexdigest of the file's raw bytes."""
    hasher = xxhash.xxh64()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()
