"""File discovery: walk a corpus root, filter by globs/.gitignore, skip binaries."""

from __future__ import annotations

from pathlib import Path

import pathspec
import xxhash

from ragx.core.config import CONFIG_FILE

MAX_FILE_SIZE = 2 * 1024 * 1024
BINARY_SNIFF_BYTES = 8192
ALWAYS_SKIP_DIRS = {".ragx", ".git", "node_modules", "__pycache__", ".venv", "venv"}


def _is_binary(path: Path) -> bool:
    with path.open("rb") as f:
        chunk = f.read(BINARY_SNIFF_BYTES)
    return b"\x00" in chunk


def _load_gitignore_spec(directory: Path) -> pathspec.GitIgnoreSpec | None:
    gitignore = directory / ".gitignore"
    if not gitignore.is_file():
        return None
    lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
    return pathspec.GitIgnoreSpec.from_lines(lines)


def _is_gitignored(rel: str, specs: dict[str, pathspec.GitIgnoreSpec]) -> bool:
    """Git precedence: the deepest .gitignore with a matching pattern decides.

    `specs` maps a directory's root-relative POSIX path ("" = root) to its .gitignore.
    Directory paths carry a trailing "/" so dir-only patterns (`build/`) match them.
    """
    parts = rel.rstrip("/").split("/")
    for depth in range(len(parts) - 1, -1, -1):
        spec = specs.get("/".join(parts[:depth]))
        if spec is None:
            continue
        verdict = spec.check_file("/".join(parts[depth:]) + ("/" if rel.endswith("/") else ""))
        if verdict.include is not None:
            return verdict.include
    return False


def discover_files(
    root: Path,
    include: list[str],
    exclude: list[str],
    respect_gitignore: bool = True,
) -> list[str]:
    """Return sorted relative POSIX paths of files matching include/exclude under root.

    Always skips ALWAYS_SKIP_DIRS, hidden directories, the corpus's own root-level
    ragx.toml, binaries (NUL byte in first 8KB), and files larger than 2MB. With
    respect_gitignore, every .gitignore in the tree applies to its own subtree (git
    semantics: deeper files override shallower ones; ignored directories are pruned,
    so a negation can't re-include a file under them).
    """
    include_spec = pathspec.PathSpec.from_lines("gitwildmatch", include)
    exclude_spec = pathspec.PathSpec.from_lines("gitwildmatch", exclude) if exclude else None
    gitignore_specs: dict[str, pathspec.GitIgnoreSpec] = {}

    results: list[str] = []
    for dirpath, dirnames, filenames in root.walk():
        dir_rel = dirpath.relative_to(root).as_posix()
        dir_rel = "" if dir_rel == "." else dir_rel
        if respect_gitignore and (spec := _load_gitignore_spec(dirpath)) is not None:
            gitignore_specs[dir_rel] = spec
        dirnames[:] = [
            d
            for d in dirnames
            if d not in ALWAYS_SKIP_DIRS
            and not d.startswith(".")
            and not _is_gitignored(f"{dir_rel}/{d}/".lstrip("/"), gitignore_specs)
        ]
        for name in filenames:
            path = dirpath / name
            rel = path.relative_to(root).as_posix()

            if rel == CONFIG_FILE:
                continue
            if not include_spec.match_file(rel):
                continue
            if exclude_spec is not None and exclude_spec.match_file(rel):
                continue
            if _is_gitignored(rel, gitignore_specs):
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
