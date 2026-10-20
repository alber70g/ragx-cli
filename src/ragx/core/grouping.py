"""Fold ranked chunks into one entry per file."""

from __future__ import annotations

from collections.abc import Iterable

from ragx.core.models import ScoredChunk, ScoredFile

PER_FILE = 3


def group_by_file(ranked: Iterable[ScoredChunk], top: int, per_file: int = PER_FILE) -> list[ScoredFile]:
    """Fold the chunks a flat best-first list would show into one entry per file.

    The window ends where the flat list reaches `top` distinct files: chunks below it never
    count. Each file keeps its `per_file` best chunks inside the window and scores as their
    sum, so a file that repeats in the list outranks one that appears once. Without the
    window, graph traversal's many weak neighbour chunks pad multi-chunk files past strong
    single-chunk files (measured: graph MRR .43 -> .38 unwindowed, .65 windowed).

    `ranked` must be best-first; returns at most `top` files, best-first."""
    by_file: dict[str, list[ScoredChunk]] = {}
    for r in ranked:
        if r.chunk.file_path not in by_file and len(by_file) == top:
            break  # the flat list would show `top` files by now; later chunks never surface
        hits = by_file.setdefault(r.chunk.file_path, [])
        if len(hits) < per_file:
            hits.append(r)
    files = [ScoredFile(path, sum(h.score for h in hits), hits) for path, hits in by_file.items()]
    files.sort(key=lambda f: (-f.score, f.file_path))
    return files[:top]
