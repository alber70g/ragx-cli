"""Grep-like -B/-A/-C context lines around a result chunk, read fresh from the source file
at query time (never stored, never truncated by `query.max_chunk_chars`)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

from ragx.core.models import ScoredFile

log = logging.getLogger("ragx.context")


class Span(TypedDict):
    line_start: int  # 1-based, inclusive
    line_end: int
    text: str


def context_span(
    lines: list[str], line_start: int, line_end: int, before: int, after: int
) -> tuple[Span | None, Span | None]:
    """Context around a chunk's 1-based inclusive `[line_start, line_end]`.

    Walks outward from the line just above `line_start` / just below `line_end`, collecting
    lines until `before`/`after` NON-EMPTY lines have been seen or the file edge is reached.
    Blank lines inside the collected span are kept; blank lines above the Nth non-empty
    line (before side) or below it (after side) are not. `before`/`after` <= 0, or no lines
    on that side, yields `None` for it.
    """
    before_span = _collect_before(lines, line_start - 2, before)  # 0-based index above the chunk
    after_span = _collect_after(lines, line_end, after)  # 0-based index below the chunk
    return before_span, after_span


def _collect_before(lines: list[str], start_idx: int, n: int) -> Span | None:
    if n <= 0:
        return None
    non_empty = [i for i in range(start_idx, -1, -1) if lines[i].strip() != ""]
    if not non_empty:
        return None
    span_start = non_empty[:n][-1]  # the Nth non-empty line walking upward (or the last one)
    return _make_span(lines, span_start, start_idx)


def _collect_after(lines: list[str], start_idx: int, n: int) -> Span | None:
    if n <= 0:
        return None
    non_empty = [i for i in range(start_idx, len(lines)) if lines[i].strip() != ""]
    if not non_empty:
        return None
    span_end = non_empty[:n][-1]
    return _make_span(lines, start_idx, span_end)


def _make_span(lines: list[str], start_idx: int, end_idx: int) -> Span:
    return {
        "line_start": start_idx + 1,
        "line_end": end_idx + 1,
        "text": "\n".join(lines[start_idx : end_idx + 1]),
    }


def attach_context(root: Path, results: list[ScoredFile], before: int, after: int) -> None:
    """Attach grep-style context to every chunk in `results` (the final, grouped output —
    never the full candidate set). Reads each distinct file at most once. A file that can't
    be read (e.g. deleted since indexing) gets `{"before": None, "after": None}` on its
    chunks, with a warning logged instead of raising."""
    if before <= 0 and after <= 0:
        return
    lines_cache: dict[str, list[str] | None] = {}
    for entry in results:
        for r in entry.chunks:
            path = r.chunk.file_path
            if path not in lines_cache:
                lines_cache[path] = _read_lines(root, path)
            lines = lines_cache[path]
            if lines is None:
                r.context = {"before": None, "after": None}
                continue
            before_span, after_span = context_span(lines, r.chunk.line_start, r.chunk.line_end, before, after)
            r.context = {"before": before_span, "after": after_span}


def _read_lines(root: Path, file_path: str) -> list[str] | None:
    try:
        return (root / file_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        log.warning("context: could not read %s: %s", file_path, exc)
        return None
