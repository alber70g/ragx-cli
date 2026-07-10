"""Text chunking: markdown/code/recursive splitters producing byte-exact ChunkDraft slices."""

from __future__ import annotations

import re
from bisect import bisect_right
from pathlib import Path

from ragx.core.models import ChunkDraft

MARKDOWN_EXTS = {".md", ".markdown"}
CODE_EXTS = {".py", ".ts", ".js", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb"}
SEPARATORS = ["\n\n", "\n", ". ", " "]

_HEADING_RE = re.compile(r"^#{1,6} .*$", re.MULTILINE)
_CODE_BOUNDARY_RE = re.compile(r"^(?:def|class|function|fn|func)\b", re.MULTILINE)


def _find_boundary(s: str, start: int, hard_end: int) -> int:
    region = s[start:hard_end]
    for sep in SEPARATORS:
        idx = region.rfind(sep)
        if idx != -1:
            candidate = start + idx + len(sep)
            if candidate > start:
                return candidate
    return hard_end


def _recursive_split_range(s: str, target_chars: int, hard_max_chars: int, overlap: float) -> list[tuple[int, int]]:
    n = len(s)
    if n == 0:
        return []
    ranges: list[tuple[int, int]] = []
    pos = 0
    while pos < n:
        remaining = n - pos
        if remaining <= hard_max_chars:
            end = n
        else:
            hard_end = min(pos + hard_max_chars, n)
            end = _find_boundary(s, pos, hard_end)
        ranges.append((pos, end))
        if end >= n:
            break
        chunk_len = end - pos
        overlap_chars = int(round(chunk_len * overlap))
        next_pos = end - overlap_chars
        if next_pos <= pos:
            next_pos = end
        pos = next_pos
    return ranges


def _recursive_split(
    text: str, start: int, end: int, target_chars: int, hard_max_chars: int, overlap: float
) -> list[tuple[int, int]]:
    rel = _recursive_split_range(text[start:end], target_chars, hard_max_chars, overlap)
    return [(start + a, start + b) for a, b in rel]


def _sections_from_boundaries(text: str, starts: list[int]) -> list[tuple[int, int]]:
    sections: list[tuple[int, int]] = []
    if not starts:
        return [(0, len(text))]
    if starts[0] > 0:
        sections.append((0, starts[0]))
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        sections.append((s, e))
    return sections


def _markdown_split(text: str, target_chars: int, hard_max_chars: int, overlap: float) -> list[tuple[int, int]]:
    starts = [m.start() for m in _HEADING_RE.finditer(text)]
    sections = _sections_from_boundaries(text, starts)

    ranges: list[tuple[int, int]] = []
    buffer: tuple[int, int] | None = None
    for s, e in sections:
        size = e - s
        if size > hard_max_chars:
            if buffer is not None:
                ranges.append(buffer)
                buffer = None
            ranges.extend(_recursive_split(text, s, e, target_chars, hard_max_chars, overlap))
            continue
        if buffer is None:
            buffer = (s, e)
        elif (buffer[1] - buffer[0]) + size <= target_chars:
            buffer = (buffer[0], e)
        else:
            ranges.append(buffer)
            buffer = (s, e)
    if buffer is not None:
        ranges.append(buffer)
    return ranges


def _code_split(text: str, target_chars: int, hard_max_chars: int, overlap: float) -> list[tuple[int, int]]:
    starts = [m.start() for m in _CODE_BOUNDARY_RE.finditer(text)]
    sections = _sections_from_boundaries(text, starts)

    ranges: list[tuple[int, int]] = []
    for s, e in sections:
        if (e - s) > hard_max_chars:
            ranges.extend(_recursive_split(text, s, e, target_chars, hard_max_chars, overlap))
        else:
            ranges.append((s, e))
    return ranges


def _char_byte_offsets(text: str) -> list[int]:
    offsets = [0] * (len(text) + 1)
    total = 0
    for i, ch in enumerate(text):
        offsets[i] = total
        total += len(ch.encode("utf-8"))
    offsets[len(text)] = total
    return offsets


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def chunk_text(text: str, path: str, size_tokens: int = 800, overlap: float = 0.15) -> list[ChunkDraft]:
    """Split `text` into byte-exact ChunkDrafts, dispatching on `path`'s extension."""
    if not text.strip():
        return []

    target_chars = size_tokens * 4
    hard_max_chars = int(round(target_chars * 1.5))

    ext = Path(path).suffix.lower()
    if ext in MARKDOWN_EXTS:
        ranges = _markdown_split(text, target_chars, hard_max_chars, overlap)
    elif ext in CODE_EXTS:
        ranges = _code_split(text, target_chars, hard_max_chars, overlap)
    else:
        ranges = _recursive_split(text, 0, len(text), target_chars, hard_max_chars, overlap)

    char_to_byte = _char_byte_offsets(text)
    line_starts = _line_starts(text)

    drafts: list[ChunkDraft] = []
    for start, end in ranges:
        if end <= start:
            continue
        chunk_str = text[start:end]
        if not chunk_str.strip():
            continue
        drafts.append(
            ChunkDraft(
                text=chunk_str,
                byte_start=char_to_byte[start],
                byte_end=char_to_byte[end],
                line_start=bisect_right(line_starts, start),
                line_end=bisect_right(line_starts, end - 1),
            )
        )
    return drafts
