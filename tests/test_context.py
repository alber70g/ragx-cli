"""Grep-like -B/-A context around a chunk: N counts non-empty lines, but blank lines
inside the collected span are kept (grep's actual -B/-A behavior), and the span starts/ends
exactly at the Nth non-empty line — blanks beyond it never leak in."""

from __future__ import annotations

from pathlib import Path

from ragx.core.context import attach_context, context_span
from ragx.core.models import Chunk, ScoredChunk, ScoredFile

LINES = [
    "one",  # 1
    "",  # 2 (blank)
    "two",  # 3
    "three",  # 4
    "",  # 5 (blank)
    "",  # 6 (blank)
    "four",  # 7  <- chunk start
    "five",  # 8  <- chunk end
    "",  # 9 (blank)
    "six",  # 10
    "seven",  # 11
]


def test_before_counts_non_empty_lines_but_keeps_blanks_inside_the_span():
    # chunk at line 7: walking up, "three"(4) and "two"(3) are the 2 nearest non-empty lines;
    # the blank lines 5-6 sit between them and the chunk and must be kept.
    before, _ = context_span(LINES, line_start=7, line_end=8, before=2, after=0)
    assert before == {"line_start": 3, "line_end": 6, "text": "two\nthree\n\n"}


def test_before_excludes_blank_lines_above_the_nth_non_empty_line():
    # before=1 should start AT "three" (line 4), not include the blank line 2 or "one".
    before, _ = context_span(LINES, line_start=7, line_end=8, before=1, after=0)
    assert before == {"line_start": 4, "line_end": 6, "text": "three\n\n"}


def test_after_mirrors_before():
    after_span = context_span(LINES, line_start=7, line_end=8, before=0, after=2)[1]
    assert after_span == {"line_start": 9, "line_end": 11, "text": "\nsix\nseven"}


def test_n_zero_returns_none():
    before, after = context_span(LINES, line_start=7, line_end=8, before=0, after=0)
    assert before is None
    assert after is None


def test_fewer_than_n_non_empty_lines_returns_what_exists():
    # only "one" exists above line 3; before=5 must not go out of bounds.
    before, _ = context_span(LINES, line_start=3, line_end=3, before=5, after=0)
    assert before == {"line_start": 1, "line_end": 2, "text": "one\n"}


def test_chunk_at_file_start_has_no_before_context():
    before, _ = context_span(LINES, line_start=1, line_end=1, before=3, after=0)
    assert before is None


def test_chunk_at_file_end_has_no_after_context():
    _, after = context_span(LINES, line_start=11, line_end=11, before=0, after=3)
    assert after is None


def test_no_non_empty_lines_on_either_side_returns_none():
    only_blanks = ["", "", "x", "", ""]
    before, after = context_span(only_blanks, line_start=3, line_end=3, before=3, after=3)
    assert before is None
    assert after is None


def _hit(id_: int, path: str, line_start: int, line_end: int) -> ScoredChunk:
    chunk = Chunk(
        id=id_, file_path=path, text="x", byte_start=0, byte_end=1,
        line_start=line_start, line_end=line_end,
    )
    return ScoredChunk(chunk=chunk, score=1.0)


def test_attach_context_reads_the_real_file_and_fills_in_spans(tmp_path):
    (tmp_path / "doc.md").write_text("\n".join(LINES) + "\n")
    hit = _hit(1, "doc.md", line_start=7, line_end=8)
    results = [ScoredFile("doc.md", 1.0, [hit])]

    attach_context(tmp_path, results, before=2, after=2)

    assert hit.context["before"] == {"line_start": 3, "line_end": 6, "text": "two\nthree\n\n"}
    assert hit.context["after"] == {"line_start": 9, "line_end": 11, "text": "\nsix\nseven"}


def test_attach_context_skips_entirely_when_both_sides_are_zero(tmp_path):
    (tmp_path / "doc.md").write_text("\n".join(LINES) + "\n")
    hit = _hit(1, "doc.md", line_start=7, line_end=8)
    results = [ScoredFile("doc.md", 1.0, [hit])]

    attach_context(tmp_path, results, before=0, after=0)

    assert hit.context is None


def test_attach_context_nulls_out_on_a_missing_file_instead_of_raising(tmp_path):
    hit = _hit(1, "gone.md", line_start=1, line_end=1)
    results = [ScoredFile("gone.md", 1.0, [hit])]

    attach_context(Path(tmp_path), results, before=3, after=3)

    assert hit.context == {"before": None, "after": None}
