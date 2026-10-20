"""Folding chunks per file: a file's repeated hits become one entry whose score is the sum
of its best chunks, so repetition in the result list turns into a higher rank."""

from ragx.core.grouping import group_by_file
from ragx.core.models import Chunk, ScoredChunk


def _hit(id_: int, path: str, score: float) -> ScoredChunk:
    chunk = Chunk(id=id_, file_path=path, text="x", byte_start=0, byte_end=1, line_start=id_, line_end=id_)
    return ScoredChunk(chunk=chunk, score=score)


def test_file_with_several_hits_folds_and_outranks_single_best_hit():
    ranked = [
        _hit(1, "single.md", 1.0),
        _hit(2, "repeated.md", 0.9),
        _hit(3, "repeated.md", 0.8),
        _hit(4, "other.md", 0.7),
    ]
    files = group_by_file(ranked, top=10)
    assert [f.file_path for f in files] == ["repeated.md", "single.md", "other.md"]
    assert [r.chunk.id for r in files[0].chunks] == [2, 3]
    assert abs(files[0].score - 1.7) < 1e-9


def test_only_best_three_chunks_count_and_show():
    """A long transcript with many weak hits must not win on volume alone."""
    ranked = [_hit(i, "long.md", 0.5) for i in range(1, 11)] + [_hit(20, "short.md", 1.4)]
    ranked.sort(key=lambda r: -r.score)
    files = group_by_file(ranked, top=10)
    long_entry = next(f for f in files if f.file_path == "long.md")
    assert len(long_entry.chunks) == 3
    assert abs(long_entry.score - 1.5) < 1e-9


def test_top_limits_entries_not_chunks():
    ranked = [_hit(i, f"f{i % 5}.md", 1.0 - i / 100) for i in range(30)]
    files = group_by_file(ranked, top=4)
    assert len(files) == 4
    assert len({f.file_path for f in files}) == 4


def test_weak_chunks_below_the_visible_window_do_not_lift_a_file():
    """Graph traversal adds many low-heat chunks; if they counted, a file with one strong
    hit plus weak padding would beat a file with one stronger hit."""
    ranked = [
        _hit(1, "strong.md", 1.0),
        _hit(2, "padded.md", 0.9),
        _hit(3, "third.md", 0.5),
        _hit(4, "padded.md", 0.2),  # below the window once 3 files are seen
    ]
    ranked.insert(3, _hit(5, "fourth.md", 0.3))  # 4th distinct file closes a top=3 window
    files = group_by_file(ranked, top=3)
    assert [f.file_path for f in files] == ["strong.md", "padded.md", "third.md"]
    assert [r.chunk.id for r in files[1].chunks] == [2]
