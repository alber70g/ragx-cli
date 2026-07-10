from __future__ import annotations

from pathlib import Path

import pytest

from ragx.core.models import ChunkDraft, FileRecord
from ragx.core.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    with Store(tmp_path / "index.db") as s:
        yield s


def make_file(path: str, chunk_count: int = 0) -> FileRecord:
    return FileRecord(path=path, content_hash="deadbeef", mtime=1.0, chunk_count=chunk_count)


def make_draft(text: str = "hello", start: int = 0) -> ChunkDraft:
    return ChunkDraft(
        text=text, byte_start=start, byte_end=start + len(text), line_start=1, line_end=1
    )


def test_meta_roundtrip(store: Store):
    assert store.get_meta("model") is None
    store.set_meta("model", "nomic")
    assert store.get_meta("model") == "nomic"
    store.set_meta("model", "other")
    assert store.get_meta("model") == "other"


def test_file_roundtrip(store: Store):
    assert store.file_count() == 0
    store.upsert_file(make_file("a.py"))
    assert store.file_count() == 1
    assert store.get_file_hashes() == {"a.py": "deadbeef"}
    store.upsert_file(FileRecord(path="a.py", content_hash="feedface", mtime=2.0, chunk_count=1))
    assert store.file_count() == 1
    assert store.get_file_hashes() == {"a.py": "feedface"}


def test_chunk_roundtrip(store: Store):
    store.upsert_file(make_file("a.py"))
    ids = store.insert_chunks("a.py", [make_draft("one"), make_draft("two", start=3)])
    assert len(ids) == 2
    chunks = store.get_chunks(ids)
    assert [c.text for c in chunks] == ["one", "two"]
    assert store.chunk_ids_for_file("a.py") == ids
    assert store.all_chunk_ids() == ids
    assert store.chunk_count() == 2


def test_get_chunks_preserves_order_and_skips_unknown(store: Store):
    store.upsert_file(make_file("a.py"))
    ids = store.insert_chunks("a.py", [make_draft("one"), make_draft("two", start=3)])
    reversed_with_unknown = [ids[1], 9999, ids[0]]
    chunks = store.get_chunks(reversed_with_unknown)
    assert [c.text for c in chunks] == ["two", "one"]


def test_delete_file_cascades_chunks_and_edges(store: Store):
    store.upsert_file(make_file("a.py"))
    store.upsert_file(make_file("b.py"))
    a_ids = store.insert_chunks("a.py", [make_draft("one"), make_draft("two", start=3)])
    b_ids = store.insert_chunks("b.py", [make_draft("three")])

    # edges: a[0]-b[0], a[1]-b[0]
    store.replace_edges(a_ids[0], [(b_ids[0], 0.9)])
    store.replace_edges(a_ids[1], [(b_ids[0], 0.5)])
    assert store.edge_count() == 2

    removed = store.delete_file("a.py")
    assert sorted(removed) == sorted(a_ids)
    assert store.file_count() == 1
    assert store.chunk_count() == 1
    assert store.get_chunks(a_ids) == []
    # edges touching removed chunks (both src and dst side) are gone
    assert store.edge_count() == 0
    assert store.neighbors(b_ids[0]) == []


def test_id_monotonicity_after_delete_and_reinsert(store: Store):
    store.upsert_file(make_file("a.py"))
    first_ids = store.insert_chunks("a.py", [make_draft("one")])
    store.delete_file("a.py")
    store.upsert_file(make_file("a.py"))
    second_ids = store.insert_chunks("a.py", [make_draft("one-again")])
    assert second_ids[0] > first_ids[0]


def test_replace_edges_normalization_and_dedup(store: Store):
    store.upsert_file(make_file("a.py"))
    ids = store.insert_chunks("a.py", [make_draft("one"), make_draft("two", start=3),
                                        make_draft("three", start=6)])
    c3, c5 = ids[0], ids[1]  # use as stand-ins for ids 3 and 5 style test
    # simulate src=5, dst=3 style insertion regardless of actual numeric ids
    hi, lo = max(c3, c5), min(c3, c5)
    store.replace_edges(hi, [(lo, 0.75)])
    assert store.neighbors(lo) == [(hi, 0.75)]
    assert store.neighbors(hi) == [(lo, 0.75)]

    # replace_edges again on the same src clears old neighbors first
    store.replace_edges(hi, [(lo, 0.1)])
    assert store.neighbors(hi) == [(lo, 0.1)]


def test_neighbors_sorted_weight_desc(store: Store):
    store.upsert_file(make_file("a.py"))
    ids = store.insert_chunks(
        "a.py",
        [make_draft("one"), make_draft("two", start=3), make_draft("three", start=6)],
    )
    store.replace_edges(ids[0], [(ids[1], 0.2), (ids[2], 0.8)])
    assert store.neighbors(ids[0]) == [(ids[2], 0.8), (ids[1], 0.2)]


def test_delete_edges_incident(store: Store):
    store.upsert_file(make_file("a.py"))
    ids = store.insert_chunks(
        "a.py",
        [make_draft("one"), make_draft("two", start=3), make_draft("three", start=6)],
    )
    store.replace_edges(ids[0], [(ids[1], 0.5)])
    store.replace_edges(ids[2], [(ids[1], 0.4)])
    assert store.edge_count() == 2
    store.delete_edges_incident([ids[1]])
    assert store.edge_count() == 0

    # unknown ids -> no-op, no error
    store.delete_edges_incident([12345])
    assert store.edge_count() == 0


def test_context_manager_closes(tmp_path: Path):
    path = tmp_path / "index.db"
    with Store(path) as s:
        s.set_meta("k", "v")
    with Store(path) as s2:
        assert s2.get_meta("k") == "v"
