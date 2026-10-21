"""core/reedge.rebuild_edges: `ragx-cli index --edges-only` — rebuilds sub-chunks, graph edges
and communities from stored chunks without re-chunking or re-embedding files."""

from __future__ import annotations

import pytest

from ragx.core.config import Config, db_path, write_default_config
from ragx.core.errors import RagxError
from ragx.core.graph_export import build_graph
from ragx.core.indexer import run_index
from ragx.core.reedge import rebuild_edges
from ragx.core.store import Store

from tests.test_integration import FakeEmbedder, make_corpus


class CountingEmbedder(FakeEmbedder):
    """Records every batch of texts passed to embed_documents."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return super().embed_documents(texts)


def _write_concept_corpus(root):
    # mixed.md is one chunk holding two concepts and is long enough to split into
    # sub-chunks; dogs.md/rockets.md each hold one concept and stay whole.
    dogs = "Dogs bark loudly at strangers passing. " * 20
    rockets = "Rockets launch toward distant orbit tonight. " * 20
    (root / "mixed.md").write_text(dogs + rockets)
    (root / "dogs.md").write_text(dogs)
    (root / "rockets.md").write_text(rockets)


def test_rebuild_edges_subchunk_matches_full_index_and_reuses_chunk_vectors(tmp_path):
    root_a = tmp_path / "a"
    root_a.mkdir()
    _write_concept_corpus(root_a)
    write_default_config(root_a)
    cfg_a = Config.load(root_a)
    cfg_a.set("graph.edge_source", "chunk")
    emb_a = CountingEmbedder()
    run_index(root_a, cfg_a, emb_a)  # built with edge_source="chunk": no sub-chunks yet

    with Store(db_path(root_a)) as store:
        dogs_text = store.get_chunks(store.chunk_ids_for_file("dogs.md"))[0].text
        rockets_text = store.get_chunks(store.chunk_ids_for_file("rockets.md"))[0].text

    cfg_a.set("graph.edge_source", "subchunk")
    emb_a.calls.clear()
    stats = rebuild_edges(root_a, cfg_a, emb_a)
    assert stats.edges_total > 0
    assert stats.communities_total >= 1
    assert stats.files_indexed == 0 and stats.chunks_added == 0  # not touched by this path

    with Store(db_path(root_a)) as store:
        all_ids = store.all_chunk_ids()
        parents = {cid for cid, _ in store.all_subchunk_vectors()}
        assert parents == set(all_ids)  # every chunk has at least one sub-chunk row
        assert store.get_meta("edge_source") == "subchunk"
        edges_a = store.all_edges()
        graph = build_graph(root_a, store)
    assert graph["meta"]["edge_source"] == "subchunk"  # web view now works without re-indexing

    # dogs.md/rockets.md stayed whole chunks: their full chunk text never went through
    # embed_documents again — only mixed.md's sub-chunk pieces (or nothing) did.
    flat_calls = [t for call in emb_a.calls for t in call]
    assert dogs_text not in flat_calls
    assert rockets_text not in flat_calls

    # equivalence: a --full subchunk index of the same corpus produces the same edges
    root_b = tmp_path / "b"
    root_b.mkdir()
    _write_concept_corpus(root_b)
    write_default_config(root_b)
    cfg_b = Config.load(root_b)
    cfg_b.set("graph.edge_source", "subchunk")
    run_index(root_b, cfg_b, FakeEmbedder())
    with Store(db_path(root_b)) as store:
        edges_b = store.all_edges()

    assert [(s, d) for s, d, _ in edges_a] == [(s, d) for s, d, _ in edges_b]
    for (_, _, wa), (_, _, wb) in zip(edges_a, edges_b):
        assert abs(wa - wb) < 1e-6


def test_rebuild_edges_embedding_model_mismatch_raises(tmp_path):
    make_corpus(tmp_path)
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    run_index(tmp_path, cfg, FakeEmbedder())

    class OtherEmbedder(FakeEmbedder):
        model = "other-embedder"

    with pytest.raises(RagxError):
        rebuild_edges(tmp_path, cfg, OtherEmbedder())


def test_rebuild_edges_reverse_flip_clears_subchunks(tmp_path):
    make_corpus(tmp_path)
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    cfg.set("graph.edge_source", "subchunk")
    emb = FakeEmbedder()
    run_index(tmp_path, cfg, emb)
    with Store(db_path(tmp_path)) as store:
        assert store.all_subchunk_vectors()

    cfg.set("graph.edge_source", "chunk")
    stats = rebuild_edges(tmp_path, cfg, emb)

    with Store(db_path(tmp_path)) as store:
        assert store.all_subchunk_vectors() == []
        assert store.get_meta("edge_source") == "chunk"
        assert stats.edges_total == store.edge_count()


def test_rebuild_edges_fails_loud_when_no_index_exists(tmp_path):
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    with pytest.raises(RagxError, match="ragx-cli index"):
        rebuild_edges(tmp_path, cfg, FakeEmbedder())
