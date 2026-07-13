"""End-to-end: init -> index -> query on a tmp corpus with a deterministic fake embedder."""

from __future__ import annotations

import hashlib
import math

import pytest

from ragx.core.config import Config, db_path, write_default_config
from ragx.core.errors import ManifestMismatchError
from ragx.core.store import Store
from ragx.core.indexer import run_index
from ragx.core.query import QueryOptions, run_query, to_files_json, to_query_json


class FakeEmbedder:
    """Deterministic bag-of-words-ish embedding so related texts land near each other."""

    model = "fake-embedder"

    def dimension(self) -> int:
        return 64

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * 64
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % 64] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed_one(t) for t in texts]

    def embed_queries(self, texts):
        return [self._embed_one(t) for t in texts]


def make_corpus(root):
    (root / "pets.md").write_text("# Pets\n\nDogs bark loudly. Cats purr softly.\n")
    (root / "space.md").write_text("# Space\n\nRockets launch to orbit. Planets circle stars.\n")
    (root / "cooking.md").write_text("# Cooking\n\nPasta boils in salted water. Sauce simmers.\n")


def test_index_then_query(tmp_path):
    make_corpus(tmp_path)
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    emb = FakeEmbedder()

    stats = run_index(tmp_path, cfg, emb)
    assert stats.files_indexed == 3
    assert stats.chunks_added >= 3

    out = run_query(tmp_path, cfg, emb, "dogs bark cats", QueryOptions(top=3))
    assert out.results
    assert out.results[0].chunk.file_path == "pets.md"

    doc = to_query_json(out)
    assert doc["schema"] == "ragx.query.v1"
    r = doc["results"][0]
    assert set(r) >= {"chunk_id", "file", "line_start", "byte_start", "score", "breakdown", "text"}

    files = to_files_json(out)
    assert files["schema"] == "ragx.files.v1"
    assert files["files"][0]["file"] == "pets.md"


def test_incremental_reindex(tmp_path):
    make_corpus(tmp_path)
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    emb = FakeEmbedder()
    run_index(tmp_path, cfg, emb)

    # unchanged corpus: nothing re-processed
    stats = run_index(tmp_path, cfg, emb, changed_only=True)
    assert stats.files_indexed == 0
    assert stats.files_unchanged == 3

    # edit one file, delete another
    (tmp_path / "pets.md").write_text("# Pets\n\nDogs bark. Hamsters run on wheels.\n")
    (tmp_path / "space.md").unlink()
    stats = run_index(tmp_path, cfg, emb, changed_only=True)
    assert stats.files_indexed == 1
    assert stats.files_deleted == 1

    out = run_query(tmp_path, cfg, emb, "hamsters wheels", QueryOptions(top=3))
    assert out.results[0].chunk.file_path == "pets.md"
    assert all(r.chunk.file_path != "space.md" for r in out.results)


def test_graph_edges_and_traversal(tmp_path):
    make_corpus(tmp_path)
    (tmp_path / "pets2.md").write_text("# More pets\n\nDogs bark at cats. Cats purr at dogs.\n")
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    cfg.set("graph.min_edge_sim", "0.3")  # fake embeddings are coarse
    emb = FakeEmbedder()
    stats = run_index(tmp_path, cfg, emb)
    assert stats.edges_total > 0

    out = run_query(
        tmp_path, cfg, emb, "dogs bark cats",
        QueryOptions(top=4, expand=False, rerank=False, explain=True),
    )
    assert out.results
    files = {r.chunk.file_path for r in out.results}
    assert {"pets.md", "pets2.md"} <= files
    # explain traces exist and parent-chains terminate at a seed (hop 0)
    assert all(r.explain is not None for r in out.results)
    assert any(r.explain["hop"] == 0 for r in out.results)


def test_subchunk_edges_end_to_end(tmp_path):
    # mixed.md is one chunk holding two concepts; each concept also has its own file.
    # Sub-chunk edges must connect mixed.md to BOTH concept files at near-exact weight,
    # which a pooled whole-chunk cosine (~0.7 to each) cannot express.
    dogs = "Dogs bark loudly at strangers passing. " * 20
    rockets = "Rockets launch toward distant orbit tonight. " * 20
    (tmp_path / "mixed.md").write_text(dogs + rockets)
    (tmp_path / "dogs.md").write_text(dogs)
    (tmp_path / "rockets.md").write_text(rockets)
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    cfg.set("graph.edge_source", "subchunk")
    emb = FakeEmbedder()

    stats = run_index(tmp_path, cfg, emb)
    assert stats.edges_total > 0

    with Store(db_path(tmp_path)) as store:
        assert len(store.all_subchunk_vectors()) > store.chunk_count()  # mixed.md split
        [mid] = store.chunk_ids_for_file("mixed.md")
        [did] = store.chunk_ids_for_file("dogs.md")
        [rid] = store.chunk_ids_for_file("rockets.md")
        weights = dict(store.neighbors(mid))
        assert weights[did] > 0.95  # concept-level edge: pooled cosine is only ~0.7
        assert weights[rid] > 0.95

    out = run_query(tmp_path, cfg, emb, "dogs bark strangers", QueryOptions(top=3, expand=False, rerank=False))
    assert out.results[0].chunk.file_path in ("dogs.md", "mixed.md")


def test_subchunk_incremental_reindex(tmp_path):
    make_corpus(tmp_path)
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    cfg.set("graph.edge_source", "subchunk")
    cfg.set("graph.min_edge_sim", "0.3")
    emb = FakeEmbedder()
    run_index(tmp_path, cfg, emb)

    (tmp_path / "pets.md").write_text("# Pets\n\nDogs bark. Hamsters run on wheels.\n")
    (tmp_path / "space.md").unlink()
    stats = run_index(tmp_path, cfg, emb, changed_only=True)
    assert stats.files_indexed == 1
    assert stats.files_deleted == 1

    with Store(db_path(tmp_path)) as store:
        # deleted file's sub-chunk vectors cascaded away with its chunks
        parents = {cid for cid, _ in store.all_subchunk_vectors()}
        assert parents == set(store.all_chunk_ids())
    out = run_query(tmp_path, cfg, emb, "hamsters wheels", QueryOptions(top=3))
    assert out.results[0].chunk.file_path == "pets.md"


def test_changed_only_fails_loud_on_edge_source_flip(tmp_path):
    make_corpus(tmp_path)
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    emb = FakeEmbedder()
    run_index(tmp_path, cfg, emb)  # built with edge_source=chunk

    cfg.set("graph.edge_source", "subchunk")
    with pytest.raises(ManifestMismatchError):
        run_index(tmp_path, cfg, emb, changed_only=True)


def test_index_persists_communities(tmp_path):
    make_corpus(tmp_path)
    (tmp_path / "pets2.md").write_text("# More pets\n\nDogs bark at cats. Cats purr at dogs.\n")
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    cfg.set("graph.min_edge_sim", "0.3")  # fake embeddings are coarse
    emb = FakeEmbedder()

    stats = run_index(tmp_path, cfg, emb)
    with Store(db_path(tmp_path)) as store:
        assert stats.communities_total == store.community_count() >= 1
        edged_chunk_ids = {src for src, _, _ in store.all_edges()} | {
            dst for _, dst, _ in store.all_edges()
        }
        members = set()
        for cid, _ in store.community_sizes():
            members.update(store.community_members(cid))
        assert edged_chunk_ids <= members


def test_changed_reindex_recomputes_communities(tmp_path):
    make_corpus(tmp_path)
    (tmp_path / "pets2.md").write_text("# More pets\n\nDogs bark at cats. Cats purr at dogs.\n")
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    cfg.set("graph.min_edge_sim", "0.3")
    emb = FakeEmbedder()
    run_index(tmp_path, cfg, emb)

    with Store(db_path(tmp_path)) as store:
        deleted_ids = set(store.chunk_ids_for_file("space.md"))
    (tmp_path / "space.md").unlink()
    run_index(tmp_path, cfg, emb, changed_only=True)

    with Store(db_path(tmp_path)) as store:
        members = set()
        for cid, _ in store.community_sizes():
            members.update(store.community_members(cid))
        assert not (deleted_ids & members)
        edged_chunk_ids = {src for src, _, _ in store.all_edges()} | {
            dst for _, dst, _ in store.all_edges()
        }
        assert edged_chunk_ids <= members
