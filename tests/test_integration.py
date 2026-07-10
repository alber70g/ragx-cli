"""End-to-end: init -> index -> query on a tmp corpus with a deterministic fake embedder."""

from __future__ import annotations

import hashlib
import math

from ragx.core.config import Config, write_default_config
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
