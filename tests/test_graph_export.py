"""core/graph_export.build_graph: the ragx.graph.v1 payload the web viewer renders."""

from __future__ import annotations

from ragx.core.config import Config, write_default_config
from ragx.core.graph_export import build_graph
from ragx.core.indexer import run_index
from ragx.core.store import Store
from ragx.core.config import db_path

from tests.test_integration import FakeEmbedder, make_corpus


def _subchunk_corpus(tmp_path):
    # mixed.md holds two concepts; dogs.md/rockets.md each hold one. Whole-chunk cosine
    # pools both concepts into one vector, but sub-chunk edges connect mixed.md to both
    # concept files near-exactly — the graph view is only correct when edge_source=subchunk.
    dogs = "Dogs bark loudly at strangers passing. " * 20
    rockets = "Rockets launch toward distant orbit tonight. " * 20
    (tmp_path / "mixed.md").write_text(dogs + rockets)
    (tmp_path / "dogs.md").write_text(dogs)
    (tmp_path / "rockets.md").write_text(rockets)
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    cfg.set("graph.edge_source", "subchunk")
    cfg.set("graph.subchunk_size_tokens", "128")
    emb = FakeEmbedder()
    run_index(tmp_path, cfg, emb)
    return cfg


def test_build_graph_schema_and_meta_counts(tmp_path):
    _subchunk_corpus(tmp_path)
    with Store(db_path(tmp_path)) as store:
        graph = build_graph(tmp_path, store)

    assert graph["schema"] == "ragx.graph.v1"
    meta = graph["meta"]
    assert meta["root"] == tmp_path.name
    assert meta["edge_source"] == "subchunk"
    assert meta["subchunk_size_tokens"] == 128
    assert meta["embedding_model"] == "fake-embedder"
    assert meta["nodes"] == len(graph["nodes"]) == 3
    assert meta["edges"] == len(graph["edges"]) > 0
    assert meta["communities"] >= 1


def test_build_graph_node_fields_and_degree(tmp_path):
    _subchunk_corpus(tmp_path)
    with Store(db_path(tmp_path)) as store:
        graph = build_graph(tmp_path, store)

    by_file = {n["file"]: n for n in graph["nodes"]}
    mixed = by_file["mixed.md"]
    # mixed.md has a sub-chunk edge to both dogs.md and rockets.md
    assert mixed["degree"] >= 2
    assert mixed["label"].startswith("Dogs bark")
    assert isinstance(mixed["community"], int) or mixed["community"] is None

    src, dst, weight = graph["edges"][0]
    assert isinstance(src, int) and isinstance(dst, int)
    assert 0.0 <= weight <= 1.0


def test_build_graph_exports_chunk_mode_index_with_whole_chunk_edges(tmp_path):
    # Index built with edge_source="chunk": no sub-chunks exist, so the graph must render
    # the stored whole-chunk-cosine edges instead of refusing.
    make_corpus(tmp_path)
    (tmp_path / "pets2.md").write_text("# More pets\n\nDogs bark at cats. Cats purr at dogs.\n")
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    cfg.set("graph.edge_source", "chunk")
    cfg.set("graph.min_edge_sim", "0.3")  # fake embeddings are coarse
    stats = run_index(tmp_path, cfg, FakeEmbedder())
    assert stats.edges_total > 0

    with Store(db_path(tmp_path)) as store:
        graph = build_graph(tmp_path, store)
        assert graph["meta"]["edge_source"] == "chunk"
        assert graph["meta"]["subchunk_size_tokens"] is None
        assert graph["edges"] == [
            [src, dst, round(weight, 4)] for src, dst, weight in store.all_edges()
        ]
