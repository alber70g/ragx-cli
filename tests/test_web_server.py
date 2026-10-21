"""Web server handlers: start on port 0 in a thread, hit it with httpx. No network beyond
localhost loopback, no LM Studio (autouse conftest fixture stubs ensure_for_section)."""

from __future__ import annotations

import threading

import httpx
import pytest

from ragx.core.config import Config, db_path, write_default_config
from ragx.core.indexer import run_index
from ragx.core.store import Store
from ragx.web import static_files
from ragx.web.server import build_server

from tests.test_integration import FakeEmbedder


class MismatchedEmbedder(FakeEmbedder):
    """Same vectors, different `.model` — triggers run_query's manifest-mismatch RagxError."""

    model = "mismatched-embedder"


@pytest.fixture
def running_server(tmp_path):
    (tmp_path / "pets.md").write_text("# Pets\n\nDogs bark loudly. Cats purr softly.\n")
    (tmp_path / "space.md").write_text("# Space\n\nRockets launch to orbit.\n")
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    cfg.set("graph.edge_source", "subchunk")
    run_index(tmp_path, cfg, FakeEmbedder())

    static_root = tmp_path / "no_static_here"  # deliberately absent: serves 404
    server = build_server(tmp_path, cfg, host="127.0.0.1", port=0, static_root=static_root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield server, f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_api_graph_returns_schema_and_counts(running_server):
    _, base_url = running_server
    resp = httpx.get(f"{base_url}/api/graph")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json; charset=utf-8"
    doc = resp.json()
    assert doc["schema"] == "ragx.graph.v1"
    assert doc["meta"]["nodes"] == len(doc["nodes"])


def test_api_graph_gzip_encodes_the_same_payload(running_server):
    _, base_url = running_server
    plain = httpx.get(f"{base_url}/api/graph").json()
    resp = httpx.get(f"{base_url}/api/graph", headers={"Accept-Encoding": "gzip"})
    assert resp.headers.get("content-encoding") == "gzip"
    assert resp.json() == plain  # httpx transparently decodes gzip; content must still match


def test_api_chunk_200_then_404_for_unknown_id(running_server):
    server, base_url = running_server
    with Store(db_path(server.state.root)) as store:
        chunk_id = store.all_chunk_ids()[0]

    resp = httpx.get(f"{base_url}/api/chunk/{chunk_id}")
    assert resp.status_code == 200
    doc = resp.json()
    assert doc["chunk_id"] == chunk_id
    assert "neighbors" in doc and "text" in doc

    resp = httpx.get(f"{base_url}/api/chunk/999999")
    assert resp.status_code == 404
    assert "error" in resp.json()


def test_static_root_missing_serves_404_at_slash(running_server):
    # static/ doesn't exist in this fixture (frontend owns it) — GET / must 404, not crash.
    _, base_url = running_server
    resp = httpx.get(f"{base_url}/")
    assert resp.status_code == 404


def test_static_serves_a_real_file_under_the_root(tmp_path, running_server):
    server, base_url = running_server
    static_root = tmp_path / "static_root"
    static_root.mkdir()
    (static_root / "index.html").write_text("<html>ok</html>")
    server.state.static_root = static_root

    resp = httpx.get(f"{base_url}/static/index.html")
    assert resp.status_code == 200
    assert resp.text == "<html>ok</html>"


def test_static_path_traversal_rejected_even_when_target_exists_outside_root(tmp_path):
    # unit-level, not over HTTP: httpx collapses ".." client-side before sending the
    # request, which would hide a server-side traversal bug behind client normalization.
    static_root = tmp_path / "static_root"
    static_root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve me")

    assert static_files.resolve(static_root, "../secret.txt") is None
    assert static_files.resolve(static_root, "nonexistent.html") is None


def test_post_query_200_with_fake_embedder(running_server):
    server, base_url = running_server
    server.state.query_runner._embedder = FakeEmbedder()

    resp = httpx.post(f"{base_url}/api/query", json={"text": "dogs bark cats"})
    assert resp.status_code == 200
    doc = resp.json()
    assert doc["schema"] == "ragx.query.v3"
    assert isinstance(doc["elapsed_ms"], int)
    assert doc["results"][0]["file"] == "pets.md"


def test_post_query_400_on_invalid_json_body(running_server):
    _, base_url = running_server
    resp = httpx.post(f"{base_url}/api/query", content=b"{not json", headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_post_query_400_on_missing_text(running_server):
    _, base_url = running_server
    resp = httpx.post(f"{base_url}/api/query", json={})
    assert resp.status_code == 400


def test_post_query_400_on_negative_top(running_server):
    server, base_url = running_server
    server.state.query_runner._embedder = FakeEmbedder()
    resp = httpx.post(f"{base_url}/api/query", json={"text": "dogs", "top": -1})
    assert resp.status_code == 400


def test_post_query_400_on_wrong_type_for_expand(running_server):
    _, base_url = running_server
    resp = httpx.post(f"{base_url}/api/query", json={"text": "dogs", "expand": "yes"})
    assert resp.status_code == 400


def test_post_query_502_on_ragx_error_from_embedding_model_mismatch(running_server):
    server, base_url = running_server
    # index was built with FakeEmbedder ("fake-embedder"); a differently-named embedder
    # makes run_query's manifest check raise RagxError — the API must surface it as 502,
    # not crash the server or return a generic 500.
    server.state.query_runner._embedder = MismatchedEmbedder()

    resp = httpx.post(f"{base_url}/api/query", json={"text": "dogs"})
    assert resp.status_code == 502
    assert "error" in resp.json()

    # the server itself keeps running for later requests
    resp = httpx.get(f"{base_url}/api/graph")
    assert resp.status_code == 200
