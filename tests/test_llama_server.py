"""llama-server providers: HTTP scoring/embedding (respx-mocked), construction guards,
registry routing."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from ragx.core.config import Config, write_default_config
from ragx.core.errors import RagxError
from ragx.providers.llama_embedder import LlamaServerEmbedder
from ragx.providers.llama_process import MAX_BATCH_TOKENS, LlamaServerProcess
from ragx.providers.llama_server import LlamaServerReranker
from ragx.providers.registry import make_embedder, make_reranker

BASE = "http://127.0.0.1:9814"
DEAD = "http://127.0.0.1:59999"  # nothing listens here; connection refused instantly


def _mock_health(ok: bool = True) -> None:
    route = respx.get(f"{BASE}/health")
    if ok:
        route.mock(return_value=httpx.Response(200, json={"status": "ok"}))
    else:
        route.mock(side_effect=httpx.ConnectError("down"))


@respx.mock
def test_score_maps_results_back_by_index():
    _mock_health()
    respx.post(f"{BASE}/v1/rerank").mock(
        return_value=httpx.Response(
            200,
            json={"results": [  # server returns them sorted by relevance, not input order
                {"index": 2, "relevance_score": 3.5},
                {"index": 0, "relevance_score": 1.5},
                {"index": 1, "relevance_score": -2.0},
            ]},
        )
    )
    rr = LlamaServerReranker(base_url=BASE)
    assert rr.score("q", ["a", "b", "c"]) == [1.5, -2.0, 3.5]


@respx.mock
def test_score_raises_ragx_error_on_http_failure():
    _mock_health()
    respx.post(f"{BASE}/v1/rerank").mock(return_value=httpx.Response(503, text="loading"))
    rr = LlamaServerReranker(base_url=BASE)
    with pytest.raises(RagxError, match="rerank request failed"):
        rr.score("q", ["a"])


def test_construction_fails_without_server_or_gguf():
    with pytest.raises(RagxError, match="rerank.gguf is not set"):
        LlamaServerReranker(base_url=DEAD)


def test_construction_fails_on_missing_gguf(tmp_path):
    with pytest.raises(RagxError, match="does not exist"):
        LlamaServerReranker(base_url=DEAD, gguf=str(tmp_path / "nope.gguf"))


def test_construction_fails_on_missing_binary(tmp_path, monkeypatch):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(RagxError, match="llama-server binary not found"):
        LlamaServerReranker(base_url=DEAD, gguf=str(gguf), server_bin="llama-server-nope")


def test_registry_degrades_to_none_when_unavailable(tmp_path, capsys):
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    cfg.set("rerank.provider", "llama-server")
    cfg.set("rerank.base_url", DEAD)
    assert make_reranker(cfg) is None
    assert "reranker unavailable" in capsys.readouterr().err


def test_registry_rejects_unknown_provider(tmp_path, capsys):
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    cfg.set("rerank.provider", "bogus")
    assert make_reranker(cfg) is None
    assert "unknown rerank provider" in capsys.readouterr().err


@respx.mock
def test_registry_builds_llama_reranker_when_reachable(tmp_path):
    _mock_health()
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    cfg.set("rerank.provider", "llama-server")
    rr = make_reranker(cfg)
    assert isinstance(rr, LlamaServerReranker)


# --- embedder ----------------------------------------------------------------

EMBED_BASE = "http://127.0.0.1:9813"


@respx.mock
def test_embedder_serves_prefixed_embeddings():
    respx.get(f"{EMBED_BASE}/health").mock(return_value=httpx.Response(200))
    route = respx.post(f"{EMBED_BASE}/v1/embeddings").mock(
        return_value=httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]}
        )
    )
    emb = LlamaServerEmbedder(
        base_url=f"{EMBED_BASE}/v1", gguf="", model="m", query_prefix="Query: "
    )
    assert emb.embed_queries(["hello"]) == [[1.0, 0.0]]
    assert json.loads(route.calls[0].request.content)["input"] == ["Query: hello"]
    assert emb.dimension() == 2


def test_embedder_construction_fails_loud_when_unspawnable():
    with pytest.raises(RagxError, match="embeddings.gguf is not set"):
        LlamaServerEmbedder(base_url="http://127.0.0.1:59999/v1", gguf="", model="m")


@respx.mock
def test_registry_swaps_default_base_url_for_llama_embedder(tmp_path):
    respx.get(f"{EMBED_BASE}/health").mock(return_value=httpx.Response(200))
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    cfg.set("embeddings.provider", "llama-server")  # base_url stays the LM Studio default
    emb = make_embedder(cfg)
    assert isinstance(emb, LlamaServerEmbedder)
    assert emb._proc.root_url == EMBED_BASE


def test_spawn_passes_a_batch_large_enough_for_real_chunks(tmp_path, monkeypatch):
    """Regression: llama.cpp's default 512-token physical batch cannot hold a ~800-token
    chunk, and a non-causal rerank/embedding input cannot be split across ubatches — the
    server answers 500 on every real query without these flags."""
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x")
    recorded: dict = {}

    class FakeProc:
        """Alive until terminated — _spawn treats an exited process as a load failure."""

        def __init__(self):
            self.rc = None

        def poll(self):
            return self.rc

        def terminate(self):
            self.rc = 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, **kwargs):
        recorded["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr("shutil.which", lambda _: "/fake/llama-server")
    monkeypatch.setattr("subprocess.Popen", fake_popen)
    proc = LlamaServerProcess(
        "http://127.0.0.1:9814", str(gguf), "llama-server", mode="--rerank", section="rerank"
    )
    monkeypatch.setattr(proc, "healthy", lambda: True)  # "came up" right after spawn
    proc._spawn()

    cmd = recorded["cmd"]
    batch = str(MAX_BATCH_TOKENS)
    assert MAX_BATCH_TOKENS >= 2048  # a chunk plus its query must fit in one ubatch
    for flag, value in (("-b", batch), ("-ub", batch), ("-c", "0"), ("--parallel", "1")):
        assert cmd[cmd.index(flag) + 1] == value, f"{flag} missing or wrong: {cmd}"
