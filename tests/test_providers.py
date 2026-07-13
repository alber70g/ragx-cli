from __future__ import annotations

import httpx
import pytest
import respx

from ragx.core.config import Config, DEFAULTS
from ragx.core.errors import RagxError
from ragx.providers.openai_compat import OpenAICompatEmbedder, OpenAICompatGenerator
from ragx.providers.registry import make_embedder, make_generator, make_reranker


def _embedding_response(n: int) -> dict:
    return {"data": [{"embedding": [float(i), float(i) + 0.5]} for i in range(n)]}


@respx.mock
def test_embed_documents_applies_prefix_and_preserves_order():
    route = respx.post("http://x/embeddings").mock(
        return_value=httpx.Response(200, json=_embedding_response(2))
    )
    emb = OpenAICompatEmbedder("http://x", "m", doc_prefix="doc: ", batch_size=32)
    result = emb.embed_documents(["a", "b"])
    assert result == [[0.0, 0.5], [1.0, 1.5]]
    sent = route.calls[0].request
    import json

    payload = json.loads(sent.content)
    assert payload["input"] == ["doc: a", "doc: b"]
    assert payload["model"] == "m"


@respx.mock
def test_embed_queries_applies_query_prefix():
    respx.post("http://x/embeddings").mock(return_value=httpx.Response(200, json=_embedding_response(1)))
    emb = OpenAICompatEmbedder("http://x", "m", query_prefix="q: ")
    emb.embed_queries(["hello"])
    import json

    payload = json.loads(respx.calls[0].request.content)
    assert payload["input"] == ["q: hello"]


@respx.mock
def test_empty_input_makes_no_http_call():
    route = respx.post("http://x/embeddings").mock(return_value=httpx.Response(200, json=_embedding_response(0)))
    emb = OpenAICompatEmbedder("http://x", "m")
    assert emb.embed_documents([]) == []
    assert emb.embed_queries([]) == []
    assert route.call_count == 0


@respx.mock
def test_batching_splits_requests_and_preserves_order():
    calls = {"n": 0}

    def responder(request):
        import json

        body = json.loads(request.content)
        n = len(body["input"])
        start = calls["n"]
        calls["n"] += n
        return httpx.Response(
            200, json={"data": [{"embedding": [float(start + i)]} for i in range(n)]}
        )

    route = respx.post("http://x/embeddings").mock(side_effect=responder)
    emb = OpenAICompatEmbedder("http://x", "m", batch_size=2)
    texts = ["t0", "t1", "t2", "t3", "t4"]
    result = emb.embed_documents(texts)
    assert route.call_count == 3
    assert result == [[0.0], [1.0], [2.0], [3.0], [4.0]]


@respx.mock
def test_retry_on_500_then_success():
    responses = [httpx.Response(500, text="boom"), httpx.Response(200, json=_embedding_response(1))]
    route = respx.post("http://x/embeddings").mock(side_effect=responses)
    emb = OpenAICompatEmbedder("http://x", "m")
    result = emb.embed_documents(["a"])
    assert result == [[0.0, 0.5]]
    assert route.call_count == 2


@respx.mock
def test_retry_exhausted_raises_ragx_error():
    route = respx.post("http://x/embeddings").mock(return_value=httpx.Response(500, text="boom"))
    emb = OpenAICompatEmbedder("http://x", "m")
    with pytest.raises(RagxError):
        emb.embed_documents(["a"])
    assert route.call_count == 3


@respx.mock
def test_dimension_caches_after_one_probe_call():
    route = respx.post("http://x/embeddings").mock(return_value=httpx.Response(200, json=_embedding_response(1)))
    emb = OpenAICompatEmbedder("http://x", "m")
    assert emb.dimension() == 2
    assert emb.dimension() == 2
    assert route.call_count == 1


@respx.mock
def test_generator_payload_shape():
    route = respx.post("http://x/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "hi there"}}]}
        )
    )
    gen = OpenAICompatGenerator("http://x", "m")
    out = gen.generate("sys prompt", "user prompt")
    assert out == "hi there"
    import json

    payload = json.loads(route.calls[0].request.content)
    assert payload["model"] == "m"
    assert payload["temperature"] == 0.3
    assert payload["messages"] == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "user prompt"},
    ]


def _config(overrides: dict | None = None) -> Config:
    data = {k: dict(v) for k, v in DEFAULTS.items()}
    for section, values in (overrides or {}).items():
        data.setdefault(section, {}).update(values)
    return Config(data)


def test_make_embedder_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cfg = _config()
    emb = make_embedder(cfg)
    assert isinstance(emb, OpenAICompatEmbedder)
    assert emb.base_url == DEFAULTS["embeddings"]["base_url"]


def test_make_embedder_ollama_defaults_base_url():
    cfg = _config({"embeddings": {"provider": "ollama"}})
    emb = make_embedder(cfg)
    assert isinstance(emb, OpenAICompatEmbedder)
    assert emb.base_url == "http://localhost:11434/v1"


def test_make_embedder_unknown_provider_raises():
    cfg = _config({"embeddings": {"provider": "bogus"}})
    with pytest.raises(RagxError):
        make_embedder(cfg)


def test_make_generator_disabled_returns_none():
    cfg = _config({"expansion": {"enabled": False}})
    assert make_generator(cfg) is None


def test_make_generator_enabled_returns_instance():
    cfg = _config()
    gen = make_generator(cfg)
    assert isinstance(gen, OpenAICompatGenerator)


def test_make_reranker_disabled_returns_none():
    cfg = _config({"rerank": {"enabled": False}})
    assert make_reranker(cfg) is None


def test_make_reranker_missing_extra_returns_none_and_warns(capsys):
    try:
        import sentence_transformers  # noqa: F401

        pytest.skip("sentence-transformers installed; missing-extra path not reachable")
    except ImportError:
        pass
    cfg = _config({"rerank": {"enabled": True, "model": "does-not-matter"}})
    result = make_reranker(cfg)
    assert result is None
    captured = capsys.readouterr()
    assert "rerank" in captured.err.lower() or "sentence-transformers" in captured.err.lower()


def test_make_reranker_model_load_failure_returns_none_and_warns(monkeypatch, capsys):
    """A download/network failure inside CrossEncoder degrades to no-rerank with a hint."""
    import sys
    import types

    class _FailingCrossEncoder:
        def __init__(self, model):
            raise OSError("Connection to huggingface.co failed")

    fake = types.ModuleType("sentence_transformers")
    fake.CrossEncoder = _FailingCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    cfg = _config({"rerank": {"enabled": True, "model": "BAAI/bge-reranker-v2-m3"}})
    result = make_reranker(cfg)
    assert result is None
    err = capsys.readouterr().err
    assert "huggingface.co" in err
    assert "HF_ENDPOINT" in err


def test_api_key_env_resolved_and_sent(monkeypatch):
    monkeypatch.setenv("RAGX_TEST_KEY", "sk-test-123")
    cfg = _config({"embeddings": {"api_key_env": "RAGX_TEST_KEY"}})
    with respx.mock(base_url="http://localhost:1234/v1") as mock:
        route = mock.post("/embeddings").respond(
            200, json={"data": [{"embedding": [0.1, 0.2]}]}
        )
        make_embedder(cfg).embed_documents(["x"])
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-test-123"


def test_api_key_env_missing_fails_loud(monkeypatch):
    monkeypatch.delenv("RAGX_NO_SUCH_KEY", raising=False)
    cfg = _config({"embeddings": {"api_key_env": "RAGX_NO_SUCH_KEY"}})
    with pytest.raises(RagxError, match="RAGX_NO_SUCH_KEY"):
        make_embedder(cfg)


def test_openai_base_url_env_applies_to_untouched_default(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1/")
    assert make_embedder(_config()).base_url.rstrip("/") == "https://api.example.com/v1"


def test_openai_base_url_env_loses_to_explicit_config(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    cfg = _config({"embeddings": {"base_url": "http://myhost:9999/v1"}})
    assert make_embedder(cfg).base_url == "http://myhost:9999/v1"


def test_openai_api_key_env_fallback_sends_header(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient")
    with respx.mock(base_url="http://localhost:1234/v1") as mock:
        route = mock.post("/embeddings").respond(200, json={"data": [{"embedding": [0.1]}]})
        make_embedder(_config()).embed_documents(["x"])
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-ambient"


def test_api_key_env_config_beats_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient")
    monkeypatch.setenv("MY_KEY", "sk-explicit")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cfg = _config({"embeddings": {"api_key_env": "MY_KEY"}})
    with respx.mock(base_url="http://localhost:1234/v1") as mock:
        route = mock.post("/embeddings").respond(200, json={"data": [{"embedding": [0.1]}]})
        make_embedder(cfg).embed_documents(["x"])
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-explicit"
