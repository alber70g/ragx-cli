"""machine_probe: the reuse ranking `init` preselects (pure, per rule), and the live
probes behind it (fake `lms` script, respx /props, fake HF cache)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from ragx.core.catalog import EMBEDDINGS, RERANKER
from ragx.core.lmstudio import InstalledModel
from ragx.core.machine_probe import MachineState, hf_cached, pick_reuse, probe_machine

ROOT = Path("/lms/models")


def _m(key: str, path: str, type_: str = "embedding") -> InstalledModel:
    return InstalledModel(model_key=key, type=type_, format="gguf", path=path, size_bytes=1)


BGE_M3 = _m("text-embedding-bge-m3", "gpustack/bge-m3-GGUF/bge-m3-Q8_0.gguf")
QWEN3 = _m("text-embedding-qwen3-embedding-0.6b", "Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf")
GEMMA_MLX = _m("embeddinggemma-300m", "lmstudio-community/embeddinggemma-300m-qat-mlx")
JINA = _m("jina-v5-nano", "jinaai/x/v5-nano-retrieval-Q8_0.gguf", type_="llm")  # LM Studio misclassifies it
NOMIC = _m("text-embedding-nomic-embed-text-v1.5@q4_k_m", "nomic-ai/nomic-embed-text-v1.5-GGUF/n.gguf")
RERANK_GGUF = _m("bge-reranker-v2-m3", "gpustack/bge-reranker-v2-m3-GGUF/bge-reranker-v2-m3-Q8_0.gguf", "llm")


def _state(**kw) -> MachineState:
    return MachineState(models_root=ROOT, **kw)


# --- embeddings ranking ------------------------------------------------------


def test_nothing_found_is_none():
    assert pick_reuse(_state()) is None


def test_non_catalog_model_is_never_auto_picked():
    # its doc/query prefixes are unknown; guessing them corrupts the index silently
    state = _state(lms_loaded=(NOMIC,), lms_installed=(NOMIC,), embed_server_gguf="/x/nomic.gguf")
    assert pick_reuse(state) is None


def test_rule1_loaded_in_lm_studio_wins_over_running_server_and_downloads():
    state = _state(
        lms_loaded=(QWEN3,), lms_installed=(BGE_M3, QWEN3), llama_on_path=True,
        embed_server_gguf="/x/bge-m3-Q8_0.gguf",
    )
    emb = pick_reuse(state).embeddings
    assert emb.reason == "loaded in LM Studio"
    assert emb.values["embeddings.provider"] == "openai"
    assert emb.values["embeddings.model"] == QWEN3.model_key
    assert emb.values["embeddings.query_prefix"] == EMBEDDINGS["best"].query_prefix


def test_rule2_running_llama_server_reuses_its_gguf_and_lms_key():
    gguf = str(ROOT / BGE_M3.path)
    emb = pick_reuse(_state(embed_server_gguf=gguf, lms_installed=(BGE_M3,))).embeddings
    assert emb.reason == "served by a running llama-server"
    assert emb.values["embeddings.provider"] == "llama-server"
    assert emb.values["embeddings.gguf"] == gguf
    assert emb.values["embeddings.base_url"] == "http://127.0.0.1:9813/v1"
    # same model name `ragx-cli models` writes for that GGUF, so corpora agree on the manifest
    assert emb.values["embeddings.model"] == BGE_M3.model_key


def test_rule2_gguf_outside_lm_studio_uses_file_stem_as_model():
    emb = pick_reuse(_state(embed_server_gguf="/elsewhere/Qwen3-Embedding-0.6B-Q8_0.gguf")).embeddings
    assert emb.values["embeddings.model"] == "Qwen3-Embedding-0.6B-Q8_0"


def test_rule3_downloaded_tier_order_balanced_best_fast():
    state = _state(lms_installed=(QWEN3, BGE_M3))
    assert pick_reuse(state).embeddings.label == EMBEDDINGS["balanced"].label
    state = _state(lms_installed=(GEMMA_MLX, QWEN3))
    assert pick_reuse(state).embeddings.label == EMBEDDINGS["best"].label


def test_rule3_engine_llama_server_when_on_path_and_gguf_present():
    emb = pick_reuse(_state(lms_installed=(BGE_M3,), llama_on_path=True)).embeddings
    assert emb.values["embeddings.provider"] == "llama-server"
    assert emb.values["embeddings.gguf"] == str(ROOT / BGE_M3.path)

    emb = pick_reuse(_state(lms_installed=(BGE_M3,), llama_on_path=False)).embeddings
    assert emb.values["embeddings.provider"] == "openai"
    assert "embeddings.gguf" not in emb.values


def test_rule3_mlx_download_falls_back_to_lm_studio_engine():
    # llama-server cannot load an MLX directory, even with the binary on PATH
    emb = pick_reuse(_state(lms_installed=(GEMMA_MLX,), llama_on_path=True)).embeddings
    assert emb.values["embeddings.provider"] == "openai"


def test_rule3_jina_needs_llama_server():
    assert pick_reuse(_state(lms_installed=(JINA,), llama_on_path=False)) is None
    emb = pick_reuse(_state(lms_installed=(JINA,), llama_on_path=True)).embeddings
    assert emb.values["embeddings.provider"] == "llama-server"


# --- rerank ranking ----------------------------------------------------------


def test_rerank_rule1_running_server():
    rr = pick_reuse(_state(rerank_server_gguf="/x/bge-reranker-v2-m3-Q8_0.gguf")).rerank
    assert rr.reason == "served by a running llama-server"
    assert rr.values == {"rerank.provider": "llama-server", "rerank.model": RERANKER.model,
                         "rerank.gguf": "/x/bge-reranker-v2-m3-Q8_0.gguf"}


def test_rerank_rule2_downloaded_gguf_needs_llama_server_on_path():
    assert pick_reuse(_state(lms_installed=(RERANK_GGUF,))) is None
    rr = pick_reuse(_state(lms_installed=(RERANK_GGUF,), llama_on_path=True)).rerank
    assert rr.values["rerank.gguf"] == str(ROOT / RERANK_GGUF.path)


def test_rerank_rule3_hf_cache():
    choice = pick_reuse(_state(reranker_cached=True))
    assert choice.embeddings is None
    assert choice.rerank.values == {"rerank.provider": "sentence-transformers",
                                    "rerank.model": RERANKER.model}


def test_rerank_gguf_does_not_count_as_embedding():
    # "bge-m3" must not match the reranker GGUF "bge-reranker-v2-m3"
    choice = pick_reuse(_state(lms_installed=(RERANK_GGUF,), llama_on_path=True,
                               embed_server_gguf="/x/bge-reranker-v2-m3-Q8_0.gguf"))
    assert choice.embeddings is None


# --- live probes -------------------------------------------------------------


def _fake_lms(tmp_path, ls: list[dict], ps: list[dict], fail: bool = False) -> str:
    script = tmp_path / "lms"
    script.write_text(
        "#!/bin/sh\n"
        + ("exit 3\n" if fail else "")
        + f"if [ \"$1\" = ls ]; then echo '{json.dumps(ls)}'; exit 0; fi\n"
        f"if [ \"$1\" = ps ]; then echo '{json.dumps(ps)}'; exit 0; fi\n"
        "exit 9\n"
    )
    script.chmod(0o755)
    return str(script)


@respx.mock
def test_probe_machine_reads_lms_and_props(tmp_path):
    entry = {"modelKey": "text-embedding-bge-m3", "type": "embedding", "format": "gguf",
             "path": "gpustack/bge-m3-GGUF/bge-m3-Q8_0.gguf", "sizeBytes": 6}
    lms = _fake_lms(tmp_path, ls=[entry], ps=[entry])
    respx.get("http://127.0.0.1:9813/props").mock(side_effect=httpx.ConnectError("down"))
    respx.get("http://127.0.0.1:9814/props").mock(
        return_value=httpx.Response(200, json={"model_path": "/r/bge-reranker-v2-m3-Q8_0.gguf"})
    )
    state = probe_machine(find_lms=lambda: lms, which=lambda _: None,
                          reranker_cached=lambda repo: False)
    assert [m.model_key for m in state.lms_loaded] == ["text-embedding-bge-m3"]
    assert [m.model_key for m in state.lms_installed] == ["text-embedding-bge-m3"]
    assert state.embed_server_gguf is None
    assert state.rerank_server_gguf == "/r/bge-reranker-v2-m3-Q8_0.gguf"
    assert state.llama_on_path is False


def test_probe_machine_failing_lms_is_nothing_found(tmp_path):
    lms = _fake_lms(tmp_path, ls=[], ps=[], fail=True)
    state = probe_machine(find_lms=lambda: lms, serving_gguf=lambda url: None,
                          which=lambda _: None, reranker_cached=lambda repo: False)
    assert state.lms_loaded == () and state.lms_installed == ()


def test_hf_cached_reads_a_fake_cache(tmp_path):
    pytest.importorskip("sentence_transformers")
    repo_dir = tmp_path / "models--BAAI--bge-reranker-v2-m3"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("abc123")
    snap = repo_dir / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert hf_cached(RERANKER.model, cache_dir=str(tmp_path)) is True
    assert hf_cached("BAAI/not-there", cache_dir=str(tmp_path)) is False
