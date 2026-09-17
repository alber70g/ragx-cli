"""doctor: every check is exercised with injected fake providers — no network, no
model downloads. The CLI test goes through respx so the served-provider path is real."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx
from typer.testing import CliRunner

from ragx.cli.app import app
from ragx.core.config import DEFAULTS, Config, db_path
from ragx.core.doctor import FAIL, OK, SKIP, WARN, run_doctor
from ragx.core.errors import RagxError
from ragx.core.models import ChunkDraft, FileRecord
from ragx.core.store import Store
from ragx.providers import lmstudio_process

runner = CliRunner()


class FakeEmbedder:
    model = "fake-embed"

    def dimension(self) -> int:
        return 4

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    embed_queries = embed_documents


class FakeReranker:
    model = "fake-rerank"

    def score(self, query, texts):
        return [0.1, 0.9][: len(texts)]


class FakeGenerator:
    model = "fake-llm"
    base_url = "http://fake:1234/v1"

    def generate(self, system, prompt, *, max_tokens=1024):
        return "{}"


def _cfg(**overrides) -> Config:
    cfg = Config({k: dict(v) for k, v in DEFAULTS.items()})
    for key, value in overrides.items():
        cfg.set(key.replace("__", "."), value)
    return cfg


def _run(root: Path, cfg: Config, *, embedder=None, generator=None, reranker=None):
    report = run_doctor(
        root,
        cfg,
        embedder_factory=embedder or (lambda c: FakeEmbedder()),
        generator_factory=generator or (lambda c: None),
        reranker_factory=reranker or (lambda c: FakeReranker()),
    )
    return {c.name: c for c in report.checks}, report


def test_healthy_corpus_without_index_warns_but_passes(tmp_path):
    cfg = _cfg(expansion__enabled=False)
    cfg.save(tmp_path)
    checks, report = _run(tmp_path, cfg)
    assert report.ok  # a missing index is a warning, not a failure
    assert checks["config"].status == OK
    assert checks["embeddings"].status == OK and "dim=4" in checks["embeddings"].detail
    assert checks["expansion"].status == SKIP
    assert checks["rerank"].status == OK
    assert checks["index"].status == WARN and "ragx-cli index" in checks["index"].hint


def test_embeddings_down_hints_at_autostart_being_off(tmp_path):
    cfg = _cfg(expansion__enabled=False, embeddings__autostart=False)
    cfg.save(tmp_path)

    def boom(c):
        raise RagxError("request to http://localhost:1234/v1/embeddings failed")

    checks, report = _run(tmp_path, cfg, embedder=boom)
    assert not report.ok
    assert checks["embeddings"].status == FAIL
    assert "lms server start" in checks["embeddings"].hint


def test_embeddings_down_with_autostart_on_points_at_the_start_that_failed(tmp_path):
    cfg = _cfg(expansion__enabled=False)
    cfg.save(tmp_path)

    def boom(c):
        raise RagxError("request to http://localhost:1234/v1/embeddings failed")

    checks, _ = _run(tmp_path, cfg, embedder=boom)
    assert "lms server status" in checks["embeddings"].hint


def test_autostart_actions_are_reported(tmp_path, monkeypatch):
    """What doctor started is part of the answer — otherwise a passing run hides the
    fact that it had to boot LM Studio to get there."""
    cfg = _cfg(expansion__enabled=False)
    cfg.save(tmp_path)
    monkeypatch.setattr(
        lmstudio_process,
        "ensure_for_section",
        lambda c, section: ["started the LM Studio server on port 1234", "loaded 'm' into LM Studio"],
    )
    checks, report = _run(tmp_path, cfg)
    assert report.ok
    assert checks["embeddings"].status == OK
    assert "started the LM Studio server on port 1234" in checks["embeddings"].detail


def test_llama_server_embeddings_keep_their_own_hint(tmp_path):
    """The llama-server error text is already actionable — don't bolt the LM Studio
    hint onto it and send the user to the wrong fix."""
    cfg = _cfg(expansion__enabled=False, embeddings__provider="llama-server")
    cfg.save(tmp_path)

    def boom(c):
        raise RagxError("llama-server binary not found")

    checks, _ = _run(tmp_path, cfg, embedder=boom)
    assert checks["embeddings"].status == FAIL
    assert checks["embeddings"].hint == ""


def test_rerank_failure_is_reported_not_swallowed(tmp_path):
    """make_reranker(strict=True) raises; doctor must surface the real message rather
    than reporting a healthy config with a stray warning on stderr."""
    cfg = _cfg(expansion__enabled=False)
    cfg.save(tmp_path)

    def boom(c):
        raise RagxError("sentence-transformers not installed")

    checks, report = _run(tmp_path, cfg, reranker=boom)
    assert not report.ok
    assert checks["rerank"].status == FAIL
    assert "sentence-transformers not installed" in checks["rerank"].detail


def test_rerank_disabled_is_skipped(tmp_path):
    cfg = _cfg(expansion__enabled=False, rerank__enabled=False)
    cfg.save(tmp_path)
    checks, report = _run(tmp_path, cfg, reranker=lambda c: None)
    assert report.ok
    assert checks["rerank"].status == SKIP


@respx.mock
def test_expansion_warns_when_the_model_is_not_served(tmp_path):
    cfg = _cfg()
    cfg.save(tmp_path)
    respx.get("http://fake:1234/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "some-other-model"}]})
    )
    checks, report = _run(tmp_path, cfg, generator=lambda c: FakeGenerator())
    assert report.ok  # a JIT-loadable model must not fail the run
    assert checks["expansion"].status == WARN


@respx.mock
def test_expansion_fails_when_the_server_is_down(tmp_path):
    cfg = _cfg()
    cfg.save(tmp_path)
    respx.get("http://fake:1234/v1/models").mock(side_effect=httpx.ConnectError("down"))
    checks, report = _run(tmp_path, cfg, generator=lambda c: FakeGenerator())
    assert not report.ok
    assert checks["expansion"].status == FAIL


def test_index_model_mismatch_fails(tmp_path):
    """The footgun this command exists for: config was switched, index wasn't rebuilt."""
    cfg = _cfg(expansion__enabled=False, embeddings__model="new-model")
    cfg.save(tmp_path)
    (tmp_path / ".ragx").mkdir()
    with Store(db_path(tmp_path)) as store:
        store.set_meta("embedding_model", "old-model")
        store.upsert_file(
            FileRecord(path="a.md", content_hash="deadbeef", mtime=1.0, chunk_count=1)
        )
        store.insert_chunks("a.md", [_draft()])
    checks, report = _run(tmp_path, cfg)
    assert not report.ok
    assert checks["index"].status == FAIL
    assert "index --full" in checks["index"].hint


def _draft():
    return ChunkDraft(text="hello", byte_start=0, byte_end=5, line_start=1, line_end=1)


@respx.mock
def test_cli_json_output(tmp_path):
    cfg = _cfg(expansion__enabled=False, rerank__enabled=False)
    cfg.save(tmp_path)
    respx.post("http://localhost:1234/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]})
    )
    result = runner.invoke(app, ["doctor", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["schema"] == "ragx.doctor.v1"
    assert doc["ok"] is True
    names = [c["name"] for c in doc["checks"]]
    assert names == ["config", "embeddings", "expansion", "rerank", "index"]
    embed = next(c for c in doc["checks"] if c["name"] == "embeddings")
    assert embed["status"] == OK and "dim=3" in embed["detail"]


@respx.mock
def test_cli_exit_code_1_when_a_check_fails(tmp_path):
    cfg = _cfg(expansion__enabled=False, rerank__enabled=False)
    cfg.save(tmp_path)
    respx.post("http://localhost:1234/v1/embeddings").mock(side_effect=httpx.ConnectError("down"))
    result = runner.invoke(app, ["doctor", str(tmp_path)])
    assert result.exit_code == 1
    assert "lms server status" in result.output


def test_cli_exit_code_2_outside_a_corpus(tmp_path):
    result = runner.invoke(app, ["doctor", str(tmp_path)])
    assert result.exit_code == 2
