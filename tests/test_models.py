"""catalog recommendation logic, the lms CLI wrapper (against a fake `lms` script),
and the `ragx-cli models` command with downloads mocked out."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from ragx.cli.app import app
from ragx.core import lmstudio
from ragx.core.catalog import EMBEDDINGS, RERANKER, Specs, recommend
from ragx.core.config import Config
from ragx.core.errors import RagxError

runner = CliRunner()

BIG = Specs(ram_gb=32.0, macos=True)


# --- catalog ---------------------------------------------------------------


def test_recommend_tiers():
    emb, rr, _ = recommend("best", BIG)
    assert emb is EMBEDDINGS["best"]
    assert rr is RERANKER

    emb, _, notes = recommend("jina-nano", BIG)
    assert emb is EMBEDDINGS["jina-nano"]
    assert emb.requires_llama_server
    assert any("CC-BY-NC" in n for n in notes)


def test_recommend_downgrades_on_low_ram():
    emb, _, notes = recommend("best", Specs(ram_gb=4.0, macos=False))
    assert emb.tier == "fast"
    assert any("RAM" in n for n in notes)
    # unknown RAM (0.0) must never downgrade
    emb, _, _ = recommend("best", Specs(ram_gb=0.0, macos=False))
    assert emb.tier == "best"


def test_recommend_no_enforce_warns_instead_of_downgrading():
    emb, _, notes = recommend("best", Specs(ram_gb=4.0, macos=False), enforce_ram=False)
    assert emb.tier == "best"
    assert any("may be slow" in n for n in notes)
    # plenty of RAM: no warning either way
    _, _, notes = recommend("best", BIG, enforce_ram=False)
    assert notes == []


def test_recommend_rejects_unknown_quality():
    with pytest.raises(RagxError):
        recommend("ludicrous", BIG)


# --- lms wrapper (fake `lms` shell script) ---------------------------------

LS_JSON = json.dumps(
    [
        {
            "modelKey": "text-embedding-bge-m3",
            "type": "embedding",
            "format": "gguf",
            "path": "gaianet/bge-m3-GGUF/bge-m3-Q8_0.gguf",
            "sizeBytes": 600000000,
        },
        {"modelKey": "qwen3.5-9b", "type": "llm", "format": "gguf", "path": "q/q.gguf"},
    ]
)


def _fake_lms(tmp_path, get_exit=0):
    script = tmp_path / "lms"
    script.write_text(
        "#!/bin/sh\n"
        f"if [ \"$1\" = ls ]; then echo '{LS_JSON}'; exit 0; fi\n"
        f'if [ "$1" = get ]; then echo "Error: No download options available"; exit {get_exit}; fi\n'
        "exit 9\n"
    )
    script.chmod(0o755)
    return str(script)


def test_list_models_and_find_installed(tmp_path):
    lms = _fake_lms(tmp_path)
    models = lmstudio.list_models(lms)
    assert [m.model_key for m in models] == ["text-embedding-bge-m3", "qwen3.5-9b"]
    hit = lmstudio.find_installed(lms, "bge-m3")
    assert hit is not None and hit.type == "embedding"
    assert lmstudio.find_installed(lms, "nomic") is None


def test_download_failure_raises_with_tail(tmp_path):
    lms = _fake_lms(tmp_path, get_exit=1)
    with pytest.raises(RagxError, match="No download options"):
        lmstudio.download(lms, "some/model")


def test_download_success(tmp_path):
    lmstudio.download(_fake_lms(tmp_path, get_exit=0), "some/model")


def test_find_lms_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path))
    assert lmstudio.find_lms() is None
    target = tmp_path / ".lmstudio" / "bin" / "lms"
    target.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\n")
    assert lmstudio.find_lms() == str(target)


# --- models command ---------------------------------------------------------


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    runner.invoke(app, ["init", str(tmp_path)])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ragx.cli.models_cmd._prefetch_reranker", lambda model: None)
    return tmp_path


def test_models_requires_quality_when_not_interactive(corpus):
    result = runner.invoke(app, ["models", "--yes"])
    assert result.exit_code == 2


def test_models_fails_without_lms(corpus, monkeypatch):
    monkeypatch.setattr(lmstudio, "find_lms", lambda: None)
    result = runner.invoke(app, ["models", "--quality", "balanced", "--yes"])
    assert result.exit_code == 2
    assert "lmstudio.ai" in result.output


def test_models_downloads_and_updates_config(corpus, monkeypatch):
    installed = lmstudio.InstalledModel(
        model_key="text-embedding-bge-m3", type="embedding", format="gguf",
        path="gaianet/bge-m3-GGUF/x.gguf", size_bytes=1,
    )
    calls = {"download": 0, "lookups": 0}

    def fake_find_installed(lms, fragment):
        calls["lookups"] += 1
        return installed if calls["download"] else None  # appears only after download

    monkeypatch.setattr(lmstudio, "find_lms", lambda: "/fake/lms")
    monkeypatch.setattr(lmstudio, "find_installed", fake_find_installed)
    monkeypatch.setattr(
        lmstudio, "download",
        lambda lms, ref: calls.__setitem__("download", calls["download"] + 1),
    )

    result = runner.invoke(
        app, ["models", "--quality", "balanced", "--rerank-engine", "sentence-transformers", "--embed-engine", "lm-studio", "--yes", "--json"]
    )
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["schema"] == "ragx.models.v1"
    assert doc["downloaded"] is True
    assert doc["config_updated"] is True
    assert doc["reindex_required"] is True  # default config had the nomic model
    assert calls["download"] == 1

    cfg = Config.load(corpus)
    assert cfg.get("embeddings.model") == "text-embedding-bge-m3"
    assert cfg.get("embeddings.doc_prefix") == ""
    assert cfg.get("embeddings.query_prefix") == ""
    assert cfg.get("rerank.model") == "BAAI/bge-reranker-v2-m3"


def test_models_skips_download_when_installed(corpus, monkeypatch):
    installed = lmstudio.InstalledModel(
        model_key="text-embedding-qwen3-embedding-0.6b", type="embedding", format="gguf",
        path="Qwen/Qwen3-Embedding-0.6B-GGUF/x.gguf", size_bytes=1,
    )
    monkeypatch.setattr(lmstudio, "find_lms", lambda: "/fake/lms")
    monkeypatch.setattr(lmstudio, "find_installed", lambda lms, fragment: installed)
    monkeypatch.setattr(
        lmstudio, "download",
        lambda lms, ref: pytest.fail("must not download an installed model"),
    )
    result = runner.invoke(
        app, ["models", "--quality", "best", "--rerank-engine", "sentence-transformers", "--embed-engine", "lm-studio", "--yes", "--json"]
    )
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["downloaded"] is False
    cfg = Config.load(corpus)
    assert cfg.get("rerank.model") == "BAAI/bge-reranker-v2-m3"
    assert cfg.get("embeddings.query_prefix").startswith("Instruct:")


def test_models_dry_run_touches_nothing(corpus, monkeypatch):
    monkeypatch.setattr(
        lmstudio, "find_lms", lambda: pytest.fail("dry-run must not look for lms")
    )
    before = (corpus / "ragx.toml").read_text()
    result = runner.invoke(
        app, ["models", "--quality", "fast", "--rerank-engine", "sentence-transformers", "--embed-engine", "lm-studio", "--yes", "--dry-run", "--json"]
    )
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["config_updated"] is False and doc["downloaded"] is False
    assert (corpus / "ragx.toml").read_text() == before


def test_models_llama_engine_writes_gguf_config(corpus, monkeypatch, tmp_path):
    embedding = lmstudio.InstalledModel(
        model_key="text-embedding-bge-m3", type="embedding", format="gguf",
        path="gaianet/bge-m3-GGUF/x.gguf", size_bytes=1,
    )
    gguf = lmstudio.InstalledModel(
        model_key="bge-reranker-v2-m3", type="llm", format="gguf",
        path="gpustack/bge-reranker-v2-m3-GGUF/bge-reranker-v2-m3-Q8_0.gguf", size_bytes=1,
    )
    monkeypatch.setattr(lmstudio, "find_lms", lambda: "/fake/lms")
    monkeypatch.setattr(
        lmstudio, "find_installed",
        lambda lms, fragment: gguf if "reranker" in fragment else embedding,
    )
    monkeypatch.setattr(lmstudio, "models_root", lambda: tmp_path / "lmsmodels")

    result = runner.invoke(
        app,
        ["models", "--quality", "balanced",
         "--rerank-engine", "llama-server", "--embed-engine", "lm-studio", "--yes", "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["reranker"]["engine"] == "llama-server"

    cfg = Config.load(corpus)
    assert cfg.get("rerank.provider") == "llama-server"
    assert cfg.get("rerank.gguf") == str(
        tmp_path / "lmsmodels" / "gpustack/bge-reranker-v2-m3-GGUF/bge-reranker-v2-m3-Q8_0.gguf"
    )
    assert cfg.get("rerank.model") == "BAAI/bge-reranker-v2-m3"


def test_models_all_llama_engines(corpus, monkeypatch, tmp_path):
    """--embed-engine + --rerank-engine llama-server: LM Studio is a downloader only."""
    emb_gguf = lmstudio.InstalledModel(
        model_key="bge-m3-q8", type="embedding", format="gguf",
        path="gaianet/bge-m3-GGUF/bge-m3-Q8_0.gguf", size_bytes=1,
    )
    rr_gguf = lmstudio.InstalledModel(
        model_key="bge-reranker-v2-m3", type="llm", format="gguf",
        path="gpustack/bge-reranker-v2-m3-GGUF/bge-reranker-v2-m3-Q8_0.gguf", size_bytes=1,
    )
    monkeypatch.setattr(lmstudio, "find_lms", lambda: "/fake/lms")
    monkeypatch.setattr(
        lmstudio, "find_installed",
        lambda lms, fragment: rr_gguf if "reranker" in fragment else emb_gguf,
    )
    monkeypatch.setattr(lmstudio, "models_root", lambda: tmp_path / "lmsmodels")

    result = runner.invoke(
        app,
        ["models", "--quality", "balanced",
         "--embed-engine", "llama-server", "--rerank-engine", "llama-server", "--yes", "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["embedding"]["engine"] == "llama-server"
    assert doc["reranker"]["engine"] == "llama-server"

    cfg = Config.load(corpus)
    assert cfg.get("embeddings.provider") == "llama-server"
    assert cfg.get("embeddings.gguf").endswith("bge-m3-Q8_0.gguf")
    assert cfg.get("embeddings.base_url") == "http://127.0.0.1:9813/v1"
    assert cfg.get("rerank.provider") == "llama-server"
    assert cfg.get("rerank.gguf").endswith("bge-reranker-v2-m3-Q8_0.gguf")


def test_models_jina_requires_llama_server_engine(corpus, monkeypatch, tmp_path):
    result = runner.invoke(
        app, ["models", "--quality", "jina-nano", "--embed-engine", "lm-studio",
              "--rerank-engine", "llama-server", "--yes"],
    )
    assert result.exit_code == 2
    assert "llama-server" in result.output

    gguf = lmstudio.InstalledModel(
        model_key="v5-nano-retrieval", type="embedding", format="gguf",
        path="jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF/v5-nano-retrieval-Q8_0.gguf",
        size_bytes=1,
    )
    monkeypatch.setattr(lmstudio, "find_lms", lambda: "/fake/lms")
    monkeypatch.setattr(lmstudio, "find_installed", lambda lms, fragment: gguf)
    monkeypatch.setattr(lmstudio, "models_root", lambda: tmp_path / "lmsmodels")
    result = runner.invoke(
        app, ["models", "--quality", "jina-nano", "--embed-engine", "llama-server",
              "--rerank-engine", "llama-server", "--yes", "--json"],
    )
    assert result.exit_code == 0, result.output
    cfg = Config.load(corpus)
    assert cfg.get("embeddings.provider") == "llama-server"
    assert cfg.get("embeddings.query_prefix") == "Query: "
    assert cfg.get("embeddings.gguf").endswith("v5-nano-retrieval-Q8_0.gguf")


def test_models_rejects_unknown_engine(corpus):
    result = runner.invoke(
        app, ["models", "--quality", "fast", "--rerank-engine", "bogus", "--yes"]
    )
    assert result.exit_code == 2


def test_models_jina_auto_locks_embed_engine_when_unset(corpus, monkeypatch, tmp_path):
    """quality=jina-nano without --embed-engine no longer errors: the engine is
    locked to llama-server (the only one that can serve it)."""
    gguf = lmstudio.InstalledModel(
        model_key="v5-nano-retrieval", type="embedding", format="gguf",
        path="jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF/v5-nano-retrieval-Q8_0.gguf",
        size_bytes=1,
    )
    monkeypatch.setattr(lmstudio, "find_lms", lambda: "/fake/lms")
    monkeypatch.setattr(lmstudio, "find_installed", lambda lms, fragment: gguf)
    monkeypatch.setattr(lmstudio, "models_root", lambda: tmp_path / "lmsmodels")
    result = runner.invoke(
        app, ["models", "--quality", "jina-nano", "--rerank-engine", "llama-server",
              "--yes", "--json"],
    )
    assert result.exit_code == 0, result.output
    doc = json.loads(result.stdout)
    assert doc["embedding"]["engine"] == "llama-server"
    assert any("locked" in n for n in doc["notes"])
    cfg = Config.load(corpus)
    assert cfg.get("embeddings.provider") == "llama-server"


# --- interactive flow ---------------------------------------------------------


def _script_prompts(monkeypatch, answers, confirms):
    """Feed scripted answers into the numbered pickers and confirms; a TTY stdin
    makes run_flow take the interactive branch (prompts never touch real stdin)."""
    from types import SimpleNamespace

    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    answer_iter, confirm_iter = iter(answers), iter(confirms)
    monkeypatch.setattr("typer.prompt", lambda *a, **k: next(answer_iter))
    monkeypatch.setattr("typer.confirm", lambda *a, **k: next(confirm_iter))


def test_pick_numbered_reprompts_until_valid(monkeypatch):
    from ragx.cli import models_ui

    answers = iter(["9", "abc", "2"])
    monkeypatch.setattr("typer.prompt", lambda *a, **k: next(answers))
    assert models_ui.pick_numbered(["a", "b", "c"], default=1) == 2


def test_pick_tier_low_ram_defaults_and_override(monkeypatch):
    from ragx.cli import models_ui

    low = Specs(ram_gb=4.0, macos=False)
    # decline the RAM override for "best", then settle on "fast"
    _script_prompts(monkeypatch, answers=["3", "1"], confirms=[False])
    assert models_ui.pick_tier(low) == "fast"
    # confirm the override: the explicit choice is honored
    _script_prompts(monkeypatch, answers=["3"], confirms=[True])
    assert models_ui.pick_tier(low) == "best"


def test_models_interactive_flow_plans_then_writes(corpus, monkeypatch, capsys):
    from ragx.cli.models_cmd import run_flow
    from ragx.core.catalog import Specs as SpecsCls

    installed = lmstudio.InstalledModel(
        model_key="text-embedding-bge-m3", type="embedding", format="gguf",
        path="gaianet/bge-m3-GGUF/x.gguf", size_bytes=1,
    )
    monkeypatch.setattr("ragx.cli.models_cmd.detect_specs", lambda: SpecsCls(32.0, True))
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(lmstudio, "find_lms", lambda: "/fake/lms")
    monkeypatch.setattr(lmstudio, "find_installed", lambda lms, fragment: installed)
    monkeypatch.setattr(
        lmstudio, "download",
        lambda lms, ref: pytest.fail("must not download an installed model"),
    )
    # tier 2 (balanced), embed engine 2 (lm-studio), rerank engine 2 (sentence-transformers)
    _script_prompts(monkeypatch, answers=["2", "2", "2"], confirms=[True])

    run_flow(corpus)

    err = capsys.readouterr().err
    assert "machine:" in err
    assert "── embedding model" in err
    assert "── plan" in err
    assert "already downloaded ✓" in err
    cfg = Config.load(corpus)
    assert cfg.get("embeddings.model") == "text-embedding-bge-m3"
    assert cfg.get("embeddings.provider") == "openai"
    assert cfg.get("rerank.provider") == "sentence-transformers"


def test_models_interactive_jina_skips_embed_engine_prompt(corpus, monkeypatch, tmp_path):
    from ragx.cli.models_cmd import run_flow
    from ragx.core.catalog import Specs as SpecsCls

    gguf = lmstudio.InstalledModel(
        model_key="v5-nano-retrieval", type="embedding", format="gguf",
        path="jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF/v5-nano-retrieval-Q8_0.gguf",
        size_bytes=1,
    )
    monkeypatch.setattr("ragx.cli.models_cmd.detect_specs", lambda: SpecsCls(32.0, True))
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(lmstudio, "find_lms", lambda: "/fake/lms")
    monkeypatch.setattr(lmstudio, "find_installed", lambda lms, fragment: gguf)
    monkeypatch.setattr(lmstudio, "models_root", lambda: tmp_path / "lmsmodels")
    # exactly ONE prompt expected (rerank engine) — an embed-engine prompt would
    # exhaust the iterator and fail the test
    _script_prompts(monkeypatch, answers=["1"], confirms=[True])

    run_flow(corpus, quality="jina-nano")

    cfg = Config.load(corpus)
    assert cfg.get("embeddings.provider") == "llama-server"
    assert cfg.get("rerank.provider") == "llama-server"


def test_init_routes_into_models_flow(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr(
        "ragx.cli.models_cmd.run_flow", lambda root, **kw: called.__setitem__("root", root)
    )
    answers = "\n\n\n\nn\n\n\n\ny\n"  # defaults, expansion off, corpus defaults, models: yes
    result = runner.invoke(app, ["init", str(tmp_path), "--interactive"], input=answers)
    assert result.exit_code == 0, result.output
    assert called["root"] == tmp_path.resolve()
