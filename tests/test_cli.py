from __future__ import annotations

import json
import tomllib
from importlib import metadata
from pathlib import Path

import httpx
import respx
from typer.testing import CliRunner

from ragx.cli.app import REPO_URL, app
from ragx.cli.output import migrate_confirm
from ragx.core.config import CONFIG_FILE, Config

runner = CliRunner()

LMSTUDIO_MODELS_URL = "http://localhost:1234/v1/models"
OLLAMA_MODELS_URL = "http://localhost:11434/v1/models"


def _mock_models(url: str, models: list[str] | None) -> None:
    """Mock a /v1/models probe: a model list, or a connection failure when None."""
    route = respx.get(url)
    if models is None:
        route.mock(side_effect=httpx.ConnectError("down"))
    else:
        route.mock(return_value=httpx.Response(200, json={"data": [{"id": m} for m in models]}))


def _read_config(tmp_path):
    with (tmp_path / CONFIG_FILE).open("rb") as f:
        return tomllib.load(f)


def test_init_fresh(tmp_path):
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / CONFIG_FILE).exists()
    assert str(tmp_path / CONFIG_FILE) in result.stdout


def test_init_already_initialized(tmp_path):
    runner.invoke(app, ["init", str(tmp_path)])
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 2
    assert "already exists" in result.output


@respx.mock
def test_init_interactive_detects_server_and_applies_answers(tmp_path):
    _mock_models(LMSTUDIO_MODELS_URL, ["text-embedding-bge-m3", "qwen3.5-9b", "deepseek-r1-8b"])
    _mock_models(OLLAMA_MODELS_URL, None)
    answers = "\n".join(
        [
            "",  # embeddings provider -> detected default (openai)
            "",  # base URL -> detected LM Studio URL
            "1",  # embedding model by number -> text-embedding-bge-m3
            "",  # api key env -> empty (local)
            "",  # enable expansion -> default yes
            "",  # expansion provider -> follows embeddings
            "",  # expansion base URL -> follows embeddings
            "",  # expansion model -> default qwen3.5-9b (non-thinking listed first)
            "",  # expansion api key env
            "**/*.md,docs/**",  # corpus include globs
            "drafts/**",  # corpus exclude globs
            "n",  # respect .gitignore
            "",  # recommend + download models now -> default no
        ]
    )
    result = runner.invoke(app, ["init", str(tmp_path), "--interactive"], input=answers + "\n")
    assert result.exit_code == 0, result.output
    assert "detected LM Studio" in result.output
    assert "thinking/reasoning models listed last" in result.output
    data = _read_config(tmp_path)
    assert data["embeddings"]["provider"] == "openai"
    assert data["embeddings"]["base_url"] == "http://localhost:1234/v1"
    assert data["embeddings"]["model"] == "text-embedding-bge-m3"
    assert data["expansion"]["enabled"] is True
    assert data["expansion"]["model"] == "qwen3.5-9b"
    assert data["corpus"]["include"] == ["**/*.md", "docs/**"]
    assert data["corpus"]["exclude"] == ["drafts/**"]
    assert data["corpus"]["respect_gitignore"] is False


@respx.mock
def test_init_interactive_no_servers_expansion_off(tmp_path):
    _mock_models(LMSTUDIO_MODELS_URL, None)
    _mock_models(OLLAMA_MODELS_URL, None)
    # provider, base URL, model, api key env (defaults), expansion off, corpus defaults,
    # models download declined
    answers = "\n\n\n\nn\n\n\n\n\n"
    result = runner.invoke(app, ["init", str(tmp_path), "--interactive"], input=answers)
    assert result.exit_code == 0, result.output
    data = _read_config(tmp_path)
    assert data["embeddings"]["model"] == "text-embedding-nomic-embed-text-v1.5@q4_k_m"
    assert data["expansion"]["enabled"] is False
    assert data["corpus"]["include"] == ["**/*"]


def test_init_yes_skips_prompts(tmp_path):
    result = runner.invoke(app, ["init", str(tmp_path), "--yes", "--interactive"])
    assert result.exit_code == 0
    data = _read_config(tmp_path)
    assert data["embeddings"]["provider"] == "openai"


def test_version_outside_corpus(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"ragx-cli {metadata.version('ragx-cli')}" in result.stdout
    assert REPO_URL in result.stdout
    assert "config:" not in result.stdout
    assert "global:" not in result.stdout


def test_version_shows_local_models_and_rc(tmp_path, monkeypatch):
    runner.invoke(app, ["init", str(tmp_path)])
    monkeypatch.chdir(tmp_path)
    (Path.home() / ".ragxrc").write_text('[embeddings]\nmodel = "rc-model"\n')

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert str(tmp_path / CONFIG_FILE) in result.stdout
    # effective view: the rc override wins over the corpus value
    assert "embeddings: openai/rc-model" in result.stdout
    assert "rerank: sentence-transformers/BAAI/bge-reranker-v2-m3" in result.stdout
    assert "embeddings.model = rc-model" in result.stdout


def test_version_marks_disabled_stages(tmp_path, monkeypatch):
    runner.invoke(app, ["init", str(tmp_path)])
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["config", "set", "expansion.enabled", "false"])

    result = runner.invoke(app, ["--version"])
    assert "expansion: openai/qwen3.5-9b (disabled)" in result.stdout


def test_config_get_set_roundtrip(tmp_path, monkeypatch):
    runner.invoke(app, ["init", str(tmp_path)])
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["config", "set", "graph.k", "12"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["config", "get", "graph.k"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "12"

    # persisted to disk, not just in-memory
    cfg = Config.load(tmp_path)
    assert cfg.get("graph.k") == 12


def test_config_get_bogus_key(tmp_path, monkeypatch):
    runner.invoke(app, ["init", str(tmp_path)])
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["config", "get", "bogus.key"])
    assert result.exit_code == 2


def test_legacy_config_auto_migrates_when_not_a_tty(tmp_path, monkeypatch):
    # piped stdin (agents): the pre-0.3.0 config self-heals without a prompt
    (tmp_path / ".ragx").mkdir()
    (tmp_path / ".ragx" / "config.toml").write_text("[graph]\nk = 12\n")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["config", "get", "graph.k"])
    assert result.exit_code == 0
    assert result.output.strip() == "12"
    assert (tmp_path / CONFIG_FILE).exists()
    assert not (tmp_path / ".ragx" / "config.toml").exists()


def test_migrate_confirm_prompts_only_on_tty(monkeypatch):
    class Tty:
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", Tty())
    assert migrate_confirm() is not None

    class Pipe:
        def isatty(self):
            return False

    monkeypatch.setattr("sys.stdin", Pipe())
    assert migrate_confirm() is None


def test_config_without_init(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "get", "graph.k"])
    assert result.exit_code == 2


def test_status_json_shape_and_stdout_purity(tmp_path, monkeypatch):
    runner.invoke(app, ["init", str(tmp_path)])
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["status", "--json"])
    # empty index -> exit 1 (success, empty result)
    assert result.exit_code == 1

    doc = json.loads(result.stdout)  # raises if stdout isn't pure JSON
    assert doc["schema"] == "ragx.status.v1"
    assert doc["root"] == str(tmp_path)
    assert doc["files"] == 0
    assert doc["chunks"] == 0
    assert doc["edges"] == 0
    assert doc["drift"] == {"new": 0, "changed": 0, "deleted": 0}


def test_status_human_readable(tmp_path, monkeypatch):
    runner.invoke(app, ["init", str(tmp_path)])
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "root:" in result.stdout
    assert "embedding:" in result.stdout


def test_status_reports_drift(tmp_path, monkeypatch):
    runner.invoke(app, ["init", str(tmp_path)])
    (tmp_path / "a.md").write_text("# A\n\nSome text.\n")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["status", "--json"])
    doc = json.loads(result.stdout)
    assert doc["drift"] == {"new": 1, "changed": 0, "deleted": 0}

    result = runner.invoke(app, ["status"])
    assert "drift: 1 new, 0 changed, 0 deleted" in result.stdout
    assert "ragx-cli index" in result.stdout


def test_index_changed_and_full_mutually_exclusive(tmp_path, monkeypatch):
    runner.invoke(app, ["init", str(tmp_path)])
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["index", "--changed", "--full"])
    assert result.exit_code == 2


def test_status_without_init(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 2


def test_status_reports_communities(tmp_path, monkeypatch):
    from ragx.core.config import db_path
    from ragx.core.models import ChunkDraft, FileRecord
    from ragx.core.store import Store

    runner.invoke(app, ["init", str(tmp_path)])
    db_path(tmp_path).parent.mkdir()
    with Store(db_path(tmp_path)) as store:
        store.upsert_file(FileRecord(path="a.py", content_hash="h1", mtime=1.0, chunk_count=1))
        [cid] = store.insert_chunks(
            "a.py",
            [ChunkDraft(text="x", byte_start=0, byte_end=1, line_start=1, line_end=1)],
        )
        store.replace_communities({cid: 0})
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["schema"] == "ragx.status.v1"
    assert doc["communities"] == 1
