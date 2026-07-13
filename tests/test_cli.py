from __future__ import annotations

import json

from typer.testing import CliRunner

from ragx.cli.app import app
from ragx.core.config import RAGX_DIR, Config

runner = CliRunner()


def test_init_fresh(tmp_path):
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / RAGX_DIR / "config.toml").exists()
    assert str(tmp_path / RAGX_DIR / "config.toml") in result.stdout


def test_init_already_initialized(tmp_path):
    runner.invoke(app, ["init", str(tmp_path)])
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 2
    assert "already exists" in result.output


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


def test_status_human_readable(tmp_path, monkeypatch):
    runner.invoke(app, ["init", str(tmp_path)])
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "root:" in result.stdout
    assert "embedding:" in result.stdout


def test_status_without_init(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 2


def test_status_reports_communities(tmp_path, monkeypatch):
    from ragx.core.config import db_path
    from ragx.core.models import ChunkDraft, FileRecord
    from ragx.core.store import Store

    runner.invoke(app, ["init", str(tmp_path)])
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
