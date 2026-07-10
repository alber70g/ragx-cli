from __future__ import annotations

import json
from pathlib import Path

import typer
from typer.testing import CliRunner

from ragx.cli.inspect_cmd import register
from ragx.core.config import RAGX_DIR, db_path
from ragx.core.models import ChunkDraft, FileRecord
from ragx.core.store import Store

runner = CliRunner()


def _make_app() -> typer.Typer:
    app = typer.Typer()
    register(app)
    return app


def _seed(root: Path) -> dict[str, int]:
    (root / RAGX_DIR).mkdir(parents=True, exist_ok=True)
    with Store(db_path(root)) as store:
        store.upsert_file(FileRecord(path="a.py", content_hash="h1", mtime=1.0, chunk_count=2))
        store.upsert_file(FileRecord(path="b.py", content_hash="h2", mtime=2.0, chunk_count=0))
        store.upsert_file(FileRecord(path="c.py", content_hash="h3", mtime=3.0, chunk_count=1))
        a_ids = store.insert_chunks(
            "a.py",
            [
                ChunkDraft(text="def foo(): pass", byte_start=0, byte_end=16, line_start=1, line_end=1),
                ChunkDraft(text="def bar(): pass", byte_start=16, byte_end=32, line_start=2, line_end=2),
            ],
        )
        c_ids = store.insert_chunks(
            "c.py",
            [ChunkDraft(text="def isolated(): pass", byte_start=0, byte_end=21, line_start=1, line_end=1)],
        )
        store.replace_edges(a_ids[0], [(a_ids[1], 0.9)])
    return {"a0": a_ids[0], "a1": a_ids[1], "c0": c_ids[0]}


def test_inspect_chunk_json(tmp_path, monkeypatch):
    ids = _seed(tmp_path)
    monkeypatch.chdir(tmp_path)
    app = _make_app()

    result = runner.invoke(app, ["inspect", "chunk", str(ids["a0"]), "--json"])
    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["schema"] == "ragx.inspect.chunk.v1"
    assert doc["file"] == "a.py"
    assert doc["line_start"] == 1
    assert doc["text"] == "def foo(): pass"
    assert doc["edges"] == [{"id": ids["a1"], "weight": 0.9, "file": "a.py", "line_start": 2, "line_end": 2}]


def test_inspect_chunk_human_readable(tmp_path, monkeypatch):
    ids = _seed(tmp_path)
    monkeypatch.chdir(tmp_path)
    app = _make_app()

    result = runner.invoke(app, ["inspect", "chunk", str(ids["a0"])])
    assert result.exit_code == 0
    assert "a.py:1-1" in result.stdout
    assert "def foo(): pass" in result.stdout


def test_inspect_chunk_no_edges_exit_1(tmp_path, monkeypatch):
    ids = _seed(tmp_path)
    monkeypatch.chdir(tmp_path)
    app = _make_app()

    result = runner.invoke(app, ["inspect", "chunk", str(ids["c0"]), "--json"])
    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert doc["edges"] == []


def test_inspect_chunk_unknown_id_exit_2(tmp_path, monkeypatch):
    _seed(tmp_path)
    monkeypatch.chdir(tmp_path)
    app = _make_app()

    result = runner.invoke(app, ["inspect", "chunk", "9999"])
    assert result.exit_code == 2
    assert "unknown chunk id" in result.output


def test_inspect_file_json(tmp_path, monkeypatch):
    ids = _seed(tmp_path)
    monkeypatch.chdir(tmp_path)
    app = _make_app()

    result = runner.invoke(app, ["inspect", "file", "a.py", "--json"])
    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["schema"] == "ragx.inspect.file.v1"
    assert doc["path"] == "a.py"
    assert doc["content_hash"] == "h1"
    assert doc["chunk_count"] == 2
    assert doc["chunks"][0] == {"id": ids["a0"], "line_start": 1, "line_end": 1, "preview": "def foo(): pass"}


def test_inspect_file_no_chunks_exit_1(tmp_path, monkeypatch):
    _seed(tmp_path)
    monkeypatch.chdir(tmp_path)
    app = _make_app()

    result = runner.invoke(app, ["inspect", "file", "b.py", "--json"])
    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert doc["chunks"] == []


def test_inspect_file_unknown_path_exit_2(tmp_path, monkeypatch):
    _seed(tmp_path)
    monkeypatch.chdir(tmp_path)
    app = _make_app()

    result = runner.invoke(app, ["inspect", "file", "nope.py"])
    assert result.exit_code == 2
    assert "unknown file" in result.output


def test_inspect_neighbors_json(tmp_path, monkeypatch):
    ids = _seed(tmp_path)
    monkeypatch.chdir(tmp_path)
    app = _make_app()

    result = runner.invoke(app, ["inspect", "neighbors", str(ids["a1"]), "--json"])
    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["schema"] == "ragx.inspect.neighbors.v1"
    assert doc["neighbors"] == [
        {"id": ids["a0"], "weight": 0.9, "file": "a.py", "line_start": 1, "line_end": 1}
    ]


def test_inspect_neighbors_none_exit_1(tmp_path, monkeypatch):
    ids = _seed(tmp_path)
    monkeypatch.chdir(tmp_path)
    app = _make_app()

    result = runner.invoke(app, ["inspect", "neighbors", str(ids["c0"]), "--json"])
    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert doc["neighbors"] == []


def test_inspect_neighbors_unknown_id_exit_2(tmp_path, monkeypatch):
    _seed(tmp_path)
    monkeypatch.chdir(tmp_path)
    app = _make_app()

    result = runner.invoke(app, ["inspect", "neighbors", "9999"])
    assert result.exit_code == 2
    assert "unknown chunk id" in result.output
