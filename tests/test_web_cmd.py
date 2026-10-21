"""`ragx-cli web` CLI wiring: flag parsing, --open/--no-open, and fail-loud on RagxError.
`build_server` itself is covered by test_web_server.py / test_graph_export.py — here we
only check the command wires its flags through and handles success/failure correctly."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ragx.cli import web_cmd
from ragx.cli.app import app
from ragx.core.errors import RagxError

runner = CliRunner()


@pytest.fixture
def initialized(tmp_path, monkeypatch):
    assert runner.invoke(app, ["init", str(tmp_path), "--yes"]).exit_code == 0
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_web_builds_server_with_cli_flags_and_opens_browser(initialized, monkeypatch):
    seen: dict = {}

    def fake_build_server(root, cfg, *, host, port, **kwargs):
        seen.update(root=root, host=host, port=port)
        return SimpleNamespace(serve_forever=lambda: None, server_close=lambda: None)

    monkeypatch.setattr(web_cmd, "build_server", fake_build_server)
    opened = []
    monkeypatch.setattr(web_cmd.webbrowser, "open", lambda url: opened.append(url))

    result = runner.invoke(app, ["web", "--host", "0.0.0.0", "--port", "9999"])

    assert result.exit_code == 0
    assert (seen["host"], seen["port"]) == ("0.0.0.0", 9999)
    assert opened == ["http://0.0.0.0:9999/"]


def test_web_no_open_skips_the_browser(initialized, monkeypatch):
    monkeypatch.setattr(
        web_cmd,
        "build_server",
        lambda *a, **k: SimpleNamespace(serve_forever=lambda: None, server_close=lambda: None),
    )
    opened = []
    monkeypatch.setattr(web_cmd.webbrowser, "open", lambda url: opened.append(url))

    result = runner.invoke(app, ["web", "--no-open"])

    assert result.exit_code == 0
    assert opened == []


def test_web_ctrl_c_during_serve_forever_exits_cleanly(initialized, monkeypatch):
    def raise_keyboard_interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(
        web_cmd,
        "build_server",
        lambda *a, **k: SimpleNamespace(serve_forever=raise_keyboard_interrupt, server_close=lambda: None),
    )
    monkeypatch.setattr(web_cmd.webbrowser, "open", lambda url: None)

    result = runner.invoke(app, ["web"])

    assert result.exit_code == 0


def test_web_fails_loud_when_build_server_raises_ragx_error(initialized, monkeypatch):
    def fake_build_server(*a, **k):
        raise RagxError("no index found — run `ragx-cli index` first")

    monkeypatch.setattr(web_cmd, "build_server", fake_build_server)

    result = runner.invoke(app, ["web"])

    assert result.exit_code == 2
    assert "no index found" in result.output
