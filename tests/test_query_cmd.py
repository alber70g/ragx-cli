"""`ragx-cli query` stage flags: vector + graph by default; expansion and rerank opt-in.
Expansion costs an LLM call and rerank loads a cross-encoder — neither may start
unless asked for."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ragx.cli import pipeline
from ragx.cli.app import app

runner = CliRunner()


@pytest.fixture
def calls(tmp_path, monkeypatch):
    assert runner.invoke(app, ["init", str(tmp_path), "--yes"]).exit_code == 0
    monkeypatch.chdir(tmp_path)
    seen: dict = {}
    monkeypatch.setattr(pipeline, "make_embedder", lambda cfg: "embedder")
    monkeypatch.setattr(pipeline, "make_generator", lambda cfg: "generator")
    monkeypatch.setattr(pipeline, "make_reranker", lambda cfg: "reranker")

    def fake_run_query(root, cfg, embedder, text, opts, generator=None, reranker=None):
        seen.update(opts=opts, generator=generator, reranker=reranker)
        return SimpleNamespace(results=[])

    monkeypatch.setattr(pipeline, "run_query", fake_run_query)
    return seen


def test_default_query_skips_expansion_and_rerank(calls):
    result = runner.invoke(app, ["query", "dogs"])
    assert result.exit_code == 1  # fake index returns no results
    assert (calls["opts"].expand, calls["opts"].graph, calls["opts"].rerank) == (False, True, False)
    assert calls["generator"] is None and calls["reranker"] is None


def test_expand_and_rerank_flags_opt_in(calls):
    runner.invoke(app, ["query", "dogs", "--expand", "--rerank"])
    assert calls["opts"].expand and calls["opts"].rerank
    assert calls["generator"] == "generator" and calls["reranker"] == "reranker"


def test_old_no_flags_still_parse_and_warn(calls):
    """Scripts written against <=0.7 pass --no-expand/--no-rerank; they must keep working."""
    result = runner.invoke(app, ["query", "dogs", "--no-expand", "--no-rerank"])
    assert "no such option" not in result.output.lower()
    assert "--no-expand is deprecated" in result.output
    assert calls["opts"].expand is False and calls["opts"].rerank is False


def test_context_defaults_to_config_query_context_lines(calls):
    runner.invoke(app, ["query", "dogs"])
    assert (calls["opts"].before, calls["opts"].after) == (5, 5)  # query.context_lines default


def test_context_flag_sets_both_sides(calls):
    runner.invoke(app, ["query", "dogs", "-C", "2"])
    assert (calls["opts"].before, calls["opts"].after) == (2, 2)


def test_before_after_override_context(calls):
    """Explicit -B/-A win over -C for their own side."""
    runner.invoke(app, ["query", "dogs", "-C", "2", "-B", "7"])
    assert (calls["opts"].before, calls["opts"].after) == (7, 2)


def test_files_only_forces_context_off(calls):
    """--files-only shows no text, so context lines are never read from disk."""
    runner.invoke(app, ["query", "dogs", "--files-only", "-C", "3"])
    assert (calls["opts"].before, calls["opts"].after) == (0, 0)


def test_negative_context_fails_loud(calls):
    result = runner.invoke(app, ["query", "dogs", "-B", "-1"])
    assert result.exit_code == 2
    assert "must be >= 0" in result.output
