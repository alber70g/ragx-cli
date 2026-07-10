from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from ragx.cli import eval_cmd as eval_cmd_mod
from ragx.core.errors import RagxError
from ragx.core.eval import evaluate, load_queries
from ragx.core.models import Chunk, QueryOutput, ScoredChunk
from ragx.core.query import QueryOptions

runner = CliRunner()


def _chunk(id_: int, file_path: str) -> Chunk:
    return Chunk(id=id_, file_path=file_path, text="x", byte_start=0, byte_end=1, line_start=1, line_end=1)


def _output(query: str, scored: list[tuple[str, float]]) -> QueryOutput:
    results = [
        ScoredChunk(chunk=_chunk(i, path), score=score) for i, (path, score) in enumerate(scored)
    ]
    return QueryOutput(query=query, results=results)


# ---------------------------------------------------------------------------
# load_queries
# ---------------------------------------------------------------------------


def test_load_queries_basic(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(
        '{"query": "a", "relevant_files": ["x.py"]}\n'
        "\n"
        '{"query": "b", "relevant_files": ["y.py", "z.py"]}\n'
    )
    result = load_queries(path)
    assert result == [
        {"query": "a", "relevant_files": ["x.py"]},
        {"query": "b", "relevant_files": ["y.py", "z.py"]},
    ]


def test_load_queries_missing_file(tmp_path):
    with pytest.raises(RagxError):
        load_queries(tmp_path / "nope.jsonl")


def test_load_queries_malformed_json_names_line(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text('{"query": "a", "relevant_files": ["x.py"]}\nnot json\n')
    with pytest.raises(RagxError, match="line 2"):
        load_queries(path)


def test_load_queries_missing_fields_names_line(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text('{"query": "a"}\n')
    with pytest.raises(RagxError, match="line 1"):
        load_queries(path)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_recall_and_mrr_hand_computed():
    queries = [
        {"query": "q1", "relevant_files": ["b.py", "d.py"]},
        {"query": "q2", "relevant_files": ["a.py"]},
    ]
    fixed_ranking = [("a.py", 0.9), ("b.py", 0.7), ("c.py", 0.5)]
    outputs = {"q1": _output("q1", fixed_ranking), "q2": _output("q2", fixed_ranking)}

    def query_fn(text: str, opts: QueryOptions) -> QueryOutput:
        return outputs[text]

    result = evaluate(queries, [("cfg", QueryOptions())], query_fn, top=10)

    assert result["schema"] == "ragx.eval.v1"
    assert result["query_count"] == 2
    (cfg,) = result["configs"]
    assert cfg["name"] == "cfg"
    # q1: ranked [a,b,c]; relevant {b,d}; hit b at rank 2 -> recall 1/2, mrr 1/2
    # q2: ranked [a,b,c]; relevant {a};   hit a at rank 1 -> recall 1/1, mrr 1
    assert cfg["recall_at_5"] == pytest.approx((0.5 + 1.0) / 2)
    assert cfg["recall_at_10"] == pytest.approx((0.5 + 1.0) / 2)
    assert cfg["mrr"] == pytest.approx((0.5 + 1.0) / 2)

    per_config = {q["query"]: q["per_config"]["cfg"]["hit_ranks"] for q in result["queries"]}
    assert per_config["q1"] == [2]
    assert per_config["q2"] == [1]


def test_evaluate_no_hit_gives_zero_mrr_and_recall():
    queries = [{"query": "q", "relevant_files": ["missing.py"]}]
    out = _output("q", [("a.py", 0.9), ("b.py", 0.5)])

    def query_fn(text: str, opts: QueryOptions) -> QueryOutput:
        return out

    result = evaluate(queries, [("cfg", QueryOptions())], query_fn)
    (cfg,) = result["configs"]
    assert cfg["recall_at_5"] == 0.0
    assert cfg["recall_at_10"] == 0.0
    assert cfg["mrr"] == 0.0
    assert result["queries"][0]["per_config"]["cfg"]["hit_ranks"] == []


def test_evaluate_top_bounds_the_ranked_window():
    queries = [{"query": "q", "relevant_files": ["b.py"]}]
    out = _output("q", [("a.py", 0.9), ("b.py", 0.7), ("c.py", 0.5)])

    def query_fn(text: str, opts: QueryOptions) -> QueryOutput:
        return out

    result = evaluate(queries, [("cfg", QueryOptions())], query_fn, top=1)
    (cfg,) = result["configs"]
    # b.py is rank 2 overall but window is truncated to top=1 -> not found
    assert cfg["mrr"] == 0.0
    assert cfg["recall_at_5"] == 0.0


def test_evaluate_no_relevant_files_is_zero_not_error():
    queries = [{"query": "q", "relevant_files": []}]
    out = _output("q", [("a.py", 0.9)])

    def query_fn(text: str, opts: QueryOptions) -> QueryOutput:
        return out

    result = evaluate(queries, [("cfg", QueryOptions())], query_fn)
    (cfg,) = result["configs"]
    assert cfg["recall_at_5"] == 0.0
    assert cfg["mrr"] == 0.0


def test_evaluate_multiple_configs_are_independent():
    queries = [{"query": "q", "relevant_files": ["a.py"]}]
    good = _output("q", [("a.py", 0.9)])
    bad = _output("q", [("z.py", 0.9)])

    def query_fn(text: str, opts: QueryOptions) -> QueryOutput:
        return good if opts.graph else bad

    configs = [("on", QueryOptions(graph=True)), ("off", QueryOptions(graph=False))]
    result = evaluate(queries, configs, query_fn)
    by_name = {c["name"]: c for c in result["configs"]}
    assert by_name["on"]["mrr"] == 1.0
    assert by_name["off"]["mrr"] == 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_eval_cmd_missing_file_exits_2(tmp_path):
    app = typer.Typer()
    eval_cmd_mod.register(app)
    result = runner.invoke(app, [str(tmp_path / "missing.jsonl")])
    assert result.exit_code == 2


def test_eval_cmd_malformed_file_exits_2(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text("not json\n")
    app = typer.Typer()
    eval_cmd_mod.register(app)
    result = runner.invoke(app, [str(path)])
    assert result.exit_code == 2


def test_eval_cmd_unknown_config_exits_2(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text('{"query": "q", "relevant_files": []}\n')
    app = typer.Typer()
    eval_cmd_mod.register(app)
    result = runner.invoke(app, [str(path), "--configs", "bogus"])
    assert result.exit_code == 2


def test_eval_cmd_json_output(tmp_path, monkeypatch):
    app = typer.Typer()
    eval_cmd_mod.register(app)

    fake_result = {
        "schema": "ragx.eval.v1",
        "top": 10,
        "query_count": 1,
        "configs": [{"name": "baseline", "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0}],
        "queries": [],
    }
    monkeypatch.setattr(eval_cmd_mod, "load_queries", lambda path: [{"query": "q", "relevant_files": []}])
    monkeypatch.setattr(eval_cmd_mod, "require_root", lambda: tmp_path)
    monkeypatch.setattr(eval_cmd_mod.Config, "load", classmethod(lambda cls, root: object()))
    monkeypatch.setattr(eval_cmd_mod, "make_embedder", lambda cfg: object())
    monkeypatch.setattr(
        eval_cmd_mod,
        "evaluate",
        lambda queries, configs, query_fn, top: fake_result,
    )

    result = runner.invoke(app, ["queries.jsonl", "--json", "--configs", "baseline"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == fake_result


def test_eval_cmd_human_table(tmp_path, monkeypatch):
    app = typer.Typer()
    eval_cmd_mod.register(app)

    fake_result = {
        "schema": "ragx.eval.v1",
        "top": 10,
        "query_count": 1,
        "configs": [{"name": "baseline", "recall_at_5": 0.5, "recall_at_10": 0.5, "mrr": 0.5}],
        "queries": [],
    }
    monkeypatch.setattr(eval_cmd_mod, "load_queries", lambda path: [{"query": "q", "relevant_files": []}])
    monkeypatch.setattr(eval_cmd_mod, "require_root", lambda: tmp_path)
    monkeypatch.setattr(eval_cmd_mod.Config, "load", classmethod(lambda cls, root: object()))
    monkeypatch.setattr(eval_cmd_mod, "make_embedder", lambda cfg: object())
    monkeypatch.setattr(
        eval_cmd_mod,
        "evaluate",
        lambda queries, configs, query_fn, top: fake_result,
    )

    result = runner.invoke(app, ["queries.jsonl", "--configs", "baseline"])
    assert result.exit_code == 0
    assert "baseline" in result.stdout
    assert "recall@5" in result.stdout
