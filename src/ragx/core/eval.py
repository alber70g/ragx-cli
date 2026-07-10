"""Evaluation harness: compare retrieval configs against a labeled queries.jsonl file.

See CONTRACTS-PHASE23.md Module I. Decoupled from the pipeline via an injected QueryFn.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ragx.core.errors import RagxError
from ragx.core.query import QueryOptions, QueryOutput, to_files_json

QueryFn = Callable[[str, QueryOptions], QueryOutput]


def load_queries(path: Path) -> list[dict]:
    """Parse `{"query": str, "relevant_files": [str, ...]}` per non-blank line."""
    if not path.exists():
        raise RagxError(f"queries file not found: {path}")
    queries: list[dict] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RagxError(f"malformed query on line {lineno}: {exc}") from exc
        if not isinstance(obj, dict) or "query" not in obj or "relevant_files" not in obj:
            raise RagxError(f"malformed query on line {lineno}: expected 'query' and 'relevant_files'")
        queries.append(obj)
    return queries


def _recall(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(ranked[:k]) & relevant)
    return hits / len(relevant)


def _hit_ranks(ranked: list[str], relevant: set[str]) -> list[int]:
    return [i + 1 for i, f in enumerate(ranked) if f in relevant]


def evaluate(
    queries: list[dict],
    configs: list[tuple[str, QueryOptions]],
    query_fn: QueryFn,
    *,
    top: int = 10,
) -> dict:
    """Run each config against each query; rank files via `to_files_json`, average metrics.

    `top` bounds the ranked-files window that recall@5/@10/mrr are computed over.
    """
    totals = {name: {"recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0} for name, _ in configs}
    per_query: list[dict] = []
    n = len(queries)

    for q in queries:
        relevant = set(q["relevant_files"])
        per_config: dict[str, dict] = {}
        for name, opts in configs:
            out = query_fn(q["query"], opts)
            ranked = [f["file"] for f in to_files_json(out)["files"]][:top]
            hit_ranks = _hit_ranks(ranked, relevant)
            mrr = 1.0 / hit_ranks[0] if hit_ranks else 0.0
            totals[name]["recall_at_5"] += _recall(ranked, relevant, 5)
            totals[name]["recall_at_10"] += _recall(ranked, relevant, 10)
            totals[name]["mrr"] += mrr
            per_config[name] = {"hit_ranks": hit_ranks}
        per_query.append({"query": q["query"], "per_config": per_config})

    configs_out = [
        {
            "name": name,
            "recall_at_5": totals[name]["recall_at_5"] / n if n else 0.0,
            "recall_at_10": totals[name]["recall_at_10"] / n if n else 0.0,
            "mrr": totals[name]["mrr"] / n if n else 0.0,
        }
        for name, _ in configs
    ]

    return {
        "schema": "ragx.eval.v1",
        "top": top,
        "query_count": n,
        "configs": configs_out,
        "queries": per_query,
    }
