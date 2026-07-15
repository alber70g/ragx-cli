"""`ragx-cli eval` command. Registered onto the main app by app.py (integration-owned)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer

from ragx.cli.output import emit_json, fail, migrate_confirm
from ragx.core.config import Config, require_root
from ragx.core.errors import RagxError
from ragx.core.eval import evaluate, load_queries
from ragx.core.query import QueryOptions, run_query
from ragx.providers.registry import make_embedder, make_generator, make_reranker

_BUILTIN_CONFIGS: dict[str, QueryOptions] = {
    "baseline": QueryOptions(expand=False, graph=False, rerank=False),
    "graph": QueryOptions(expand=False, graph=True, rerank=False),
    "rerank": QueryOptions(expand=False, graph=True, rerank=True),
    "full": QueryOptions(expand=True, graph=True, rerank=True),
}


def register(app: typer.Typer) -> None:
    app.command("eval")(eval_cmd)


def eval_cmd(
    queries_file: Path = typer.Argument(..., help="path to a queries.jsonl file"),
    json_out: bool = typer.Option(False, "--json"),
    top: int = typer.Option(10, "--top"),
    configs: str = typer.Option("baseline,graph,full", "--configs"),
) -> None:
    """Evaluate retrieval quality (recall@5/@10, MRR) against labeled queries."""
    try:
        queries = load_queries(queries_file)
    except RagxError as exc:
        fail(str(exc))

    names = [n.strip() for n in configs.split(",") if n.strip()]
    unknown = [n for n in names if n not in _BUILTIN_CONFIGS]
    if unknown:
        fail(f"unknown config(s): {', '.join(unknown)}")

    try:
        root = require_root()
        cfg = Config.load(root, confirm=migrate_confirm())
        embedder = make_embedder(cfg)
        generator = make_generator(cfg) if any(_BUILTIN_CONFIGS[n].expand for n in names) else None
        reranker = make_reranker(cfg) if any(_BUILTIN_CONFIGS[n].rerank for n in names) else None
    except RagxError as exc:
        fail(str(exc))

    def query_fn(text: str, opts: QueryOptions):
        return run_query(
            root, cfg, embedder, text, opts,
            generator=generator if opts.expand else None,
            reranker=reranker if opts.rerank else None,
        )

    # retrieve chunks deep enough that file-level ranking has >= `top` files to rank
    selected = [
        (name, replace(_BUILTIN_CONFIGS[name], top=top * 3)) for name in names
    ]
    result = evaluate(queries, selected, query_fn, top=top)

    if json_out:
        emit_json(result)
        return
    header = f"{'config':<12}{'recall@5':>10}{'recall@10':>10}{'mrr':>10}"
    typer.echo(header)
    for c in result["configs"]:
        typer.echo(f"{c['name']:<12}{c['recall_at_5']:>10.4f}{c['recall_at_10']:>10.4f}{c['mrr']:>10.4f}")
