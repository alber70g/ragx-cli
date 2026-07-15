"""`index` and `query` commands. Registered onto the main app by app.py."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from ragx.cli.output import emit_json, fail, migrate_confirm
from ragx.core.config import Config, require_root
from ragx.core.errors import RagxError
from ragx.core.indexer import run_index
from ragx.core.query import QueryOptions, run_query, to_files_json, to_query_json
from ragx.providers.registry import make_embedder, make_generator, make_reranker


def register(app: typer.Typer) -> None:
    app.command()(index)
    app.command()(query)


def index(
    path: Path | None = typer.Argument(None),
    changed: bool = typer.Option(False, "--changed", help="only re-process new/modified/deleted files"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Chunk, embed, and index the corpus (full rebuild unless --changed)."""
    try:
        root = require_root(path)
        cfg = Config.load(root, confirm=migrate_confirm())
        stats = run_index(root, cfg, make_embedder(cfg), changed_only=changed)
    except RagxError as exc:
        fail(str(exc))
    doc = {"schema": "ragx.index.v1", **stats.__dict__}
    if json_out:
        emit_json(doc)
    else:
        typer.echo(
            f"indexed {stats.files_indexed} files ({stats.chunks_added} chunks), "
            f"deleted {stats.files_deleted} ({stats.chunks_deleted} chunks), "
            f"unchanged {stats.files_unchanged}"
        )


def query(
    text: str = typer.Argument(..., help="query text, or '-' to read from stdin"),
    top: int | None = typer.Option(None, "--top"),
    json_out: bool = typer.Option(False, "--json"),
    files_only: bool = typer.Option(False, "--files-only"),
    no_expand: bool = typer.Option(False, "--no-expand"),
    no_graph: bool = typer.Option(False, "--no-graph"),
    no_rerank: bool = typer.Option(False, "--no-rerank"),
    hops: int | None = typer.Option(None, "--hops"),
    explain: bool = typer.Option(False, "--explain"),
) -> None:
    """Search the index; ranked chunks (or files with --files-only)."""
    if text == "-":
        text = sys.stdin.read().strip()
    try:
        root = require_root()
        cfg = Config.load(root, confirm=migrate_confirm())
        opts = QueryOptions(
            top=top if top is not None else cfg.get("query.top"),
            files_only=files_only,
            expand=not no_expand,
            graph=not no_graph,
            rerank=not no_rerank,
            hops=hops,
            explain=explain,
        )
        out = run_query(
            root, cfg, make_embedder(cfg), text, opts,
            generator=make_generator(cfg) if opts.expand else None,
            reranker=make_reranker(cfg) if opts.rerank else None,
        )
    except RagxError as exc:
        fail(str(exc))
    if json_out:
        if files_only:
            emit_json(to_files_json(out))
        else:
            emit_json(to_query_json(out, max_chunk_chars=cfg.get("query.max_chunk_chars")))
    elif files_only:
        for f in to_files_json(out)["files"]:
            typer.echo(f"{f['score']:.4f}  {f['file']}")
    else:
        for r in out.results:
            loc = f"{r.chunk.file_path}:{r.chunk.line_start}-{r.chunk.line_end}"
            first_line = r.chunk.text.strip().splitlines()[0][:100] if r.chunk.text.strip() else ""
            typer.echo(f"{r.score:.4f}  {loc}  {first_line}")
    if not out.results:
        raise typer.Exit(code=1)
