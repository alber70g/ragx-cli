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
from ragx.providers import lmstudio_process
from ragx.providers.registry import make_embedder, make_generator, make_reranker


def register(app: typer.Typer) -> None:
    app.command()(index)
    app.command()(query)


def index(
    path: Path | None = typer.Argument(None),
    full: bool = typer.Option(False, "--full", help="full rebuild: re-chunk and re-embed everything"),
    changed: bool = typer.Option(
        False, "--changed", hidden=True, help="deprecated: incremental is now the default"
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Chunk, embed, and index the corpus (incremental by default; --full rebuilds)."""
    if changed and full:
        fail("--changed and --full are mutually exclusive")
    if changed:
        print("warning: --changed is deprecated; incremental is now the default", file=sys.stderr)
    try:
        root = require_root(path)
        cfg = Config.load(root, confirm=migrate_confirm())
        lmstudio_process.ensure_for_section(cfg, "embeddings")
        stats = run_index(root, cfg, make_embedder(cfg), full=full)
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


def _generator(cfg: Config):
    """Expansion is optional: a backend that won't come up must not sink the query —
    expand_query degrades to a no-op on any failure."""
    try:
        lmstudio_process.ensure_for_section(cfg, "expansion")
    except RagxError as exc:
        print(f"warning: query expansion backend unavailable: {exc}", file=sys.stderr)
    return make_generator(cfg)


def query(
    text: str = typer.Argument(..., help="query text, or '-' to read from stdin"),
    top: int | None = typer.Option(None, "--top", help="result entries (files); default query.top = 20"),
    json_out: bool = typer.Option(False, "--json"),
    files_only: bool = typer.Option(False, "--files-only"),
    expand: bool = typer.Option(False, "--expand", help="LLM query expansion (multi-query + HyDE)"),
    no_graph: bool = typer.Option(False, "--no-graph"),
    rerank: bool = typer.Option(False, "--rerank", help="cross-encoder rerank of the shortlist"),
    no_expand: bool = typer.Option(
        False, "--no-expand", hidden=True, help="deprecated: expansion is now off by default"
    ),
    no_rerank: bool = typer.Option(
        False, "--no-rerank", hidden=True, help="deprecated: rerank is now off by default"
    ),
    hops: int | None = typer.Option(None, "--hops"),
    explain: bool = typer.Option(False, "--explain"),
    before: int | None = typer.Option(None, "-B", "--before", help="context lines before each chunk"),
    after: int | None = typer.Option(None, "-A", "--after", help="context lines after each chunk"),
    context: int | None = typer.Option(None, "-C", "--context", help="context lines on both sides"),
) -> None:
    """Search the index: one entry per file with its best chunks (paths only with --files-only)."""
    for flag, used in (("--no-expand", no_expand), ("--no-rerank", no_rerank)):
        if used:
            print(f"warning: {flag} is deprecated; that stage is now off by default", file=sys.stderr)
    for flag, val in (("-B/--before", before), ("-A/--after", after), ("-C/--context", context)):
        if val is not None and val < 0:
            fail(f"{flag} must be >= 0")
    if text == "-":
        text = sys.stdin.read().strip()
    try:
        root = require_root()
        cfg = Config.load(root, confirm=migrate_confirm())
        lmstudio_process.ensure_for_section(cfg, "embeddings")
        default_lines = context if context is not None else cfg.get("query.context_lines")
        before_lines = before if before is not None else default_lines
        after_lines = after if after is not None else default_lines
        if files_only:
            before_lines = after_lines = 0  # not shown; skip the file reads
        opts = QueryOptions(
            top=top if top is not None else cfg.get("query.top"),
            files_only=files_only,
            expand=expand,
            graph=not no_graph,
            rerank=rerank,
            hops=hops,
            explain=explain,
            before=before_lines,
            after=after_lines,
        )
        out = run_query(
            root, cfg, make_embedder(cfg), text, opts,
            generator=_generator(cfg) if opts.expand else None,
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
        for f in out.results:
            typer.echo(f"{f.score:.4f}  {f.file_path}")
            for r in f.chunks:
                lines = f"{r.chunk.line_start}-{r.chunk.line_end}"
                first_line = r.chunk.text.strip().splitlines()[0][:90] if r.chunk.text.strip() else ""
                typer.echo(f"        {r.score:.4f}  {lines:<9}  {first_line}")
                if r.context:
                    _echo_context(r.context.get("before"))
                    _echo_context(r.context.get("after"))
    if not out.results:
        raise typer.Exit(code=1)


def _echo_context(span: dict | None) -> None:
    if not span:
        return
    for offset, line in enumerate(span["text"].split("\n")):
        typer.echo(f"        {span['line_start'] + offset}-  {line}")
