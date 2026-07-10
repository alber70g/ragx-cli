"""ragx CLI shell: init, status, config. index/query are wired in a later pass."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from ragx.cli.output import emit_json, fail
from ragx.core.config import Config, RAGX_DIR, db_path, require_root, write_default_config
from ragx.core.errors import RagxError

app = typer.Typer(add_completion=False, no_args_is_help=True)
config_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(config_app, name="config")

from ragx.cli import eval_cmd, inspect_cmd, pipeline  # noqa: E402  (needs `app` defined above)

pipeline.register(app)
inspect_cmd.register(app)
eval_cmd.register(app)


def _require_root() -> Path:
    try:
        return require_root()
    except RagxError as exc:
        fail(str(exc))


@app.command()
def init(path: Path = typer.Argument(Path("."))) -> None:
    """Create .ragx/ with a default config at PATH (default: cwd)."""
    root = path.resolve()
    if (root / RAGX_DIR).exists():
        fail(f"{root / RAGX_DIR} already exists")
    cfg_path = write_default_config(root)
    typer.echo(f"created {cfg_path}")


def _counts(root: Path) -> tuple[int, int, int]:
    db = db_path(root)
    if not db.exists():
        return 0, 0, 0
    try:
        from ragx.core.store import Store
    except ImportError:
        return 0, 0, 0
    with Store(db) as store:
        return store.file_count(), store.chunk_count(), store.edge_count()


@app.command()
def status(json_out: bool = typer.Option(False, "--json")) -> None:
    """Show corpus root, embedding config, and index counts."""
    root = _require_root()
    cfg = Config.load(root)
    files, chunks, edges = _counts(root)
    doc = {
        "schema": "ragx.status.v1",
        "root": str(root),
        "embedding_provider": cfg.get("embeddings.provider"),
        "embedding_model": cfg.get("embeddings.model"),
        "files": files,
        "chunks": chunks,
        "edges": edges,
    }
    if json_out:
        emit_json(doc)
    else:
        typer.echo(f"root: {doc['root']}")
        typer.echo(f"embedding: {doc['embedding_provider']}/{doc['embedding_model']}")
        typer.echo(f"files: {files}  chunks: {chunks}  edges: {edges}")
    if files == 0 and chunks == 0 and edges == 0:
        raise typer.Exit(code=1)


@config_app.command("get")
def config_get(key: str) -> None:
    root = _require_root()
    cfg = Config.load(root)
    try:
        value = cfg.get(key)
    except RagxError as exc:
        fail(str(exc))
    typer.echo(str(value))


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    root = _require_root()
    cfg = Config.load(root)
    try:
        cfg.set(key, value)
    except RagxError as exc:
        fail(str(exc))
    cfg.save(root)
    typer.echo(f"{key} = {cfg.get(key)}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s: %(message)s")
    app()
