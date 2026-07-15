"""ragx-cli CLI shell: init, status, config. index/query are wired in a later pass."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from ragx.cli.init_prompt import collect_answers
from ragx.cli.output import emit_json, fail, migrate_confirm
from ragx.core.config import (
    DEFAULTS,
    Config,
    CONFIG_FILE,
    config_path,
    db_path,
    rc_path,
    require_root,
    write_default_config,
    write_rc_value,
)
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


def _load_config(root: Path) -> Config:
    try:
        return Config.load(root, confirm=migrate_confirm())
    except RagxError as exc:
        fail(str(exc))


@app.command()
def init(
    path: Path = typer.Argument(Path(".")),
    yes: bool = typer.Option(False, "--yes", "-y", help="skip prompts and write the default config"),
    interactive: bool | None = typer.Option(
        None,
        "--interactive/--no-interactive",
        help="force prompts on/off (default: prompt only when stdin is a TTY)",
    ),
) -> None:
    """Create a ragx.toml config at PATH (default: cwd).

    On a TTY this walks through embeddings, query-expansion, and corpus include/exclude
    settings (probing local LM Studio/Ollama for models); piped stdin or --yes writes
    the defaults unchanged.
    """
    root = path.resolve()
    if (root / CONFIG_FILE).exists():
        fail(f"{root / CONFIG_FILE} already exists")
    ask = (interactive if interactive is not None else sys.stdin.isatty()) and not yes
    if ask:
        cfg = Config({k: dict(v) for k, v in DEFAULTS.items()})
        for key, value in collect_answers().items():
            cfg.set(key, value)
        cfg.save(root)
        cfg_path = config_path(root)
    else:
        cfg_path = write_default_config(root)
    typer.echo(f"created {cfg_path}")


def _counts(root: Path) -> tuple[int, int, int, int]:
    db = db_path(root)
    if not db.exists():
        return 0, 0, 0, 0
    try:
        from ragx.core.store import Store
    except ImportError:
        return 0, 0, 0, 0
    with Store(db) as store:
        return store.file_count(), store.chunk_count(), store.edge_count(), store.community_count()


@app.command()
def status(json_out: bool = typer.Option(False, "--json")) -> None:
    """Show corpus root, embedding config, index counts, and drift vs the corpus on disk."""
    from ragx.core.indexer import corpus_drift

    root = _require_root()
    cfg = _load_config(root)
    files, chunks, edges, communities = _counts(root)
    drift = corpus_drift(root, cfg)
    doc = {
        "schema": "ragx.status.v1",
        "root": str(root),
        "embedding_provider": cfg.get("embeddings.provider"),
        "embedding_model": cfg.get("embeddings.model"),
        "files": files,
        "chunks": chunks,
        "edges": edges,
        "communities": communities,
        "drift": drift,
    }
    if json_out:
        emit_json(doc)
    else:
        typer.echo(f"root: {doc['root']}")
        typer.echo(f"embedding: {doc['embedding_provider']}/{doc['embedding_model']}")
        typer.echo(f"files: {files}  chunks: {chunks}  edges: {edges}  communities: {communities}")
        line = f"drift: {drift['new']} new, {drift['changed']} changed, {drift['deleted']} deleted"
        if any(drift.values()):
            line += "  (run `ragx-cli index`)"
        typer.echo(line)
    if files == 0 and chunks == 0 and edges == 0:
        raise typer.Exit(code=1)


@config_app.command("get")
def config_get(key: str) -> None:
    root = _require_root()
    cfg = _load_config(root)
    try:
        value = cfg.get(key)
    except RagxError as exc:
        fail(str(exc))
    typer.echo(str(value))


@config_app.command("set")
def config_set(
    key: str,
    value: str,
    global_: bool = typer.Option(False, "--global", help="write to ~/.ragxrc (provider settings only)"),
) -> None:
    if global_:
        try:
            coerced = write_rc_value(key, value)
        except RagxError as exc:
            fail(str(exc))
        typer.echo(f"{key} = {coerced} ({rc_path()})")
        return
    root = _require_root()
    cfg = _load_config(root)
    try:
        cfg.set(key, value)
    except RagxError as exc:
        fail(str(exc))
    cfg.save(root)
    # reload so the echoed value is the effective one (an rc override warns and wins)
    typer.echo(f"{key} = {Config.load(root).get(key)}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s: %(message)s")
    app()
