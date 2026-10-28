"""ragx-cli CLI shell: init, status, config. index/query are wired in a later pass."""

from __future__ import annotations

import logging
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import typer

from ragx.cli.init_prompt import collect_answers
from ragx.cli.output import emit_json, fail, migrate_confirm
from ragx.core import machine_probe
from ragx.core.config import (
    PROVIDER_SECTIONS,
    Config,
    CONFIG_FILE,
    config_path,
    db_path,
    find_root,
    load_rc,
    rc_path,
    require_root,
    write_rc_value,
    write_rc_values,
)
from ragx.core.errors import RagxError

REPO_URL = "https://github.com/alber70g/ragx-cli"
INIT_NOTE = (
    "note: corpus.include/exclude default to source code and text docs, minus lockfiles,\n"
    "dependency manifests and build output. An AI agent (Claude Code, Codex, ...) can:\n"
    f"  1. tailor corpus.include/exclude in {CONFIG_FILE} to this project's files\n"
    "  2. add a section to CLAUDE.md or AGENTS.md telling agents to search this corpus\n"
    '     with `ragx-cli query "..."` (--json for machine output, --files-only for paths)'
)

app = typer.Typer(add_completion=False, no_args_is_help=True)
config_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(config_app, name="config")

from ragx.cli import doctor_cmd, eval_cmd, inspect_cmd, models_cmd, pipeline, web_cmd  # noqa: E402  (needs `app` defined above)

pipeline.register(app)
inspect_cmd.register(app)
eval_cmd.register(app)
models_cmd.register(app)
doctor_cmd.register(app)
web_cmd.register(app)


def _print_version() -> None:
    try:
        version = metadata.version("ragx-cli")
    except metadata.PackageNotFoundError:
        version = "unknown"
    typer.echo(f"ragx-cli {version}")
    typer.echo(REPO_URL)
    root = find_root()
    if root is not None:
        try:
            cfg = Config.load(root, confirm=migrate_confirm())
            typer.echo(f"config: {config_path(root)}")
            for section in ("embeddings", "expansion", "rerank"):
                line = f"  {section}: {cfg.get(f'{section}.provider')}/{cfg.get(f'{section}.model')}"
                if cfg.data[section].get("enabled") is False:
                    line += " (disabled)"
                typer.echo(line)
        except RagxError as exc:
            typer.echo(f"config: unreadable ({exc})")
    rc = rc_path()
    if rc.exists():
        typer.echo(f"global: {rc}")
        try:
            for section, values in load_rc().items():
                for key, value in values.items():
                    typer.echo(f"  {section}.{key} = {value}")
        except RagxError as exc:
            typer.echo(f"  unreadable ({exc})")


def _version_option(value: bool) -> None:
    if value:
        _print_version()
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_option,
        is_eager=True,
        help="show version, repo, and configured models (local + ~/.ragxrc)",
    ),
) -> None:
    """Similarity-graph RAG for your files."""


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
    yes: bool = typer.Option(False, "--yes", "-y", help="skip prompts; reuse a detected setup or write the defaults"),
    interactive: bool | None = typer.Option(
        None,
        "--interactive/--no-interactive",
        help="force prompts on/off (default: prompt only when stdin is a TTY)",
    ),
) -> None:
    """Create a ragx.toml config at PATH (default: cwd).

    Without ~/.ragxrc, init looks for a catalog embedding model / reranker this machine
    already has (loaded in LM Studio, served by a running llama-server, downloaded, or in
    the Hugging Face cache) and preselects it; provider answers then seed ~/.ragxrc. With
    an existing ~/.ragxrc nothing is probed and its keys are not asked. On a TTY this walks
    through embeddings, query-expansion, and corpus include/exclude settings; piped stdin
    or --yes uses the detected setup (or the defaults) unchanged.
    """
    root = path.resolve()
    if (root / CONFIG_FILE).exists():
        fail(f"{root / CONFIG_FILE} already exists")
    ask = (interactive if interactive is not None else sys.stdin.isatty()) and not yes
    try:
        answers, reused = _init_answers(ask)
        seed_rc = not rc_path().exists()
        seed = {k: v for k, v in answers.items() if seed_rc and k.partition(".")[0] in PROVIDER_SECTIONS}
        if seed:
            write_rc_values(seed)
            typer.echo(f"created {rc_path()} — applies to every ragx corpus on this machine", err=True)
        cfg = Config.fresh()  # read after the rc is seeded, so save() leaves its keys out
        for key, value in answers.items():
            if key not in seed:
                cfg.set(key, value)
        cfg.save(root)
    except RagxError as exc:
        fail(str(exc))
    typer.echo(f"created {config_path(root)}")
    typer.echo(INIT_NOTE)
    if ask and reused != {"embeddings", "rerank"} and typer.confirm(
        "recommend + download models via LM Studio now? (rerun anytime: `ragx-cli models`)",
        default=False,
    ):
        try:
            models_cmd.run_flow(root)
        except RagxError as exc:
            fail(str(exc))


def _init_answers(ask: bool) -> tuple[dict[str, Any], set[str]]:
    """(dotted answers, sections whose detected setup was reused). An existing rc is
    read, never probed past or modified; otherwise the machine is probed for reuse."""
    if rc_path().exists():
        preset = {f"{s}.{k}": v for s, values in load_rc().items() for k, v in values.items()}
        listed = ", ".join(f"{k}={v}" for k, v in preset.items()) or "none"
        typer.echo(f"provider settings from ~/.ragxrc: {listed}", err=True)
        return collect_answers(preset=preset) if ask else ({}, set())
    reuse = machine_probe.pick_reuse(machine_probe.probe_machine())
    if ask:
        return collect_answers(reuse=reuse)
    answers: dict[str, Any] = {}
    reused: set[str] = set()
    for section, found in (("embeddings", reuse and reuse.embeddings), ("rerank", reuse and reuse.rerank)):
        if found:
            typer.echo(f"{section}: reusing {found.label} ({found.reason})", err=True)
            answers.update(found.values)
            reused.add(section)
    return answers, reused


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
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per embedding batch
    app()
