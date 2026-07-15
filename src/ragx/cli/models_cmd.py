"""`ragx-cli models`: recommend, download (via LM Studio), and configure an
embedding + reranker combo. Repeatable — run it again to switch models. Registered
onto the main app by app.py; `init` routes here as an optional final step."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import typer

from ragx.cli import models_ui as ui
from ragx.cli.output import emit_json, fail, migrate_confirm
from ragx.core import lmstudio
from ragx.core.catalog import QUALITY_TIERS, detect_specs, recommend
from ragx.core.config import Config, require_root
from ragx.core.errors import RagxError

INSTALL_HINT = "LM Studio's `lms` CLI not found — install LM Studio (https://lmstudio.ai) and launch it once"


def register(app: typer.Typer) -> None:
    app.command()(models)


def models(
    path: Path | None = typer.Argument(None),
    quality: str | None = typer.Option(
        None, "--quality", help="fast | balanced | best | jina-nano (CC-BY-NC, llama-server only)"
    ),
    rerank_engine: str | None = typer.Option(
        None, "--rerank-engine",
        help="llama-server (GGUF via LM Studio, no huggingface.co needed) | sentence-transformers",
    ),
    embed_engine: str | None = typer.Option(
        None, "--embed-engine",
        help="llama-server (auto-spawned by ragx; LM Studio only downloads) | lm-studio (LM Studio serves)",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="skip the confirmation prompt"),
    dry_run: bool = typer.Option(False, "--dry-run", help="recommend only; no downloads, no config writes"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Recommend + download an embedding/reranker combo and write it to ragx.toml."""
    try:
        root = require_root(path)
        run_flow(root, quality=quality, rerank_engine=rerank_engine,
                 embed_engine=embed_engine, yes=yes, dry_run=dry_run, json_out=json_out)
    except RagxError as exc:
        fail(str(exc))


def run_flow(
    root: Path,
    *,
    quality: str | None = None,
    rerank_engine: str | None = None,
    embed_engine: str | None = None,
    yes: bool = False,
    dry_run: bool = False,
    json_out: bool = False,
) -> None:
    """The actual flow, callable from `init`. Raises RagxError on failure."""
    cfg = Config.load(root, confirm=migrate_confirm())
    interactive = sys.stdin.isatty() and not yes
    specs = detect_specs()
    llama_found = shutil.which("llama-server") is not None
    lms = lmstudio.find_lms() if (interactive or not dry_run) else None
    ui.header(specs, llama_found, lms_found=None if (dry_run and not interactive) else lms is not None)

    if quality is None:
        if not interactive:
            raise RagxError(f"--quality required when not interactive ({'/'.join(QUALITY_TIERS)})")
        quality = ui.pick_tier(specs)
    embedding, reranker, notes = recommend(quality, specs, enforce_ram=not interactive)

    if embedding.requires_llama_server and embed_engine != "llama-server":
        if embed_engine is not None:
            raise RagxError(
                f"{embedding.label} can only be served by the llama-server engine "
                "(LM Studio downloads it but cannot serve EuroBERT models) — "
                "rerun with --embed-engine llama-server"
            )
        emb_engine = "llama-server"
        lock_note = f"embedding engine locked to llama-server — LM Studio cannot serve {embedding.label}"
        if interactive:
            ui.echo(f"\n   → {lock_note}")
        else:
            notes.append(lock_note)
    else:
        emb_engine = _resolve_engine(
            "embedding engine", embed_engine, ui.EMBED_ENGINES,
            interactive, llama_found, fallback="lm-studio",
        )
    engine = _resolve_engine(
        "rerank engine", rerank_engine, ui.RERANK_ENGINES,
        interactive, llama_found, fallback="sentence-transformers",
    )

    engine_desc = (
        "llama.cpp llama-server, GGUF downloaded via LM Studio — no huggingface.co needed"
        if engine == "llama-server"
        else "sentence-transformers, downloaded from Hugging Face"
    )
    emb_desc = (
        "llama-server auto-spawned by ragx" if emb_engine == "llama-server" else "served by LM Studio"
    )
    spec_line = f"{embedding.params}, {embedding.languages} languages, {embedding.context} context"
    _echo(f"embedding: {embedding.label} ({spec_line}) — {emb_desc}")
    _echo(f"reranker:  {reranker.label} ({reranker.notes}) — via {engine_desc}")
    for note in notes:
        _echo(f"note: {note}")

    if dry_run:
        _finish(json_out, embedding, reranker, notes, engine=engine, emb_engine=emb_engine,
                downloaded=False, config_updated=False, reindex_required=False)
        return

    if lms is None:
        raise RagxError(INSTALL_HINT)

    emb_frag = embedding.gguf_fragment if emb_engine == "llama-server" else embedding.repo_fragment
    emb_pre = lmstudio.find_installed(lms, emb_frag)
    rr_pre = lmstudio.find_installed(lms, reranker.gguf_fragment) if engine == "llama-server" else None

    if interactive:
        if emb_pre is not None and emb_pre.model_key == cfg.get("embeddings.model"):
            reindex_line = "not needed — embedding model unchanged"
        else:
            reindex_line = "REQUIRED — embedding model changes; run `ragx-cli index --full` after"
        ui.show_plan(embedding=embedding, reranker=reranker, emb_engine=emb_engine,
                     rerank_engine=engine, emb_installed=emb_pre is not None,
                     rr_installed=rr_pre is not None, reindex_line=reindex_line)
        if not typer.confirm("   proceed — download and update ragx.toml?", default=True, err=True):
            raise typer.Exit(code=0)

    embed_updates: list[tuple[str, str]]
    if emb_engine == "llama-server":
        installed, downloaded = _ensure_downloaded(
            lms, embedding.gguf_ref, embedding.gguf_fragment, pre=emb_pre
        )
        embed_updates = [
            ("embeddings.provider", "llama-server"),
            ("embeddings.gguf", str(lmstudio.models_root() / installed.path)),
            ("embeddings.base_url", "http://127.0.0.1:9813/v1"),
        ]
    else:
        installed, downloaded = _ensure_downloaded(
            lms, embedding.ref, embedding.repo_fragment, pre=emb_pre
        )
        embed_updates = [
            ("embeddings.provider", "openai"),
            ("embeddings.gguf", ""),
            ("embeddings.base_url", "http://localhost:1234/v1"),
        ]

    rerank_updates: list[tuple[str, str]]
    if engine == "llama-server":
        gguf_model, gguf_downloaded = _ensure_downloaded(
            lms, reranker.gguf_ref, reranker.gguf_fragment, pre=rr_pre
        )
        downloaded = downloaded or gguf_downloaded
        gguf_path = str(lmstudio.models_root() / gguf_model.path)
        rerank_updates = [("rerank.provider", "llama-server"), ("rerank.gguf", gguf_path),
                          ("rerank.model", reranker.model)]
    else:
        prefetch_note = _prefetch_reranker(reranker.model)
        if prefetch_note:
            _echo(prefetch_note)
        rerank_updates = [("rerank.provider", "sentence-transformers"), ("rerank.gguf", ""),
                          ("rerank.model", reranker.model)]

    reindex_required = cfg.get("embeddings.model") != installed.model_key
    for key, value in (
        ("embeddings.model", installed.model_key),
        ("embeddings.doc_prefix", embedding.doc_prefix),
        ("embeddings.query_prefix", embedding.query_prefix),
        *embed_updates,
        *rerank_updates,
    ):
        cfg.set(key, value)
    cfg.save(root)
    _echo(f"ragx.toml updated: embeddings.model={installed.model_key} (engine: {emb_engine})  "
          f"rerank.model={reranker.model} (engine: {engine})")
    if reindex_required:
        _echo("embedding model changed — the existing index is invalid; run `ragx-cli index --full`")
    _finish(json_out, embedding, reranker, notes, engine=engine, emb_engine=emb_engine,
            downloaded=downloaded, config_updated=True, reindex_required=reindex_required,
            model_key=installed.model_key)


def _resolve_engine(
    label: str, value: str | None, options: tuple[tuple[str, str], ...],
    interactive: bool, llama_found: bool, *, fallback: str,
) -> str:
    """Explicit flag wins; interactive shows a numbered menu; otherwise default to
    llama-server when the binary is present (fully ragx-managed, no huggingface.co),
    falling back to `fallback`."""
    names = tuple(name for name, _ in options)
    if value is not None:
        if value not in names:
            raise RagxError(f"{label} must be one of {'/'.join(names)}")
        return value
    default = "llama-server" if llama_found else fallback
    if interactive:
        return ui.pick_engine(label, options, default, llama_found)
    return default


def _ensure_downloaded(lms: str, ref: str, fragment: str, pre: lmstudio.InstalledModel | None = None):
    """Download `ref` via LM Studio unless a model matching `fragment` is already there
    (`pre` = an earlier find_installed result to reuse)."""
    installed = pre if pre is not None else lmstudio.find_installed(lms, fragment)
    if installed is not None:
        _echo(f"already downloaded: {installed.model_key}")
        return installed, False
    _echo(f"downloading {ref} via LM Studio ...")
    lmstudio.download(lms, ref)
    installed = lmstudio.find_installed(lms, fragment)
    if installed is None:
        raise RagxError(
            f"downloaded {ref} but no model matching {fragment!r} shows up in "
            "`lms ls` — inspect LM Studio's My Models"
        )
    return installed, True


def _prefetch_reranker(model: str) -> str | None:
    """Warm the HF cache so the first query doesn't stall; LM Studio can't fetch
    safetensors cross-encoders, so this stays on the huggingface_hub path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return f"reranker {model} not pre-downloaded — install `ragx-cli[rerank]` first"
    _echo(f"pre-downloading reranker {model} from Hugging Face ...")
    try:
        snapshot_download(model)
    except Exception as exc:  # network/auth issues must not sink the config update
        return f"reranker pre-download failed ({exc}) — it will download on first query"
    return None


def _echo(msg: str) -> None:
    typer.echo(msg, err=True)  # keep stdout pure for --json


def _finish(
    json_out: bool, embedding, reranker, notes, *, engine: str, emb_engine: str,
    downloaded: bool, config_updated: bool, reindex_required: bool, model_key: str | None = None,
) -> None:
    if not json_out:
        return
    emit_json({
        "schema": "ragx.models.v1",
        "embedding": {"label": embedding.label, "ref": embedding.ref, "model_key": model_key,
                      "tier": embedding.tier, "engine": emb_engine},
        "reranker": {"label": reranker.label, "model": reranker.model, "engine": engine},
        "notes": notes,
        "downloaded": downloaded,
        "config_updated": config_updated,
        "reindex_required": reindex_required,
    })
