"""Interactive Q&A for `ragx-cli init`: probes local servers, prompts pre-filled with
detected models / DEFAULTS, returns dotted-key overrides. The caller writes the config.

Keys ~/.ragxrc already sets are not asked (`preset`); a detected reusable setup
(`reuse`) is offered first, as the default answer, with the reason it was picked."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import typer

from ragx.cli.detect import (
    ServerInfo,
    detect_servers,
    is_embedding_model,
    looks_thinking,
    probe_models,
)
from ragx.core.config import DEFAULTS
from ragx.core.machine_probe import Reuse, ReuseChoice

OLLAMA_BASE_URL = "http://localhost:11434/v1"
PROVIDERS = ("openai", "ollama")
MAX_LISTED = 15


def _prompt_provider(label: str, default: str) -> str:
    while True:
        value = typer.prompt(f"  {label} provider ({'/'.join(PROVIDERS)})", default=default)
        if value in PROVIDERS:
            return value
        typer.echo(f"  unknown provider {value!r} — choose one of: {', '.join(PROVIDERS)}")


def _base_url_default(provider: str, detected: dict[str, ServerInfo], fallback: str) -> str:
    if provider in detected:
        return detected[provider].base_url
    return OLLAMA_BASE_URL if provider == "ollama" else fallback


def _models_for(detected: dict[str, ServerInfo], provider: str, base_url: str) -> list[str]:
    info = detected.get(provider)
    if info and info.base_url == base_url:
        return info.models
    return probe_models(base_url)


def _prompt_model(label: str, candidates: list[str], fallback: str) -> str:
    """Free-text prompt; when the server listed candidates, show them numbered and
    accept either a number or a model name."""
    if not candidates:
        return typer.prompt(f"  {label}", default=fallback)
    listed = candidates[:MAX_LISTED]
    for i, model in enumerate(listed, 1):
        typer.echo(f"    {i}. {model}")
    default = fallback if fallback in candidates else listed[0]
    value = typer.prompt(f"  {label} (number or name)", default=default).strip()
    if value.isdigit() and 1 <= int(value) <= len(listed):
        return listed[int(value) - 1]
    return value


def _offer(section: str, reuse: Reuse | None, answers: dict[str, Any]) -> bool:
    """Offer a detected setup; accepting it fills `answers` with its provider keys."""
    if reuse is None:
        return False
    if not typer.confirm(f"  {section}: use {reuse.label} ({reuse.reason})?", default=True):
        return False
    answers.update(reuse.values)
    return True


def collect_answers(
    preset: dict[str, Any] | None = None, reuse: ReuseChoice | None = None
) -> tuple[dict[str, Any], set[str]]:
    """Ask for embeddings, query-expansion, and corpus settings. Enter accepts the default.

    `preset`: dotted keys ~/.ragxrc sets — used as-is, never asked, never returned.
    Returns (answers, sections whose detected `reuse` setup was accepted)."""
    d = DEFAULTS
    preset = preset or {}
    answers: dict[str, Any] = {}
    reused: set[str] = set()

    def ask(key: str, prompt: Callable[[], Any]) -> Any:
        if key in preset:
            return preset[key]
        answers[key] = prompt()
        return answers[key]

    detected = {} if preset else detect_servers()  # an rc means no provider probing
    for provider, info in detected.items():
        name = "LM Studio" if provider == "openai" else "Ollama"
        typer.echo(f"detected {name} at {info.base_url} ({len(info.models)} models)")

    typer.echo("Embeddings (index + search; openai = any OpenAI-compatible server, e.g. LM Studio)")
    if _offer("embeddings", reuse and reuse.embeddings, answers):
        reused.add("embeddings")
        emb_provider, emb_url = answers["embeddings.provider"], answers["embeddings.base_url"]
        emb_key_env = d["embeddings"]["api_key_env"]
    else:
        default_provider = next(iter(detected)) if len(detected) == 1 else d["embeddings"]["provider"]
        emb_provider = ask("embeddings.provider", lambda: _prompt_provider("embeddings", default_provider))
        emb_url = ask("embeddings.base_url", lambda: typer.prompt(
            "  base URL", default=_base_url_default(emb_provider, detected, d["embeddings"]["base_url"])
        ))
        ask("embeddings.model", lambda: _prompt_model(
            "model",
            [m for m in _models_for(detected, emb_provider, emb_url) if is_embedding_model(m)],
            d["embeddings"]["model"],
        ))
        emb_key_env = ask("embeddings.api_key_env", lambda: typer.prompt(
            "  env var holding the API key (empty for local servers)",
            default=d["embeddings"]["api_key_env"],
        ))
    if _offer("reranker", reuse and reuse.rerank, answers):
        reused.add("rerank")

    typer.echo("Query expansion (optional LLM call at query time: variants + HyDE)")
    enabled = ask("expansion.enabled", lambda: typer.confirm(
        "  enable expansion?", default=d["expansion"]["enabled"]
    ))
    if enabled:
        # llama-server serves embeddings only — expansion then defaults to LM Studio
        same = emb_provider in PROVIDERS
        provider = ask("expansion.provider", lambda: _prompt_provider(
            "expansion", emb_provider if same else d["expansion"]["provider"]
        ))
        exp_url = ask("expansion.base_url", lambda: typer.prompt(
            "  base URL",
            default=emb_url
            if same and provider == emb_provider
            else _base_url_default(provider, detected, d["expansion"]["base_url"]),
        ))

        def pick_model() -> str:
            gen_models = [m for m in _models_for(detected, provider, exp_url) if not is_embedding_model(m)]
            if any(looks_thinking(m) for m in gen_models):
                typer.echo("  (thinking/reasoning models listed last — prefer a non-thinking model here)")
            return _prompt_model("model", sorted(gen_models, key=looks_thinking), d["expansion"]["model"])

        ask("expansion.model", pick_model)
        ask("expansion.api_key_env", lambda: typer.prompt(
            "  env var holding the API key (empty for local servers)", default=emb_key_env
        ))

    typer.echo("Corpus files (gitignore-style globs, comma-separated)")
    answers["corpus.include"] = typer.prompt("  include", default=",".join(d["corpus"]["include"]))
    answers["corpus.exclude"] = typer.prompt(
        "  exclude (empty for none)", default=",".join(d["corpus"]["exclude"])
    )
    answers["corpus.respect_gitignore"] = typer.confirm(
        "  respect .gitignore?", default=d["corpus"]["respect_gitignore"]
    )
    return answers, reused
