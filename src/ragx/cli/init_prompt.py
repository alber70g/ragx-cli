"""Interactive Q&A for `ragx-cli init`: probes local servers, prompts pre-filled with
detected models / DEFAULTS, returns dotted-key overrides. The caller writes the config."""

from __future__ import annotations

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


def collect_answers() -> dict[str, Any]:
    """Ask for embeddings, query-expansion, and corpus settings. Enter accepts the default."""
    d = DEFAULTS
    answers: dict[str, Any] = {}

    detected = detect_servers()
    for provider, info in detected.items():
        name = "LM Studio" if provider == "openai" else "Ollama"
        typer.echo(f"detected {name} at {info.base_url} ({len(info.models)} models)")

    typer.echo("Embeddings (index + search; openai = any OpenAI-compatible server, e.g. LM Studio)")
    default_provider = next(iter(detected)) if len(detected) == 1 else d["embeddings"]["provider"]
    provider = _prompt_provider("embeddings", default_provider)
    answers["embeddings.provider"] = provider
    base_url = typer.prompt(
        "  base URL", default=_base_url_default(provider, detected, d["embeddings"]["base_url"])
    )
    answers["embeddings.base_url"] = base_url
    models = _models_for(detected, provider, base_url)
    answers["embeddings.model"] = _prompt_model(
        "model", [m for m in models if is_embedding_model(m)], d["embeddings"]["model"]
    )
    answers["embeddings.api_key_env"] = typer.prompt(
        "  env var holding the API key (empty for local servers)",
        default=d["embeddings"]["api_key_env"],
    )

    typer.echo("Query expansion (optional LLM call at query time: variants + HyDE)")
    enabled = typer.confirm("  enable expansion?", default=d["expansion"]["enabled"])
    answers["expansion.enabled"] = enabled
    if enabled:
        provider = _prompt_provider("expansion", answers["embeddings.provider"])
        answers["expansion.provider"] = provider
        exp_url = typer.prompt(
            "  base URL",
            default=answers["embeddings.base_url"]
            if provider == answers["embeddings.provider"]
            else _base_url_default(provider, detected, d["expansion"]["base_url"]),
        )
        answers["expansion.base_url"] = exp_url
        gen_models = [m for m in _models_for(detected, provider, exp_url) if not is_embedding_model(m)]
        if any(looks_thinking(m) for m in gen_models):
            typer.echo("  (thinking/reasoning models listed last — prefer a non-thinking model here)")
        answers["expansion.model"] = _prompt_model(
            "model", sorted(gen_models, key=looks_thinking), d["expansion"]["model"]
        )
        answers["expansion.api_key_env"] = typer.prompt(
            "  env var holding the API key (empty for local servers)",
            default=answers["embeddings.api_key_env"],
        )

    typer.echo("Corpus files (gitignore-style globs, comma-separated)")
    answers["corpus.include"] = typer.prompt("  include", default=",".join(d["corpus"]["include"]))
    answers["corpus.exclude"] = typer.prompt(
        "  exclude (empty for none)", default=",".join(d["corpus"]["exclude"])
    )
    answers["corpus.respect_gitignore"] = typer.confirm(
        "  respect .gitignore?", default=d["corpus"]["respect_gitignore"]
    )
    return answers
