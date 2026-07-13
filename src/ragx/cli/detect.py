"""Probe local OpenAI-compatible servers (LM Studio, Ollama) for available models.
Used only by the interactive `init` flow — never at index/query time."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

_EMBED_HINTS = ("embed", "bge-m3", "gte-", "-e5-")
# /v1/models exposes no capability flags on either backend, so "thinking model"
# detection is a best-effort name heuristic (Ollama's native /api/show does have a
# `thinking` capability, but that would cost one extra call per model).
_THINKING_HINTS = ("think", "reason", "-r1", "r1-", "qwq")


@dataclass
class ServerInfo:
    base_url: str
    models: list[str]


def probe_models(base_url: str) -> list[str]:
    """GET {base_url}/models; returns [] on any failure (server down, odd payload)."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=1.0)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [m["id"] for m in data if isinstance(m, dict) and "id" in m]
    except Exception:
        return []


def detect_servers() -> dict[str, ServerInfo]:
    """Probe the LM Studio ("openai" provider) and Ollama default ports; only servers
    that answered with at least one model are returned."""
    out: dict[str, ServerInfo] = {}
    for provider, url in (("openai", LMSTUDIO_BASE_URL), ("ollama", OLLAMA_BASE_URL)):
        models = probe_models(url)
        if models:
            out[provider] = ServerInfo(url, models)
    return out


def is_embedding_model(model_id: str) -> bool:
    return any(h in model_id.lower() for h in _EMBED_HINTS)


def looks_thinking(model_id: str) -> bool:
    return any(h in model_id.lower() for h in _THINKING_HINTS)
