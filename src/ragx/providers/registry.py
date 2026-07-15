"""Factories that build provider instances from Config."""

from __future__ import annotations

import os
import sys

from ragx.core.config import Config
from ragx.core.errors import RagxError
from ragx.providers.base import Embedder, Generator, Reranker
from ragx.providers.openai_compat import OpenAICompatEmbedder, OpenAICompatGenerator
from ragx.providers.st_reranker import STReranker

_OLLAMA_BASE_URL = "http://localhost:11434/v1"
# DEFAULTS["embeddings"]["base_url"] — the LM Studio default; if a user switches to the
# "ollama" provider without also overriding base_url, we swap in the ollama default instead.
_DEFAULT_OPENAI_BASE_URL = "http://localhost:1234/v1"


def _resolve_api_key(cfg: Config, section: str) -> str | None:
    """`<section>.api_key_env` names an env var (fails loud if unset); otherwise fall back to
    the conventional OPENAI_API_KEY when present."""
    env_name = cfg.get(f"{section}.api_key_env")
    if env_name:
        key = os.environ.get(env_name)
        if not key:
            raise RagxError(
                f"[{section}] api_key_env = {env_name!r} but that environment variable is not set"
            )
        return key
    return os.environ.get("OPENAI_API_KEY") or None


def _resolve_base_url(cfg: Config, section: str) -> str:
    """Honor the conventional OPENAI_BASE_URL env var, but only while `<section>.base_url`
    is still the built-in default — an explicit `ragx-cli config set` always wins."""
    base_url = cfg.get(f"{section}.base_url")
    env_url = os.environ.get("OPENAI_BASE_URL")
    if env_url and base_url == _DEFAULT_OPENAI_BASE_URL:
        return env_url.rstrip("/")
    return base_url


_LLAMA_EMBED_BASE_URL = "http://127.0.0.1:9813/v1"  # rerank's llama-server sits on 9814


def make_embedder(cfg: Config) -> Embedder:
    provider = cfg.get("embeddings.provider")
    base_url = cfg.get("embeddings.base_url")
    if provider == "llama-server":
        from ragx.providers.llama_embedder import LlamaServerEmbedder

        if base_url == _DEFAULT_OPENAI_BASE_URL:  # same convention as the ollama swap below
            base_url = _LLAMA_EMBED_BASE_URL
        return LlamaServerEmbedder(
            base_url=base_url,
            gguf=cfg.get("embeddings.gguf"),
            model=cfg.get("embeddings.model"),
            doc_prefix=cfg.get("embeddings.doc_prefix"),
            query_prefix=cfg.get("embeddings.query_prefix"),
            batch_size=cfg.get("embeddings.batch_size"),
            server_bin=cfg.get("embeddings.server_bin"),
        )
    if provider == "openai":
        base_url = _resolve_base_url(cfg, "embeddings")
    elif provider == "ollama":
        if base_url == _DEFAULT_OPENAI_BASE_URL:
            base_url = _OLLAMA_BASE_URL
    else:
        raise RagxError(f"unknown embeddings provider: {provider!r}")
    return OpenAICompatEmbedder(
        base_url=base_url,
        model=cfg.get("embeddings.model"),
        doc_prefix=cfg.get("embeddings.doc_prefix"),
        query_prefix=cfg.get("embeddings.query_prefix"),
        batch_size=cfg.get("embeddings.batch_size"),
        api_key=_resolve_api_key(cfg, "embeddings"),
    )


def make_generator(cfg: Config) -> Generator | None:
    if not cfg.get("expansion.enabled"):
        return None
    return OpenAICompatGenerator(
        base_url=_resolve_base_url(cfg, "expansion"),
        model=cfg.get("expansion.model"),
        api_key=_resolve_api_key(cfg, "expansion"),
    )


def make_reranker(cfg: Config) -> Reranker | None:
    if not cfg.get("rerank.enabled"):
        return None
    provider = cfg.get("rerank.provider")
    try:
        if provider == "llama-server":
            from ragx.providers.llama_server import LlamaServerReranker

            return LlamaServerReranker(
                base_url=cfg.get("rerank.base_url"),
                gguf=cfg.get("rerank.gguf"),
                server_bin=cfg.get("rerank.server_bin"),
            )
        if provider != "sentence-transformers":
            raise RagxError(f"unknown rerank provider: {provider!r}")
        return STReranker(model=cfg.get("rerank.model"))
    except RagxError as exc:
        print(f"warning: reranker unavailable: {exc}", file=sys.stderr)
        return None
