"""Factories that build provider instances from Config."""

from __future__ import annotations

import os
import sys

from ragx.core.config import DEFAULT_OPENAI_BASE_URL, Config, effective_base_url
from ragx.core.errors import RagxError
from ragx.providers.base import Embedder, Generator, Reranker
from ragx.providers.openai_compat import OpenAICompatEmbedder, OpenAICompatGenerator
from ragx.providers.st_reranker import STReranker

_OLLAMA_BASE_URL = "http://localhost:11434/v1"
# if a user switches to the "ollama" provider without also overriding base_url, we swap
# in the ollama default instead of leaving them pointed at LM Studio's port.
_DEFAULT_OPENAI_BASE_URL = DEFAULT_OPENAI_BASE_URL


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
        base_url = effective_base_url(cfg, "embeddings")
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
        base_url=effective_base_url(cfg, "expansion"),
        model=cfg.get("expansion.model"),
        api_key=_resolve_api_key(cfg, "expansion"),
    )


def make_reranker(cfg: Config, *, strict: bool = False) -> Reranker | None:
    """Build the configured reranker. Query-time callers degrade to no-rerank on failure;
    `strict=True` (doctor) re-raises so the real error can be reported."""
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
        if strict:
            raise
        print(f"warning: reranker unavailable: {exc}", file=sys.stderr)
        return None
