"""Factories that build provider instances from Config."""

from __future__ import annotations

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


def make_embedder(cfg: Config) -> Embedder:
    provider = cfg.get("embeddings.provider")
    base_url = cfg.get("embeddings.base_url")
    if provider == "openai":
        pass
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
    )


def make_generator(cfg: Config) -> Generator | None:
    if not cfg.get("expansion.enabled"):
        return None
    return OpenAICompatGenerator(
        base_url=cfg.get("expansion.base_url"),
        model=cfg.get("expansion.model"),
    )


def make_reranker(cfg: Config) -> Reranker | None:
    if not cfg.get("rerank.enabled"):
        return None
    try:
        return STReranker(model=cfg.get("rerank.model"))
    except RagxError as exc:
        print(f"warning: reranker unavailable: {exc}", file=sys.stderr)
        return None
