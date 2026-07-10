"""Provider protocols. Implementations live beside this module; factories in registry.py."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Text → vector. Implementations apply doc/query prefixes themselves."""

    model: str

    def dimension(self) -> int:
        """Vector dimension. May lazily probe the backend once and cache."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed corpus chunks (applies doc_prefix if configured). Preserves order."""
        ...

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed search queries (applies query_prefix if configured). Preserves order."""
        ...


@runtime_checkable
class Generator(Protocol):
    """LLM text generation, used only by the optional query-expansion stage."""

    model: str

    def generate(self, system: str, prompt: str, *, max_tokens: int = 1024) -> str:
        """Single completion. Returns raw text; caller parses any JSON."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder relevance scoring."""

    model: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance score per text, same order. Higher is more relevant."""
        ...
