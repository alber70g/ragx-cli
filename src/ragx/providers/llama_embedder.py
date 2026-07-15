"""Embedder served by llama.cpp: `llama-server --embedding` speaking OpenAI-compatible
/v1/embeddings. Same auto-spawn lifecycle as the llama-server reranker, so with both
engines on llama-server, LM Studio is only a downloader and nothing but ragx-managed
processes run at query time."""

from __future__ import annotations

from typing import Sequence

from ragx.providers.llama_process import LlamaServerProcess
from ragx.providers.openai_compat import OpenAICompatEmbedder


class LlamaServerEmbedder:
    """Embedder implementing providers.base.Embedder via a managed llama-server."""

    model: str

    def __init__(
        self,
        base_url: str,
        gguf: str,
        model: str,
        doc_prefix: str = "",
        query_prefix: str = "",
        batch_size: int = 32,
        server_bin: str = "llama-server",
    ) -> None:
        self._proc = LlamaServerProcess(
            base_url, gguf, server_bin, mode="--embedding", section="embeddings"
        )
        self._inner = OpenAICompatEmbedder(
            base_url=f"{self._proc.root_url}/v1",
            model=model,
            doc_prefix=doc_prefix,
            query_prefix=query_prefix,
            batch_size=batch_size,
        )
        self.model = model
        if not self._proc.healthy():
            self._proc.check_spawnable()  # embeddings are required — fail loud early

    def dimension(self) -> int:
        self._proc.ensure()
        return self._inner.dimension()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._proc.ensure()
        return self._inner.embed_documents(texts)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        self._proc.ensure()
        return self._inner.embed_queries(texts)
