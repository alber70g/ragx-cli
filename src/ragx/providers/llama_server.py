"""Reranker served by llama.cpp: `llama-server --rerank` speaking POST /v1/rerank.

The GGUF can come straight from LM Studio's download dir, so reranking needs no
huggingface.co access at all. Process lifecycle lives in llama_process.py.
Validated: Q8_0 GGUFs of both catalog rerankers score within ~0.5 logit of their
safetensors originals (see DECISIONS.md #2)."""

from __future__ import annotations

from typing import Sequence

import httpx

from ragx.core.errors import RagxError
from ragx.providers.llama_process import LlamaServerProcess


class LlamaServerReranker:
    """Reranker implementing providers.base.Reranker against a llama.cpp rerank server."""

    model: str

    def __init__(self, base_url: str, gguf: str = "", server_bin: str = "llama-server") -> None:
        self._proc = LlamaServerProcess(
            base_url, gguf, server_bin, mode="--rerank", section="rerank"
        )
        self.model = self._proc.gguf.name if self._proc.gguf else self._proc.root_url
        if not self._proc.healthy():
            self._proc.check_spawnable()  # raise now (degrades to a warning) instead of mid-query

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self._proc.ensure()
        try:
            resp = httpx.post(
                f"{self._proc.root_url}/v1/rerank",
                json={"query": query, "documents": list(texts)},
                timeout=120,
            )
            resp.raise_for_status()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            raise RagxError(f"llama-server rerank request failed: {exc}") from exc
        scores = [0.0] * len(texts)
        for r in resp.json()["results"]:
            scores[r["index"]] = float(r["relevance_score"])
        return scores
