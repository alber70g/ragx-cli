"""OpenAI-compatible HTTP providers: works with LM Studio, Ollama's /v1, and OpenAI itself."""

from __future__ import annotations

import time
from typing import Sequence

import httpx

from ragx.core.errors import RagxError

_MAX_RETRIES = 3


def _post_with_retry(client: httpx.Client, url: str, payload: dict) -> dict:
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.post(url, json=payload)
            if resp.status_code >= 500:
                last_exc = RagxError(f"{url} returned {resp.status_code}: {resp.text}")
            else:
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError as exc:
            last_exc = exc
        if attempt < _MAX_RETRIES - 1:
            time.sleep(2**attempt * 0.01)
    raise RagxError(f"request to {url} failed after {_MAX_RETRIES} attempts: {last_exc}")


class OpenAICompatEmbedder:
    """Embedder implementing providers.base.Embedder against an OpenAI-compatible /embeddings API."""

    model: str

    def __init__(
        self,
        base_url: str,
        model: str,
        doc_prefix: str = "",
        query_prefix: str = "",
        api_key: str | None = None,
        batch_size: int = 32,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout, headers=headers)
        self._dim: int | None = None

    def dimension(self) -> int:
        if self._dim is None:
            vecs = self._embed_raw(["probe"])
            self._dim = len(vecs[0])
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed_prefixed(texts, self.doc_prefix)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed_prefixed(texts, self.query_prefix)

    def _embed_prefixed(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        return self._embed_raw([f"{prefix}{t}" for t in texts])

    def _embed_raw(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            data = _post_with_retry(
                self._client,
                f"{self.base_url}/embeddings",
                {"model": self.model, "input": list(batch)},
            )
            out.extend(item["embedding"] for item in data["data"])
        return out


class OpenAICompatGenerator:
    """Generator implementing providers.base.Generator against an OpenAI-compatible chat API."""

    model: str

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def generate(self, system: str, prompt: str, *, max_tokens: int = 1024) -> str:
        data = _post_with_retry(
            self._client,
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": max_tokens,
            },
        )
        return data["choices"][0]["message"]["content"]
