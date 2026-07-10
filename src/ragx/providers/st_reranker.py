"""Cross-encoder reranker backed by sentence-transformers (optional extra)."""

from __future__ import annotations

from typing import Sequence

from ragx.core.errors import RagxError


class STReranker:
    """Reranker implementing providers.base.Reranker via a sentence-transformers CrossEncoder."""

    model: str

    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RagxError(
                "sentence-transformers not installed — install with: uv pip install 'ragx[rerank]'"
            ) from exc
        self.model = model
        self._encoder = CrossEncoder(model)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        pairs = [[query, text] for text in texts]
        scores = self._encoder.predict(pairs)
        return [float(s) for s in scores]
