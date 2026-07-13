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
                "sentence-transformers not installed — install with: uv tool install ragx-cli --with ragx-cli[rerank]"
            ) from exc
        self.model = model
        try:
            self._encoder = CrossEncoder(model)
        except Exception as exc:
            raise RagxError(
                f"failed to load reranker model {model!r}: {exc} — if huggingface.co is "
                "unreachable from this network, point rerank.model at a local copy of the "
                "model directory, set HF_ENDPOINT to a reachable mirror, or disable "
                "reranking (rerank.enabled false); see README 'Reranker on restricted "
                "networks (no huggingface.co)'"
            ) from exc

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        pairs = [[query, text] for text in texts]
        scores = self._encoder.predict(pairs)
        return [float(s) for s in scores]
