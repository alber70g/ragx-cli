"""Cross-encoder reranker backed by sentence-transformers (optional extra)."""

from __future__ import annotations

import logging
import sys
from typing import Sequence

from ragx.core.errors import RagxError

_MANUAL_INSTALL_HINT = (
    "downloading the reranker model from huggingface.co is failing — to install the "
    "model manually, use a mirror, or disable reranking, see "
    "https://github.com/alber70g/ragx-cli#reranker-on-restricted-networks-no-huggingfaceco"
)


class _DownloadHintHandler(logging.Handler):
    """Emits the manual-install hint once, on the first huggingface_hub retry warning."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self._fired = False

    def emit(self, record: logging.LogRecord) -> None:
        if self._fired:
            return
        message = record.getMessage().lower()
        if "retry" in message or "thrown while requesting" in message:
            self._fired = True
            # logging drops re-entrant calls made while a record is being
            # handled, so the hint must go to stderr directly.
            print(f"WARNING: {_MANUAL_INSTALL_HINT}", file=sys.stderr)


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
        hint = _DownloadHintHandler()
        hub_logger = logging.getLogger("huggingface_hub")
        hub_logger.addHandler(hint)
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
        finally:
            hub_logger.removeHandler(hint)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        pairs = [[query, text] for text in texts]
        scores = self._encoder.predict(pairs)
        return [float(s) for s in scores]
