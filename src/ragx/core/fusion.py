"""Reciprocal Rank Fusion over multiple rankings."""

from __future__ import annotations

from collections.abc import Sequence


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    """score(d) = sum over rankings containing d of 1/(k + rank), rank 1-based."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores
