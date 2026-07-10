"""Score normalization and multi-signal combination."""

from __future__ import annotations

from collections.abc import Iterable


def normalize(scores: dict[int, float]) -> dict[int, float]:
    """Min-max scale to [0, 1]; empty -> {}; all-equal -> all 1.0."""
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi == lo:
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


def combine(
    candidates: Iterable[int],
    vector: dict[int, float],
    heat: dict[int, float],
    rerank: dict[int, float] | None,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> dict[int, float]:
    """Combine per-signal scores, min-max normalizing each component over candidates first."""
    ids = list(candidates)
    raw_vector = {cid: vector.get(cid, 0.0) for cid in ids}
    raw_heat = {cid: heat.get(cid, 0.0) for cid in ids}
    norm_vector = normalize(raw_vector)
    norm_heat = normalize(raw_heat)

    if rerank is not None:
        raw_rerank = {cid: rerank.get(cid, 0.0) for cid in ids}
        norm_rerank = normalize(raw_rerank)
        return {
            cid: alpha * norm_rerank[cid] + beta * norm_heat[cid] + gamma * norm_vector[cid]
            for cid in ids
        }

    denom = beta + gamma
    w_heat = beta / denom if denom else 0.0
    w_vector = gamma / denom if denom else 0.0
    return {cid: w_heat * norm_heat[cid] + w_vector * norm_vector[cid] for cid in ids}
