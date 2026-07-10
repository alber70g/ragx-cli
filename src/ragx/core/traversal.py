"""Heat-propagation traversal: spreads seed relevance heat across the similarity graph."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass
class TraversalResult:
    heat: dict[int, float]  # node -> final heat
    trace: dict[int, dict]  # node -> {"seed": int, "parent": int | None, "edge_weight": float, "hop": int}


def propagate_heat(
    seeds: dict[int, float],
    neighbors_fn: Callable[[int], list[tuple[int, float]]],
    query_sim_fn: Callable[[Sequence[int]], dict[int, float]],
    *,
    hops: int = 2,
    decay: float = 0.5,
    query_floor: float = 0.35,
    max_frontier: int = 150,
) -> TraversalResult:
    heat: dict[int, float] = {}
    trace: dict[int, dict] = {}
    for seed_id, score in seeds.items():
        heat[seed_id] = score
        trace[seed_id] = {"seed": seed_id, "parent": None, "edge_weight": 0.0, "hop": 0}

    sim_cache: dict[int, float] = {}
    frontier = sorted(seeds)

    for hop in range(1, hops + 1):
        if not frontier:
            break

        edges: list[tuple[int, int, float]] = []
        first_seen: list[int] = []
        seen_this_hop: set[int] = set()
        for u in frontier:
            for v, w in neighbors_fn(u):
                edges.append((u, v, w))
                if v not in seeds and v not in sim_cache and v not in seen_this_hop:
                    first_seen.append(v)
                    seen_this_hop.add(v)

        if first_seen:
            sim_cache.update(query_sim_fn(first_seen))

        improved: set[int] = set()
        for u, v, w in edges:
            if v not in seeds and sim_cache.get(v, 0.0) < query_floor:
                continue
            contribution = heat[u] * w * decay
            if contribution > heat.get(v, float("-inf")):
                heat[v] = contribution
                trace[v] = {
                    "seed": trace[u]["seed"],
                    "parent": u,
                    "edge_weight": w,
                    "hop": hop,
                }
                improved.add(v)

        frontier = sorted(improved, key=lambda n: (-heat[n], n))[:max_frontier]

    return TraversalResult(heat=heat, trace=trace)
