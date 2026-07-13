"""Leiden community detection over the chunk edge list. Pure functions: callers persist."""

from __future__ import annotations

from typing import Sequence

import graspologic_native as gn


def leiden_communities(
    edges: Sequence[tuple[int, int, float]],
    *,
    resolution: float = 1.0,
    seed: int = 42,
) -> dict[int, int]:
    """Partition chunk ids into communities from an undirected weighted edge list.

    `edges` is (src, dst, weight) with src<dst (as stored). Returns chunk_id ->
    community_id, community ids renumbered to consecutive ints 0..n-1 ordered by
    each community's smallest member chunk_id. Isolates (no edges) get no entry.
    """
    if not edges:
        return {}
    ordered = sorted(edges)
    _, raw = gn.leiden(
        edges=[(str(s), str(d), float(w)) for s, d, w in ordered],
        starting_communities=None,
        resolution=resolution,
        randomness=0.001,
        iterations=1,
        use_modularity=True,
        seed=seed,
        trials=1,
    )
    members: dict[int, list[int]] = {}
    for node, cid in raw.items():
        members.setdefault(cid, []).append(int(node))
    ordered_groups = sorted(members.values(), key=min)
    return {
        chunk_id: new_cid
        for new_cid, group in enumerate(ordered_groups)
        for chunk_id in group
    }
