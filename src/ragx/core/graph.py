"""kNN edge construction over a VectorIndex.

Pure functions: no Store access. Callers persist the resulting edges.
"""

from __future__ import annotations

from typing import Sequence

from ragx.core.vectors import VectorIndex


def knn_edges(
    index: VectorIndex, ids: Sequence[int], k: int, min_sim: float
) -> dict[int, list[tuple[int, float]]]:
    """For each id, search its own stored vector for k+1 hits, drop self,
    keep sim >= min_sim, cap at k. Returns id -> [(neighbor_id, sim), ...] sim-desc.
    """
    if not ids:
        return {}
    vectors = index.get_vectors(ids)
    result: dict[int, list[tuple[int, float]]] = {}
    for chunk_id, vector in zip(ids, vectors):
        hits = index.search(vector, k + 1)
        neighbors = [
            (nid, sim) for nid, sim in hits if nid != chunk_id and sim >= min_sim
        ][:k]
        result[chunk_id] = neighbors
    return result


def affected_ids(new_edges: dict[int, list[tuple[int, float]]]) -> set[int]:
    """Every neighbor id referenced in new_edges values, minus new_edges' own keys —
    the incremental-maintenance set whose edge lists the indexer must recompute.
    """
    referenced = {nid for neighbors in new_edges.values() for nid, _ in neighbors}
    return referenced - new_edges.keys()
