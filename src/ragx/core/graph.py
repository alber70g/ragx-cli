"""kNN edge construction over a VectorIndex.

Pure functions: no Store access. Callers persist the resulting edges.

Two edge sources:
- knn_edges: weight = cosine of whole-chunk embeddings (HNSW search).
- subchunk_knn_edges: every sub-chunk links to its k nearest sub-chunks in
  other chunks; links collapse to parent edges (max weight, no per-chunk
  cap), so a chunk mixing several concepts gets a sharp edge per concept and
  the edge budget grows with concept count. Pairs whose whole-chunk cosine
  is already near-duplicate-high are dropped — those are the measured
  precision-killers (see research/fine-grained-sub-chunk-edges-*.md).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

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


def subchunk_knn_edges(
    sub_parents: Sequence[int],
    sub_matrix: np.ndarray,
    src_ids: Sequence[int],
    chunk_vectors: Mapping[int, Sequence[float]],
    k: int,
    min_sim: float,
    near_dup_sim: float,
) -> dict[int, list[tuple[int, float]]]:
    """Edges from per-sub-chunk kNN: sub-chunk links are first-class.

    sub_parents[i] is the chunk id owning row i of sub_matrix (unit vectors,
    n_sub x dim). Each of src's sub-chunks independently links to its k most
    similar sub-chunks in OTHER chunks (sim >= min_sim); links collapse to
    parent-chunk edges keeping the max weight. There is deliberately NO
    per-chunk cap — the edge budget scales with a chunk's concept count, so
    one dominant concept cannot displace the others' links. Near-duplicate
    parents (whole-chunk cosine from chunk_vectors >= near_dup_sim) are
    dropped. Same return shape as knn_edges.
    """
    if not src_ids or len(sub_parents) == 0:
        return {}
    parents = np.asarray(sub_parents, dtype=np.int64)
    cvecs = {cid: np.asarray(v, dtype=np.float32) for cid, v in chunk_vectors.items()}

    result: dict[int, list[tuple[int, float]]] = {}
    for src in src_ids:
        own = parents == src
        rows = sub_matrix[own]
        if rows.size == 0:
            result[src] = []
            continue
        sims = rows @ sub_matrix.T  # (r, n_sub)
        sims[:, own] = -1.0  # a sub-chunk never links back into its own chunk
        best: dict[int, float] = {}
        take = min(k, sims.shape[1])
        for row in sims:
            for j in np.argpartition(-row, take - 1)[:take]:
                weight = float(row[j])
                if weight < min_sim:
                    continue
                nid = int(parents[j])
                if weight > best.get(nid, -1.0):
                    best[nid] = weight
        src_vec = cvecs.get(src)
        neighbors: list[tuple[int, float]] = []
        for nid, weight in sorted(best.items(), key=lambda t: -t[1]):
            nvec = cvecs.get(nid)
            if src_vec is not None and nvec is not None and float(src_vec @ nvec) >= near_dup_sim:
                continue
            neighbors.append((nid, max(0.0, min(1.0, weight))))
        result[src] = neighbors
    return result


def affected_ids(new_edges: dict[int, list[tuple[int, float]]]) -> set[int]:
    """Every neighbor id referenced in new_edges values, minus new_edges' own keys —
    the incremental-maintenance set whose edge lists the indexer must recompute.
    """
    referenced = {nid for neighbors in new_edges.values() for nid, _ in neighbors}
    return referenced - new_edges.keys()
