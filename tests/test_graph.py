import numpy as np
import pytest

from ragx.core.graph import affected_ids, knn_edges, subchunk_knn_edges
from ragx.core.vectors import VectorIndex

DIM = 3

# Constructed unit vectors with known cosines:
#   10: (1, 0, 0)
#   11: (0.8, 0.6, 0)   cos(10,11) = 0.8
#   12: (0.6, 0.8, 0)   cos(10,12) = 0.6, cos(11,12) = 0.96
#   13: (0, 1, 0)       cos(10,13) = 0.0, cos(11,13) = 0.6, cos(12,13) = 0.8
#   14: (-1, 0, 0)      cos(10,14) = -1.0 -> clamped to 0.0 by VectorIndex.search
VECTORS = {
    10: [1.0, 0.0, 0.0],
    11: [0.8, 0.6, 0.0],
    12: [0.6, 0.8, 0.0],
    13: [0.0, 1.0, 0.0],
    14: [-1.0, 0.0, 0.0],
}


@pytest.fixture
def index(tmp_path):
    idx = VectorIndex.create(tmp_path / "v.hnsw", dim=DIM, capacity=64)
    ids = list(VECTORS.keys())
    idx.add(ids, [VECTORS[i] for i in ids])
    return idx


def test_self_excluded_and_sim_desc_order(index):
    edges = knn_edges(index, [10], k=4, min_sim=0.0)
    neighbor_ids = [nid for nid, _ in edges[10]]
    assert 10 not in neighbor_ids
    sims = [s for _, s in edges[10]]
    assert sims == sorted(sims, reverse=True)


def test_known_cosines(index):
    edges = knn_edges(index, [10], k=2, min_sim=0.5)
    assert edges[10] == [pytest.approx((11, 0.8), abs=1e-6), pytest.approx((12, 0.6), abs=1e-6)]


def test_min_sim_filter(index):
    edges = knn_edges(index, [10], k=4, min_sim=0.7)
    assert len(edges[10]) == 1
    nid, sim = edges[10][0]
    assert nid == 11
    assert sim == pytest.approx(0.8, abs=1e-6)


def test_k_cap(index):
    # 11's neighbors by sim desc: 12 (0.96), 10 (0.8), 13 (0.6), 14 (~0, clamped)
    edges = knn_edges(index, [11], k=2, min_sim=0.0)
    assert len(edges[11]) == 2
    assert [nid for nid, _ in edges[11]] == [12, 10]


def test_empty_ids(index):
    assert knn_edges(index, [], k=5, min_sim=0.0) == {}


def test_batches_get_vectors_in_one_call(index, monkeypatch):
    calls = []
    original = index.get_vectors

    def spy(ids):
        calls.append(list(ids))
        return original(ids)

    monkeypatch.setattr(index, "get_vectors", spy)
    knn_edges(index, [10, 11, 12], k=2, min_sim=0.0)
    assert len(calls) == 1
    assert calls[0] == [10, 11, 12]


def test_affected_ids_correctness():
    new_edges = {1: [(2, 0.9), (3, 0.5)], 2: [(1, 0.9)]}
    # referenced = {2, 3}; own keys = {1, 2}; affected = {2, 3} - {1, 2} = {3}
    assert affected_ids(new_edges) == {3}


def test_affected_ids_empty():
    assert affected_ids({}) == set()


# --- subchunk_knn_edges -------------------------------------------------
#
# Chunk 1 mixes two concepts (a1, a2); chunk 2 matches concept a2 only;
# chunk 3 is unrelated; chunk 4 is a near-duplicate of chunk 1.
#   cos(a2, b1) = 0.8   but pooled cos(chunk1, chunk2) ~= 0.57
#   cos(chunk1, chunk4) = 1.0 (near-dup)

R2 = 2**-0.5
SUB_PARENTS = [1, 1, 2, 3, 4, 4]
SUB_MATRIX = np.asarray(
    [
        [1, 0, 0, 0],  # a1 (chunk 1)
        [0, 1, 0, 0],  # a2 (chunk 1)
        [0, 0.8, 0.6, 0],  # b1 (chunk 2)
        [0, 0, 0, 1],  # c1 (chunk 3)
        [1, 0, 0, 0],  # d1 (chunk 4) == a1
        [0, 1, 0, 0],  # d2 (chunk 4) == a2
    ],
    dtype=np.float32,
)
CHUNK_VECTORS = {
    1: [R2, R2, 0, 0],
    2: [0, 0.8, 0.6, 0],
    3: [0, 0, 0, 1],
    4: [R2, R2, 0, 0],
}


def _sub_edges(src_ids, *, k=8, min_sim=0.7, near_dup_sim=2.0):
    return subchunk_knn_edges(
        SUB_PARENTS, SUB_MATRIX, src_ids, CHUNK_VECTORS, k, min_sim, near_dup_sim
    )


def test_subchunk_max_aggregation_beats_pooled_cosine():
    edges = _sub_edges([1])
    weights = dict(edges[1])
    # pooled cosine 1<->2 is ~0.57 (below min_sim) but the concept-level max is 0.8
    assert weights[2] == pytest.approx(0.8, abs=1e-6)
    assert 1 not in weights  # self excluded
    assert 3 not in weights  # unrelated chunk filtered by min_sim


def test_subchunk_near_dup_guard():
    with_guard = _sub_edges([1], near_dup_sim=0.9)
    assert [nid for nid, _ in with_guard[1]] == [2]  # chunk 4 (pooled cos 1.0) dropped

    without_guard = _sub_edges([1])
    weights = dict(without_guard[1])
    assert weights[4] == pytest.approx(1.0, abs=1e-6)  # guard off: near-dup edge kept


def test_subchunk_k_is_per_sub_chunk_and_desc_order():
    # k=1: BOTH of chunk 1's sub-chunks pick their single best link, and both
    # point at chunk 4's sub-chunks -> the links collapse to one parent edge
    edges = _sub_edges([1], k=1)
    assert len(edges[1]) == 1
    assert edges[1][0][0] == 4

    sims = [s for _, s in _sub_edges([1])[1]]
    assert sims == sorted(sims, reverse=True)


def test_subchunk_edge_budget_scales_with_concepts():
    # two concepts in chunk 10, k=1 per sub-chunk: each concept keeps its own
    # edge (2 edges total) — under a per-CHUNK k=1 cap one would be displaced
    parents = [10, 10, 20, 30]
    matrix = np.asarray(
        [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
    )
    cvecs = {10: [R2, R2, 0, 0], 20: [1, 0, 0, 0], 30: [0, 1, 0, 0]}
    edges = subchunk_knn_edges(parents, matrix, [10], cvecs, k=1, min_sim=0.5, near_dup_sim=2.0)
    assert dict(edges[10]) == {20: pytest.approx(1.0), 30: pytest.approx(1.0)}


def test_subchunk_src_without_subchunks_gets_no_edges():
    assert _sub_edges([99]) == {99: []}


def test_subchunk_empty_inputs():
    assert _sub_edges([]) == {}
    assert subchunk_knn_edges([], np.zeros((0, 4), dtype=np.float32), [1], {}, 8, 0.5, 0.9) == {}
