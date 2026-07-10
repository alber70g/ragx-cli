import pytest

from ragx.core.graph import affected_ids, knn_edges
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
