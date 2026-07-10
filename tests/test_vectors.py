import random

import pytest

from ragx.core.errors import RagxError
from ragx.core.vectors import VectorIndex

DIM = 8


def _cluster_vector(seed: int, base: list[float]) -> list[float]:
    rng = random.Random(seed)
    return [b + rng.uniform(-0.01, 0.01) for b in base]


def _two_clusters(n_per_cluster: int = 10) -> tuple[list[int], list[list[float]], list[float], list[float]]:
    base_a = [1.0] * DIM
    base_b = [-1.0] * DIM
    ids: list[int] = []
    vecs: list[list[float]] = []
    for i in range(n_per_cluster):
        ids.append(i)
        vecs.append(_cluster_vector(i, base_a))
    for i in range(n_per_cluster):
        ids.append(1000 + i)
        vecs.append(_cluster_vector(i, base_b))
    return ids, vecs, base_a, base_b


def test_nearest_neighbor_sanity(tmp_path):
    idx = VectorIndex.create(tmp_path / "v.hnsw", dim=DIM, capacity=64)
    ids, vecs, base_a, base_b = _two_clusters()
    idx.add(ids, vecs)

    results = idx.search(base_a, k=5)
    assert len(results) == 5
    assert all(rid < 1000 for rid, _ in results)
    # sorted desc by similarity
    sims = [s for _, s in results]
    assert sims == sorted(sims, reverse=True)

    results_b = idx.search(base_b, k=5)
    assert all(rid >= 1000 for rid, _ in results_b)


def test_resize_past_initial_capacity(tmp_path):
    idx = VectorIndex.create(tmp_path / "v.hnsw", dim=DIM, capacity=4)
    ids = list(range(20))
    vecs = [[float(i)] * DIM for i in ids]
    idx.add(ids, vecs)
    assert idx.count == 20
    results = idx.search([0.0] * DIM, k=3)
    assert len(results) == 3


def test_delete_then_search_exclusion(tmp_path):
    idx = VectorIndex.create(tmp_path / "v.hnsw", dim=DIM, capacity=64)
    ids, vecs, base_a, _ = _two_clusters()
    idx.add(ids, vecs)

    before = idx.count
    idx.mark_deleted([0, 1, 2])
    assert idx.count == before - 3

    results = idx.search(base_a, k=10)
    result_ids = {rid for rid, _ in results}
    assert not ({0, 1, 2} & result_ids)

    # deleting unknown ids is a no-op
    idx.mark_deleted([99999])
    assert idx.count == before - 3


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "v.hnsw"
    idx = VectorIndex.create(path, dim=DIM, capacity=64)
    ids, vecs, base_a, base_b = _two_clusters()
    idx.add(ids, vecs)
    idx.mark_deleted([0, 1])
    before_results = idx.search(base_a, k=5)
    before_count = idx.count
    idx.save()

    loaded = VectorIndex.load(path, dim=DIM)
    assert loaded.count == before_count
    after_results = loaded.search(base_a, k=5)
    assert after_results == before_results

    result_ids = {rid for rid, _ in after_results}
    assert not ({0, 1} & result_ids)


def test_empty_index_search(tmp_path):
    idx = VectorIndex.create(tmp_path / "v.hnsw", dim=DIM)
    assert idx.count == 0
    assert idx.search([0.0] * DIM, k=5) == []


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(RagxError):
        VectorIndex.load(tmp_path / "missing.hnsw", dim=DIM)


def test_get_vectors_roundtrip(tmp_path):
    idx = VectorIndex.create(tmp_path / "v.hnsw", dim=4)
    idx.add([10, 20], [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    vecs = idx.get_vectors([20, 10])
    assert len(vecs) == 2
    # cosine space: hnswlib stores normalized vectors; direction must be preserved
    assert vecs[0][1] > 0.99 and vecs[1][0] > 0.99
    with pytest.raises(RagxError):
        idx.get_vectors([999])
