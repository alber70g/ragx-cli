from __future__ import annotations

from ragx.core.communities import leiden_communities

# Two dense triangles (1,2,3) and (4,5,6) joined by one weak bridge edge.
TRIANGLES = [
    (1, 2, 0.95), (1, 3, 0.95), (2, 3, 0.95),
    (4, 5, 0.95), (4, 6, 0.95), (5, 6, 0.95),
    (3, 4, 0.05),
]


def test_two_cluster_partition():
    result = leiden_communities(TRIANGLES)
    assert result[1] == result[2] == result[3]
    assert result[4] == result[5] == result[6]
    assert result[1] != result[4]


def test_deterministic_and_order_independent():
    shuffled = list(reversed(TRIANGLES))
    a = leiden_communities(TRIANGLES)
    b = leiden_communities(shuffled)
    assert a == b


def test_empty_edges_returns_empty():
    assert leiden_communities([]) == {}


def test_community_ids_consecutive_from_zero():
    result = leiden_communities(TRIANGLES)
    ids = set(result.values())
    assert ids == set(range(len(ids)))
    smallest_chunk_community = result[min(result)]
    assert smallest_chunk_community == 0
