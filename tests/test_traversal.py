from __future__ import annotations

import pytest

from ragx.core.traversal import propagate_heat


def make_neighbors_fn(adjacency: dict[int, list[tuple[int, float]]]):
    def fn(node: int) -> list[tuple[int, float]]:
        return adjacency.get(node, [])

    return fn


def full_sim(sim: dict[int, float]):
    def fn(ids):
        return {i: sim.get(i, 1.0) for i in ids}

    return fn


def test_hops_zero_returns_seeds_only():
    result = propagate_heat(
        {1: 1.0},
        make_neighbors_fn({1: [(2, 0.9)]}),
        full_sim({}),
        hops=0,
    )
    assert result.heat == {1: 1.0}
    assert result.trace == {1: {"seed": 1, "parent": None, "edge_weight": 0.0, "hop": 0}}


def test_per_hop_decay_exact_chain():
    adjacency = {1: [(2, 0.8)], 2: [(3, 0.8)]}
    result = propagate_heat(
        {1: 1.0},
        make_neighbors_fn(adjacency),
        full_sim({}),
        hops=2,
        decay=0.5,
        query_floor=0.0,
    )
    assert result.heat[2] == pytest.approx(0.4)
    assert result.heat[3] == pytest.approx(0.16)
    assert result.trace[2] == {"seed": 1, "parent": 1, "edge_weight": 0.8, "hop": 1}
    assert result.trace[3] == {"seed": 1, "parent": 2, "edge_weight": 0.8, "hop": 2}


def test_max_not_sum_at_hub_node():
    # Two seeds both point at hub node 3; the higher contribution wins, not the sum.
    adjacency = {1: [(3, 0.9)], 2: [(3, 0.2)]}
    result = propagate_heat(
        {1: 1.0, 2: 1.0},
        make_neighbors_fn(adjacency),
        full_sim({}),
        hops=1,
        decay=0.5,
        query_floor=0.0,
    )
    expected = max(1.0 * 0.9 * 0.5, 1.0 * 0.2 * 0.5)
    assert result.heat[3] == expected
    assert result.trace[3]["parent"] == 1


def test_query_floor_excludes_and_prevents_relay():
    # Node 2 fails the query floor: it must not receive heat, and must not relay to node 3.
    adjacency = {1: [(2, 0.9)], 2: [(3, 0.9)]}
    result = propagate_heat(
        {1: 1.0},
        make_neighbors_fn(adjacency),
        full_sim({2: 0.1}),
        hops=2,
        decay=0.5,
        query_floor=0.35,
    )
    assert 2 not in result.heat
    assert 3 not in result.heat


def test_query_floor_allows_above_threshold():
    adjacency = {1: [(2, 0.9)]}
    result = propagate_heat(
        {1: 1.0},
        make_neighbors_fn(adjacency),
        full_sim({2: 0.5}),
        hops=1,
        decay=0.5,
        query_floor=0.35,
    )
    assert result.heat[2] == 1.0 * 0.9 * 0.5


def test_seed_exempt_from_query_floor():
    # Seed 1 has heat set up front regardless of query similarity; floor never checked for it.
    result = propagate_heat(
        {1: 1.0},
        make_neighbors_fn({}),
        full_sim({1: 0.0}),
        hops=1,
        query_floor=0.35,
    )
    assert result.heat[1] == 1.0


def test_frontier_cap_keeps_top_n_by_heat_ties_by_lower_id():
    adjacency = {
        1: [(2, 0.9), (3, 0.5), (4, 0.5)],
    }
    result = propagate_heat(
        {1: 1.0},
        make_neighbors_fn(adjacency),
        full_sim({}),
        hops=2,
        decay=0.5,
        query_floor=0.0,
        max_frontier=2,
    )
    # All three receive heat in hop 1 (heat kept), but only top 2 (node 2, then tie 3 vs 4 -> 3)
    # propagate into hop 2. Give node 4 (dropped from frontier) an onward edge to verify it
    # doesn't relay while node 3 (kept) does.
    adjacency[3] = [(5, 0.9)]
    adjacency[4] = [(6, 0.9)]
    result = propagate_heat(
        {1: 1.0},
        make_neighbors_fn(adjacency),
        full_sim({}),
        hops=2,
        decay=0.5,
        query_floor=0.0,
        max_frontier=2,
    )
    assert 2 in result.heat and 3 in result.heat and 4 in result.heat
    assert 5 in result.heat  # node 3 (kept in capped frontier) relayed onward
    assert 6 not in result.heat  # node 4 (dropped by cap) did not relay


def test_trace_parent_chain_reaches_a_seed():
    adjacency = {1: [(2, 0.8)], 2: [(3, 0.8)]}
    result = propagate_heat(
        {1: 1.0},
        make_neighbors_fn(adjacency),
        full_sim({}),
        hops=2,
        query_floor=0.0,
    )
    node = 3
    seen = set()
    while result.trace[node]["parent"] is not None:
        assert node not in seen
        seen.add(node)
        node = result.trace[node]["parent"]
    assert result.trace[node]["seed"] == node
    assert node == 1


def test_empty_seeds():
    result = propagate_heat({}, make_neighbors_fn({}), full_sim({}))
    assert result.heat == {}
    assert result.trace == {}
