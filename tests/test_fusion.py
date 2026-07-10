from __future__ import annotations

from ragx.core.fusion import rrf
from ragx.core.scoring import combine, normalize


def test_rrf_hand_computed():
    rankings = [[1, 2, 3], [2, 1, 4]]
    scores = rrf(rankings, k=60)
    assert scores[1] == 1 / 61 + 1 / 62
    assert scores[2] == 1 / 62 + 1 / 61
    assert scores[3] == 1 / 63
    assert scores[4] == 1 / 63


def test_rrf_stable_ordering_ties_broken_by_first_ranking():
    # doc 2 and doc 3 tie in score across two symmetric rankings; original insertion order
    # of the dict should reflect first-seen order, which downstream sorts can rely on.
    rankings = [[2, 3], [3, 2]]
    scores = rrf(rankings, k=60)
    assert list(scores.keys()) == [2, 3]
    assert scores[2] == scores[3]


def test_rrf_empty():
    assert rrf([]) == {}
    assert rrf([[]]) == {}


def test_normalize_empty():
    assert normalize({}) == {}


def test_normalize_all_equal():
    assert normalize({1: 5.0, 2: 5.0, 3: 5.0}) == {1: 1.0, 2: 1.0, 3: 1.0}


def test_normalize_min_max():
    result = normalize({1: 0.0, 2: 5.0, 3: 10.0})
    assert result == {1: 0.0, 2: 0.5, 3: 1.0}


def test_combine_with_rerank_exact():
    candidates = [1, 2, 3]
    vector = {1: 0.0, 2: 5.0, 3: 10.0}
    heat = {1: 1.0, 2: 1.0, 3: 1.0}
    rerank = {1: 2.0, 2: 4.0, 3: 6.0}
    result = combine(candidates, vector, heat, rerank, alpha=0.5, beta=0.3, gamma=0.2)
    # normalized: vector -> 0, 0.5, 1.0; heat -> all 1.0; rerank -> 0, 0.5, 1.0
    assert result[1] == 0.5 * 0.0 + 0.3 * 1.0 + 0.2 * 0.0
    assert result[2] == 0.5 * 0.5 + 0.3 * 1.0 + 0.2 * 0.5
    assert result[3] == 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 1.0


def test_combine_without_rerank_renormalizes_weights():
    candidates = [1, 2]
    vector = {1: 0.0, 2: 10.0}
    heat = {1: 0.0, 2: 10.0}
    result = combine(candidates, vector, heat, None, alpha=0.5, beta=0.3, gamma=0.3)
    # beta/(beta+gamma) = 0.5, gamma/(beta+gamma) = 0.5
    assert result[1] == 0.0
    assert result[2] == 0.5 * 1.0 + 0.5 * 1.0


def test_combine_missing_candidate_scores_default_to_zero_raw():
    candidates = [1, 2, 3]
    vector = {1: 10.0}  # 2 and 3 missing -> raw 0.0
    heat = {1: 10.0, 2: 10.0, 3: 10.0}  # all-equal -> normalized to 1.0
    result = combine(candidates, vector, heat, None, alpha=0.0, beta=0.0, gamma=1.0)
    assert result[1] == 1.0
    assert result[2] == 0.0
    assert result[3] == 0.0
