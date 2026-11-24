import math

from priorart.evals import evaluate, mrr, ndcg_at_k, recall_at_k


def test_recall_at_k():
    rel = {"a": 3, "b": 2, "c": 0}
    assert recall_at_k(["x", "a", "y", "b"], rel, 2) == 0.5
    assert recall_at_k(["x", "a", "y", "b"], rel, 10) == 1.0
    assert recall_at_k(["x"], {}, 10) == 0.0


def test_mrr():
    assert mrr(["x", "y", "a"], {"a": 1}) == 1 / 3
    assert mrr(["a"], {"a": 1}) == 1.0
    assert mrr(["x"], {"a": 1}) == 0.0


def test_ndcg_known_values():
    rel = {"a": 3, "b": 1}
    assert ndcg_at_k(["a", "b"], rel) == 1.0
    worse = ndcg_at_k(["b", "a"], rel)
    expected = (1 / 1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3))
    assert abs(worse - expected) < 1e-9 and worse < 1.0
    assert ndcg_at_k(["x", "y"], rel) == 0.0


def test_evaluate_runs_and_bounds(index):
    res = evaluate(index)
    for r in res.values():
        assert r["n_queries"] == 4
        for v in r["mean"].values():
            assert 0.0 <= v <= 1.0
