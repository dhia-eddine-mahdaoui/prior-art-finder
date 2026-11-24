import math

from priorart.calibrate import align, gold_pairs, load_gold, mapping_eval
from priorart.evals import average_precision, evaluate, mrr, ndcg_at_k, recall_at_k
from priorart.features import HeuristicExtractor
from priorart.mapping import MapCfg


def test_recall_at_k():
    rel = {"a": 3, "b": 2, "c": 0}
    assert recall_at_k(["x", "a", "y", "b"], rel, 2) == 0.5
    assert recall_at_k(["x", "a", "y", "b"], rel, 10) == 1.0
    assert recall_at_k(["x"], {}, 10) == 0.0


def test_mrr():
    assert mrr(["x", "y", "a"], {"a": 1}) == 1 / 3
    assert mrr(["a"], {"a": 1}) == 1.0
    assert mrr(["x"], {"a": 1}) == 0.0


def test_average_precision():
    assert average_precision(["a", "x", "b"], {"a": 1, "b": 1}) == (1 + 2 / 3) / 2
    assert average_precision(["x"], {}) == 0.0


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
        assert r["n_queries"] == 16
        for v in r["mean"].values():
            assert 0.0 <= v <= 1.0
    assert next(iter(evaluate(index, split="test").values()))["n_queries"] == 8


def test_gold_alignment_finds_each_feature(index, samples):
    gold = load_gold()
    for qid in ("D1", "D6", "D12"):
        al = align(HeuristicExtractor().extract(samples[qid]["text"]), gold[qid]["features"])
        assert len(al) >= 5


def test_mapping_accuracy_is_calibrated_and_better_than_overlap(index):
    cal = mapping_eval(index, index.ensure_mapper().cfg, "test")
    old = mapping_eval(index, MapCfg(), "test")
    assert cal["pairs"] > 1000
    assert cal["covered_precision"] >= 0.9 and cal["false_covered_rate"] <= 0.02
    assert cal["covered_recall"] > old["covered_recall"] and cal["detect_f1"] >= old["detect_f1"]
