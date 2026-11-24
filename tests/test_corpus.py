import json

from priorart.corpus_gen import build_corpus
from priorart.retrieval import DATA


def test_generator_is_deterministic():
    a, qa, ga = build_corpus(7)
    b, qb, gb = build_corpus(7)
    assert a == b and qa == qb and ga == gb


def test_committed_corpus_matches_generator():
    docs, qrels, _ = build_corpus(7)
    stored = [json.loads(l) for l in (DATA / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    assert stored == docs
    assert json.loads((DATA / "qrels.json").read_text()) == qrels


def test_labels_reference_real_documents(index):
    qrels = json.loads((DATA / "qrels.json").read_text())
    assert len(qrels) == 16
    for g in qrels.values():
        assert set(g) <= set(index.by_id)
        assert sum(1 for v in g.values() if v > 0) == 8
    assert all(d["id"].startswith("SYN-") for d in index.docs)


def test_dev_test_split_is_balanced_and_disjoint():
    samples = json.loads((DATA / "disclosures.json").read_text())
    dev = {s["id"] for s in samples if s["split"] == "dev"}
    test = {s["id"] for s in samples if s["split"] == "test"}
    assert len(dev) == 8 and len(test) == 8 and not dev & test


def test_tuning_metadata_says_dev_only():
    tuned = json.loads((DATA / "tuned.json").read_text())
    assert "dev" in tuned["dev_only"] and "classical" in tuned


def test_gold_features_cover_every_disclosure():
    gold = json.loads((DATA / "feature_gold.json").read_text())
    assert len(gold) == 16 and all(len(g["features"]) == 6 for g in gold.values())
