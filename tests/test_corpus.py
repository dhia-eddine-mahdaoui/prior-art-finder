import json

from priorart.corpus_gen import build_corpus
from priorart.retrieval import DATA


def test_generator_is_deterministic():
    a, qa = build_corpus(7)
    b, qb = build_corpus(7)
    assert a == b and qa == qb


def test_committed_corpus_matches_generator():
    docs, qrels = build_corpus(7)
    stored = [json.loads(l) for l in (DATA / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    assert stored == docs
    assert json.loads((DATA / "qrels.json").read_text()) == qrels


def test_labels_reference_real_documents(index):
    qrels = json.loads((DATA / "qrels.json").read_text())
    for g in qrels.values():
        assert set(g) <= set(index.by_id)
        assert sum(1 for v in g.values() if v > 0) == 8
    assert all(d["id"].startswith("SYN-") for d in index.docs)
