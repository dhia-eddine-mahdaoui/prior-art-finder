import json

from priorart.features import Feature, HeuristicExtractor
from priorart.retrieval import COVERED, DATA, MODES
from priorart.text import analyze


def top_ids(index, text, mode, mask=None, k=10, feats=None):
    return [index.docs[h["idx"]]["id"] for h in index.search(text, mode=mode, mask=mask, k=k, features=feats)]


def feat(text):
    return Feature("F1", text, sorted(set(analyze(text))))


def test_corpus_size_and_shape(index):
    assert 150 <= len(index.docs) <= 300
    assert {"id", "title", "abstract", "claims", "cpc", "filing_date", "assignee"} <= set(index.docs[0])


def test_every_mode_returns_k_results(index, samples):
    text = samples["D1"]["text"]
    feats = HeuristicExtractor().extract(text)
    for m in MODES:
        assert len(top_ids(index, text, m, feats=feats)) == 10


def test_close_prior_art_is_retrieved(index, samples):
    qrels = json.loads((DATA / "qrels.json").read_text())
    for sid in samples:
        text = samples[sid]["text"]
        mask = index.mask(None, samples[sid]["priority_date"])
        got = set(top_ids(index, text, "hybrid_rerank", mask, 10, HeuristicExtractor().extract(text)))
        close = {d for d, g in qrels[sid].items() if g == 3}
        assert len(close & got) >= 4, (sid, close - got)


def test_exact_document_text_ranks_itself_first(index):
    d = index.docs[17]
    assert top_ids(index, d["title"] + " " + d["abstract"], "bm25", k=1) == [d["id"]]


def test_class_filter(index, samples):
    m = index.mask(["H01M"])
    assert 0 < m.sum() < len(index.docs)
    for h in index.search(samples["D1"]["text"], "bm25", m, 20):
        assert any(c.replace(" ", "").startswith("H01M") for c in index.docs[h["idx"]]["cpc"])


def test_mapping_marks_verbatim_feature_covered(index):
    d = next(x for x in index.docs if "serpentine microchannels" in " ".join(x["claims"]))
    row = index.map_features(index.by_id[d["id"]],
                             [feat("a cooling plate having serpentine microchannels through which coolant circulates")])[0]
    assert row["status"] == "covered" and row["score"] >= COVERED
    assert any(s["hit"] for s in row["segments"])


def test_mapping_not_found_for_unrelated_feature(index):
    row = index.map_features(0, [feat("an ultraviolet sterilisation lamp inside the water reservoir")])[0]
    assert row["status"] == "not found" and row["passage"] == ""
