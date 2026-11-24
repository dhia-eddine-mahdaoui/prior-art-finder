import json

from priorart.features import Feature, HeuristicExtractor
from priorart.retrieval import DATA, MODES
from priorart.text import analyze, core_text, highlight_segments


def top_ids(index, text, mode, mask=None, k=10, feats=None):
    return [index.docs[h["idx"]]["id"] for h in index.search(text, mode=mode, mask=mask, k=k, features=feats)]


def feat(text):
    return Feature("F1", text, sorted(set(analyze(text))))


def test_corpus_size_and_shape(index):
    assert 400 <= len(index.docs) <= 600
    assert {"id", "title", "abstract", "claims", "cpc", "filing_date", "assignee"} <= set(index.docs[0])


def test_every_mode_returns_k_results(index, samples):
    text = samples["D1"]["text"]
    feats = HeuristicExtractor().extract(text)
    for m in index.modes():
        assert len(top_ids(index, text, m, feats=feats)) == 10
    assert "dense" in MODES and "dense" not in index.modes()      # optional, off by default


def test_close_prior_art_is_retrieved(index, samples):
    qrels = json.loads((DATA / "qrels.json").read_text())
    for sid in samples:
        text = samples[sid]["text"]
        mask = index.mask(None, samples[sid]["priority_date"])
        got = set(top_ids(index, text, "hybrid_rerank", mask, 10, HeuristicExtractor().extract(text)))
        close = {d for d, g in qrels[sid].items() if g == 3}
        assert len(close & got) >= 4, (sid, close - got)


def test_hybrid_beats_bm25_on_held_out_test_queries(index):
    from priorart.evals import evaluate
    r = evaluate(index, split="test", modes=["bm25", "hybrid"])
    assert r["hybrid"]["mean"]["recall@10"] >= r["bm25"]["mean"]["recall@10"]
    assert r["hybrid"]["mean"]["ndcg@10"] >= r["bm25"]["mean"]["ndcg@10"]


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
    assert row["status"] == "covered" and row["score"] >= index.ensure_mapper().cfg.covered
    assert any(s["hit"] for s in row["segments"])


def test_mapping_not_found_for_unrelated_feature(index):
    row = index.map_features(0, [feat("an ultraviolet sterilisation lamp inside the water reservoir")])[0]
    assert row["status"] == "not found" and row["passage"] == ""


def test_claim_one_is_split_into_elements(index):
    kinds = {p["kind"] for rows in index.passages for p in rows}
    assert "claim 1(a)" in kinds and "claim 1(b)" in kinds


def test_text_normalisation():
    assert analyze("forty percent") == analyze("40%") == analyze("40 per cent")
    assert analyze("five micrometres") == analyze("5 microns")
    assert analyze("charging") == analyze("charge")
    assert core_text("The battery pack of claim 1, further comprising a cold plate.") == "a cold plate."
    segs = highlight_segments("a cold plate with coolant", set(analyze("cold plate")), set(analyze("coolant")))
    assert [s["soft"] for s in segs if s["hit"]] == [False, True]
