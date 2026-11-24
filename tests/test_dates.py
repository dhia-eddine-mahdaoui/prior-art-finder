from priorart.pipeline import run_search


def test_date_filter_excludes_late_filings(index, samples):
    s = samples["D1"]
    m = index.mask(None, s["priority_date"])
    assert 0 < m.sum() < len(index.docs)
    for i, d in enumerate(index.docs):
        assert m[i] == (d["filing_date"] < s["priority_date"])


def test_same_day_filing_is_not_prior_art(index):
    assert not index.mask(None, index.docs[0]["filing_date"])[0]


def test_search_results_all_before_priority_date(index, samples):
    for s in samples.values():
        res = run_search(s["text"], index, s["priority_date"], top_k=10)
        assert res["results"]
        assert all(r["filing_date"] < s["priority_date"] for r in res["results"])


def test_late_strong_matches_exist_but_are_filtered(index, samples):
    s = samples["D2"]
    unfiltered = run_search(s["text"], index, None, top_k=10, mode="bm25")
    assert any(r["filing_date"] >= s["priority_date"] for r in unfiltered["results"])
    filtered = run_search(s["text"], index, s["priority_date"], top_k=10, mode="bm25")
    assert all(r["filing_date"] < s["priority_date"] for r in filtered["results"])
