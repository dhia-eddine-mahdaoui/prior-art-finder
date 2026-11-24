import json

from priorart.features import HeuristicExtractor, get_extractor


def test_extracts_expected_features(samples):
    feats = HeuristicExtractor().extract(samples["D1"]["text"])
    texts = " | ".join(f.text.lower() for f in feats)
    assert 5 <= len(feats) <= 8
    for needle in ("phase-change", "serpentine", "predictive model", "thermal runaway", "heat pump"):
        assert needle in texts
    assert not any(t.text.lower().startswith("we propose") for t in feats)  # preamble dropped


def test_features_are_deterministic_and_have_stems(samples):
    a = HeuristicExtractor().extract(samples["D2"]["text"])
    b = HeuristicExtractor().extract(samples["D2"]["text"])
    assert [f.to_dict() for f in a] == [f.to_dict() for f in b]
    assert all(len(f.stems) >= 4 for f in a)


def test_near_duplicate_clauses_are_merged():
    text = ("A sealed housing keeps moisture out of the sensor electronics. "
            "A sealed housing keeps moisture out of the sensor electronics.")
    assert len(HeuristicExtractor().extract(text)) == 1


def test_default_provider_is_heuristic(monkeypatch):
    monkeypatch.delenv("PRIORART_FEATURE_PROVIDER", raising=False)
    assert get_extractor().name == "heuristic"


def test_llm_provider_parses_response_and_falls_back(monkeypatch, samples):
    monkeypatch.setenv("PRIORART_FEATURE_PROVIDER", "llm")
    monkeypatch.setenv("PRIORART_LLM_BASE_URL", "http://llm.invalid")
    monkeypatch.setenv("PRIORART_LLM_MODEL", "m")
    ex = get_extractor()
    assert ex.name == "llm"
    reply = ["a cooling plate with microchannels for coolant", "a heat pump that pre-heats the pack in cold weather"]
    monkeypatch.setattr(ex, "_post", lambda payload: {"choices": [{"message": {"content": json.dumps(reply)}}]})
    assert [f.text for f in ex.extract("whatever")][0].startswith("A cooling plate")

    def boom(payload):
        raise OSError("network down")
    monkeypatch.setattr(ex, "_post", boom)
    assert len(ex.extract(samples["D1"]["text"])) >= 5  # falls back to heuristic
