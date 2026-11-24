from fastapi.testclient import TestClient

from priorart import app as app_module, pipeline


def client(index, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "VAR", tmp_path)
    monkeypatch.setattr(app_module, "index", lambda: index)
    return TestClient(app_module.app)


def test_search_feedback_and_report(index, samples, tmp_path, monkeypatch):
    c = client(index, tmp_path, monkeypatch)
    s = samples["D3"]
    body = {"text": s["text"], "priority_date": s["priority_date"], "classes": ["C09D"]}
    r = c.post("/api/search", json=body).json()
    assert r["results"] and r["features"] and r["query_id"] == "D3"
    doc = r["results"][0]["id"]
    assert c.post("/api/feedback", json={"text": s["text"], "doc_id": doc, "label": "relevant"}).status_code == 200
    assert c.post("/api/search", json=body).json()["results"][0]["feedback"] == "relevant"
    md = c.post("/api/report", json=body).json()["markdown"]
    assert "Prior-art search report" in md and doc in md and "Not legal advice" in md
    assert c.post("/api/feedback", json={"text": s["text"], "doc_id": "nope", "label": "relevant"}).status_code == 404


def test_feedback_extends_eval_set(index, samples, tmp_path, monkeypatch):
    from priorart.evals import load_qrels
    c = client(index, tmp_path, monkeypatch)
    custom = "A pump housing includes a ceramic bearing sleeve and a pressure sensor that adjusts the motor speed during start-up."
    c.post("/api/feedback", json={"text": custom, "doc_id": index.docs[3]["id"], "label": "relevant"})
    queries, qrels = load_qrels(with_feedback=True)
    extra = [q for q in queries if q not in samples]
    assert len(extra) == 1 and qrels[extra[0]][index.docs[3]["id"]] == 2


def test_frontend_served(index, tmp_path, monkeypatch):
    assert "Prior-art finder" in client(index, tmp_path, monkeypatch).get("/").text
