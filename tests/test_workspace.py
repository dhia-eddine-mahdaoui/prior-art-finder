import csv
import io

from fastapi.testclient import TestClient

from priorart import app as app_module, pipeline


def client(index, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "VAR", tmp_path)
    monkeypatch.setattr(app_module, "index", lambda: index)
    return TestClient(app_module.app)


def test_disclosure_workflow_end_to_end(index, samples, tmp_path, monkeypatch):
    c = client(index, tmp_path, monkeypatch)
    s = samples["D1"]
    p = c.post("/api/projects", json={"name": "Pack thermal", "text": s["text"], "priority_date": s["priority_date"],
                                      "classes": ["H01M"]}).json()
    pid = p["id"]
    # features are edited by the reviewer and used by the search
    feats = [f["text"] for f in c.post("/api/features", json={"text": s["text"]}).json()["features"]]
    assert len(feats) >= 5
    c.put(f"/api/projects/{pid}", json={"disclosure": {"features": feats[:4]}})
    res = c.post(f"/api/projects/{pid}/search", json={"top_k": 5}).json()
    assert len(res["features"]) == 4 and len(res["results"]) == 5
    assert res["excluded_by_date"] > 0 and res["after_priority"]
    assert all(r["filing_date"] < s["priority_date"] for r in res["results"])
    assert len(res["project"]["searches"]) == 1
    # shortlist two references, mark one relevant (also saved as relevance feedback)
    a, b = res["results"][0]["id"], res["results"][1]["id"]
    c.put(f"/api/projects/{pid}/shortlist", json={"doc_id": a, "decision": "maybe"})
    c.put(f"/api/projects/{pid}/shortlist", json={"doc_id": b, "decision": "relevant", "note": "check claim 2"})
    assert c.put(f"/api/projects/{pid}/shortlist", json={"doc_id": b, "decision": "bogus"}).status_code == 400
    assert any(r["label"] == "relevant" and r["doc_id"] == b for r in pipeline.read_feedback())
    chart = c.get(f"/api/projects/{pid}/chart").json()
    assert [r["id"] for r in chart["references"]] == [a, b] and len(chart["combined"]) == 4
    md = c.get(f"/api/projects/{pid}/chart?format=md").text
    assert "Claim chart: Pack thermal" in md and a in md and "check claim 2" in md and "Not legal advice" in md
    rows = list(csv.DictReader(io.StringIO(c.get(f"/api/projects/{pid}/chart?format=csv").text)))
    assert len(rows) == 8 and {r["status"] for r in rows} <= {"covered", "partial", "not found"}
    # an excluded reference drops out of the chart
    c.put(f"/api/projects/{pid}/shortlist", json={"doc_id": a, "decision": "not_relevant"})
    assert [r["id"] for r in c.get(f"/api/projects/{pid}/chart").json()["references"]] == [b]
    assert c.delete(f"/api/projects/{pid}/shortlist/{b}").status_code == 200
    assert c.get("/api/projects").json()[0]["searches"] == 1
    assert c.delete(f"/api/projects/{pid}").status_code == 200 and c.get(f"/api/projects/{pid}").status_code == 404


def test_search_with_edited_features_and_validation(index, samples, tmp_path, monkeypatch):
    c = client(index, tmp_path, monkeypatch)
    r = c.post("/api/search", json={"text": samples["D2"]["text"], "features": ["a piezoelectric energy harvester powering a node"]})
    assert r.status_code == 200 and len(r.json()["features"]) == 1
    pid = c.post("/api/projects", json={"name": "empty"}).json()["id"]
    assert c.post(f"/api/projects/{pid}/search", json={}).status_code == 400
    assert c.get("/api/projects/nope").status_code == 404
    assert c.get("/api/projects/nope/chart").status_code == 404


def test_eval_report_served(index, tmp_path, monkeypatch):
    c = client(index, tmp_path, monkeypatch)
    r = c.get("/api/eval").json()
    assert r["retrieval"]["tuned"]["test"]["hybrid"]["recall@10"] >= r["retrieval"]["tuned"]["test"]["bm25"]["recall@10"]
