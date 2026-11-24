"""FastAPI app: JSON API plus a single-page frontend."""
from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import pipeline, workspace
from .features import get_extractor
from .retrieval import DATA, MODES, get_index

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Prior-art finder (prototype)")


@lru_cache(maxsize=1)
def index():
    return get_index()


class SearchReq(BaseModel):
    text: str = Field(min_length=20, max_length=20000)
    priority_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    classes: list[str] = []
    top_k: int = Field(default=10, ge=1, le=30)
    mode: str = "hybrid_rerank"
    features: list[str] | None = None


class FeedbackReq(BaseModel):
    text: str = Field(min_length=20)
    doc_id: str
    label: str


class FeaturesReq(BaseModel):
    text: str = Field(min_length=20, max_length=20000)


class ProjectNew(BaseModel):
    name: str = "Untitled disclosure"
    text: str = ""
    priority_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    classes: list[str] = []


class DisclosureEdit(BaseModel):
    text: str | None = None
    priority_date: str | None = None
    classes: list[str] | None = None
    features: list[str] | None = None


class ProjectEdit(BaseModel):
    name: str | None = None
    disclosure: dict | None = None


class ProjectSearch(BaseModel):
    top_k: int = Field(default=10, ge=1, le=30)
    mode: str = "hybrid_rerank"
    priority_date: str | None = None
    classes: list[str] | None = None


class Decision(BaseModel):
    doc_id: str
    decision: str | None = None
    note: str | None = None


@app.get("/api/samples")
def samples():
    return pipeline.load_samples()


@app.get("/api/meta")
def meta():
    ix = index()
    classes = Counter(c.split()[0] for d in ix.docs for c in set(x.split()[0] for x in d["cpc"]))
    dates = sorted(d["filing_date"] for d in ix.docs)
    return {"documents": len(ix.docs), "classes": sorted(classes.items()), "modes": ix.modes(),
            "date_range": [dates[0], dates[-1]], "dense": ix.dense_docs is not None}


@app.post("/api/features")
def features(req: FeaturesReq):
    return {"features": [f.to_dict() for f in get_extractor().extract(req.text)]}


@app.post("/api/search")
def search(req: SearchReq):
    if req.mode not in MODES:
        raise HTTPException(400, "unknown mode")
    return pipeline.run_search(req.text, index(), req.priority_date, req.classes, req.top_k, req.mode, req.features)


@app.post("/api/feedback")
def feedback(req: FeedbackReq):
    if req.doc_id not in index().by_id:
        raise HTTPException(404, "unknown document")
    try:
        return pipeline.save_feedback(req.text, req.doc_id, req.label)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/report")
def report(req: SearchReq):
    res = pipeline.run_search(req.text, index(), req.priority_date, req.classes, req.top_k, req.mode, req.features)
    return {"markdown": pipeline.markdown_report(req.text, res, req.priority_date, req.classes)}


# ---- projects: draft -> search -> shortlist -> claim chart ----
def _project(pid: str) -> dict:
    p = workspace.get(pid)
    if p is None:
        raise HTTPException(404, "unknown project")
    return p


@app.get("/api/projects")
def projects():
    return workspace.list_projects()


@app.post("/api/projects")
def project_new(req: ProjectNew):
    return workspace.create(req.name, req.text, req.priority_date, req.classes)


@app.get("/api/projects/{pid}")
def project_get(pid: str):
    return _project(pid)


@app.put("/api/projects/{pid}")
def project_edit(pid: str, req: ProjectEdit):
    _project(pid)
    return workspace.update(pid, req.name, req.disclosure)


@app.delete("/api/projects/{pid}")
def project_delete(pid: str):
    if not workspace.delete(pid):
        raise HTTPException(404, "unknown project")
    return {"ok": True}


@app.post("/api/projects/{pid}/search")
def project_search(pid: str, req: ProjectSearch):
    p = _project(pid)
    d = p["disclosure"]
    if len(d["text"].strip()) < 20:
        raise HTTPException(400, "write the disclosure first (at least a sentence)")
    if req.mode not in MODES:
        raise HTTPException(400, "unknown mode")
    date = req.priority_date if req.priority_date is not None else d["priority_date"]
    classes = req.classes if req.classes is not None else d["classes"]
    res = pipeline.run_search(d["text"], index(), date or None, classes, req.top_k, req.mode, d.get("features"))
    workspace.update(pid, disclosure={"priority_date": date or None, "classes": classes})
    workspace.record_search(pid, res, {"priority_date": date or None, "classes": classes, "mode": res["mode"], "top_k": req.top_k})
    res["project"] = workspace.get(pid)
    return res


@app.put("/api/projects/{pid}/shortlist")
def project_shortlist(pid: str, req: Decision):
    p = _project(pid)
    if req.doc_id not in index().by_id:
        raise HTTPException(404, "unknown document")
    try:
        out = workspace.set_decision(pid, req.doc_id, req.decision, req.note)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if req.decision in ("relevant", "not_relevant"):   # decisions also feed the eval set
        pipeline.save_feedback(p["disclosure"]["text"], req.doc_id, req.decision)
    return out


@app.delete("/api/projects/{pid}/shortlist/{doc_id}")
def project_unshortlist(pid: str, doc_id: str):
    _project(pid)
    return workspace.remove_from_shortlist(pid, doc_id)


@app.get("/api/projects/{pid}/chart")
def project_chart(pid: str, format: str = "json"):
    p = _project(pid)
    chart = workspace.build_chart(p, index())
    if format == "json":
        return chart
    if format == "md":
        return PlainTextResponse(workspace.chart_markdown(p, chart), media_type="text/markdown")
    if format == "csv":
        return PlainTextResponse(workspace.chart_csv(chart), media_type="text/csv")
    raise HTTPException(400, "format must be json, md or csv")


# ---- evaluation report (rendered by the UI) ----
@app.get("/api/eval")
def eval_report():
    p = DATA / "eval_report.json"
    if not p.exists():
        raise HTTPException(404, "run `python -m priorart eval --save` first")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")
