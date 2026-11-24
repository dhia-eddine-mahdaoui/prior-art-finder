"""FastAPI app: JSON API plus a single-page frontend."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import pipeline
from .retrieval import MODES, get_index

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


class FeedbackReq(BaseModel):
    text: str = Field(min_length=20)
    doc_id: str
    label: str


@app.get("/api/samples")
def samples():
    return pipeline.load_samples()


@app.get("/api/meta")
def meta():
    ix = index()
    classes = Counter(c.split()[0] for d in ix.docs for c in set(x.split()[0] for x in d["cpc"]))
    dates = sorted(d["filing_date"] for d in ix.docs)
    return {"documents": len(ix.docs), "classes": sorted(classes.items()), "modes": list(MODES),
            "date_range": [dates[0], dates[-1]]}


@app.post("/api/search")
def search(req: SearchReq):
    if req.mode not in MODES:
        raise HTTPException(400, "unknown mode")
    return pipeline.run_search(req.text, index(), req.priority_date, req.classes, req.top_k, req.mode)


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
    res = pipeline.run_search(req.text, index(), req.priority_date, req.classes, req.top_k, req.mode)
    return {"markdown": pipeline.markdown_report(req.text, res, req.priority_date, req.classes)}


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")
