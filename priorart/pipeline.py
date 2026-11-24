"""End-to-end search: disclosure text -> features -> retrieval -> mapped results; plus feedback + report."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .features import get_extractor
from .retrieval import DATA, ROOT, Index, coverage, get_index

VAR = Path(os.environ.get("PRIORART_VAR", ROOT / "var"))


def query_id(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).encode()).hexdigest()[:12]


def load_samples() -> list[dict]:
    return json.loads((DATA / "disclosures.json").read_text(encoding="utf-8"))


def sample_id_for(text: str) -> str | None:
    norm = " ".join(text.split())
    for s in load_samples():
        if " ".join(s["text"].split()) == norm:
            return s["id"]
    return None


# ---- feedback (relevance judgements saved locally; extend the eval set) ----
def feedback_path() -> Path:
    return VAR / "feedback.jsonl"


def read_feedback() -> list[dict]:
    p = feedback_path()
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def save_feedback(text: str, doc_id: str, label: str) -> dict:
    if label not in ("relevant", "not_relevant", "clear"):
        raise ValueError("label must be relevant, not_relevant or clear")
    rec = {"query_id": sample_id_for(text) or query_id(text), "query_text": text, "doc_id": doc_id,
           "label": label, "ts": int(time.time())}
    VAR.mkdir(parents=True, exist_ok=True)
    with open(feedback_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def current_feedback(text: str) -> dict[str, str]:
    qid = sample_id_for(text) or query_id(text)
    out: dict[str, str] = {}
    for r in read_feedback():
        if r["query_id"] == qid:
            if r["label"] == "clear":
                out.pop(r["doc_id"], None)
            else:
                out[r["doc_id"]] = r["label"]
    return out


# ---- search ----
def run_search(text: str, index: Index | None = None, priority_date: str | None = None,
               classes: list[str] | None = None, top_k: int = 10, mode: str = "hybrid_rerank") -> dict:
    index = index or get_index()
    feats = get_extractor().extract(text)
    mask = index.mask(classes, priority_date)
    hits = index.search(text, mode=mode, mask=mask, k=top_k, features=feats)
    fb = current_feedback(text)
    results = []
    for rank, h in enumerate(hits, 1):
        d = index.docs[h["idx"]]
        mapping = index.map_features(h["idx"], feats)
        results.append({"rank": rank, "score": round(h["score"], 4), "coverage": round(coverage(mapping), 3),
                        "id": d["id"], "title": d["title"], "assignee": d["assignee"], "filing_date": d["filing_date"],
                        "cpc": d["cpc"], "abstract": d["abstract"], "mapping": mapping,
                        "summary": {s: sum(1 for m in mapping if m["status"] == s) for s in ("covered", "partial", "not found")},
                        "feedback": fb.get(d["id"])})
    return {"query_id": sample_id_for(text) or query_id(text), "mode": mode, "extractor": get_extractor().name,
            "features": [f.to_dict() for f in feats], "candidates": int(mask.sum()), "results": results}


def markdown_report(text: str, res: dict, priority_date: str | None, classes: list[str] | None) -> str:
    L = ["# Prior-art search report", "",
         "_Prototype output over a synthetic corpus. Not legal advice._", "",
         f"- Priority date filter: {priority_date or 'none'}",
         f"- Class filter: {', '.join(classes) if classes else 'none'}",
         f"- Retrieval mode: {res['mode']} (feature extractor: {res['extractor']})",
         f"- Candidate documents after filters: {res['candidates']}", "",
         "## Disclosure", "", "> " + " ".join(text.split()), "", "## Extracted features", ""]
    L += [f"- **{f['id']}** {f['text']}" for f in res["features"]]
    L += ["", "## Closest prior art", "", "| # | Document | Filed | Assignee | Coverage | Covered / partial / missing | Reviewer |",
          "|---|---|---|---|---|---|---|"]
    for r in res["results"]:
        s = r["summary"]
        L.append(f"| {r['rank']} | {r['id']} {r['title']} | {r['filing_date']} | {r['assignee']} | {r['coverage']:.2f} | "
                 f"{s['covered']} / {s['partial']} / {s['not found']} | {(r['feedback'] or '-').replace('_', ' ')} |")
    for r in res["results"][:5]:
        L += ["", f"### {r['rank']}. {r['id']} - {r['title']}", "", "| Feature | Status | Matching passage |", "|---|---|---|"]
        for m in r["mapping"]:
            passage = (m["passage"] or "-").replace("|", "/")
            L.append(f"| {m['feature_id']} {m['feature']} | {m['status']} | {m['kind']}: {passage} |")
    return "\n".join(L) + "\n"
