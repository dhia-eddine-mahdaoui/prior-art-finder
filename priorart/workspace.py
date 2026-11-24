"""Invention-disclosure projects: draft -> search -> shortlist -> claim chart.

A project keeps the disclosure draft (text, priority date, class filter, reviewer-edited feature list), the saved
searches that were run on it, and the shortlist with a decision and a note per reference. Stored as one JSON file
(var/projects.json); fine for a single reviewer, a database would replace it for a team.
"""
from __future__ import annotations

import csv
import io
import json
import threading
import time
import uuid

from . import pipeline

DECISIONS = ("relevant", "maybe", "not_relevant")
_LOCK = threading.Lock()


def _path():
    return pipeline.VAR / "projects.json"


def _load() -> dict:
    p = _path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save(data: dict) -> None:
    pipeline.VAR.mkdir(parents=True, exist_ok=True)
    tmp = _path().with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(_path())


def list_projects() -> list[dict]:
    with _LOCK:
        ps = sorted(_load().values(), key=lambda p: -p["updated"])
    return [{"id": p["id"], "name": p["name"], "updated": p["updated"], "priority_date": p["disclosure"]["priority_date"],
             "searches": len(p["searches"]), "shortlist": len(p["shortlist"]),
             "excerpt": " ".join(p["disclosure"]["text"].split())[:140]} for p in ps]


def create(name: str, text: str = "", priority_date: str | None = None, classes: list[str] | None = None) -> dict:
    now = int(time.time())
    p = {"id": uuid.uuid4().hex[:10], "name": name.strip() or "Untitled disclosure", "created": now, "updated": now,
         "disclosure": {"text": text, "priority_date": priority_date, "classes": classes or [], "features": None},
         "searches": [], "shortlist": []}
    with _LOCK:
        data = _load()
        data[p["id"]] = p
        _save(data)
    return p


def get(pid: str) -> dict | None:
    with _LOCK:
        return _load().get(pid)


def _mutate(pid: str, fn):
    with _LOCK:
        data = _load()
        p = data.get(pid)
        if p is None:
            return None
        fn(p)
        p["updated"] = int(time.time())
        _save(data)
        return p


def update(pid: str, name: str | None = None, disclosure: dict | None = None) -> dict | None:
    def fn(p):
        if name is not None:
            p["name"] = name.strip() or p["name"]
        if disclosure:
            for k in ("text", "priority_date", "classes", "features"):
                if k in disclosure:
                    p["disclosure"][k] = disclosure[k]
    return _mutate(pid, fn)


def delete(pid: str) -> bool:
    with _LOCK:
        data = _load()
        if pid not in data:
            return False
        del data[pid]
        _save(data)
        return True


def record_search(pid: str, res: dict, params: dict) -> dict | None:
    """Saved search: the settings and the ranked ids, so a reviewer can see how the picture changed."""
    def fn(p):
        n = len(p["searches"]) + 1
        p["searches"].append({"id": n, "ts": int(time.time()), "params": params, "candidates": res["candidates"],
                              "excluded_by_date": res["excluded_by_date"], "n_features": len(res["features"]),
                              "top": [{"id": r["id"], "coverage": r["coverage"]} for r in res["results"][:10]]})
        p["searches"] = p["searches"][-20:]
    return _mutate(pid, fn)


def set_decision(pid: str, doc_id: str, decision: str | None, note: str | None) -> dict | None:
    if decision is not None and decision not in DECISIONS:
        raise ValueError("decision must be one of " + ", ".join(DECISIONS))

    def fn(p):
        row = next((r for r in p["shortlist"] if r["doc_id"] == doc_id), None)
        if row is None:
            row = {"doc_id": doc_id, "decision": "maybe", "note": "", "added": int(time.time())}
            p["shortlist"].append(row)
        if decision:
            row["decision"] = decision
        if note is not None:
            row["note"] = note
    return _mutate(pid, fn)


def remove_from_shortlist(pid: str, doc_id: str) -> dict | None:
    return _mutate(pid, lambda p: p.__setitem__("shortlist", [r for r in p["shortlist"] if r["doc_id"] != doc_id]))


# ---- claim chart ---------------------------------------------------------------
def project_features(p: dict, extractor_features: list[dict] | None = None) -> list[str]:
    from .features import get_extractor
    if p["disclosure"].get("features"):
        return p["disclosure"]["features"]
    return [f.text for f in get_extractor().extract(p["disclosure"]["text"])]


def build_chart(p: dict, index, include_excluded: bool = False) -> dict:
    texts = project_features(p)
    feats = pipeline.features_from_texts(texts)
    rows = [r for r in p["shortlist"] if include_excluded or r["decision"] != "not_relevant"]
    refs = pipeline.chart_for(index, feats, [r["doc_id"] for r in rows])
    for ref, r in zip(refs, rows):
        ref["decision"], ref["note"] = r["decision"], r["note"]
    union = []
    for f in feats:
        best = "not found"
        for ref in refs:
            st = next(m["status"] for m in ref["mapping"] if m["feature_id"] == f.id)
            if st == "covered" or (st == "partial" and best == "not found"):
                best = st
        union.append({"feature_id": f.id, "feature": f.text, "status": best})
    return {"features": [f.to_dict() for f in feats], "references": refs, "combined": union}


def _cite(ref: dict, m: dict) -> str:
    return f"{ref['id']}, {m['kind']}: {m['passage']}" if m["passage"] else "-"


def chart_markdown(p: dict, chart: dict) -> str:
    d = p["disclosure"]
    L = [f"# Claim chart: {p['name']}", "",
         "_Prototype output over a synthetic corpus; statuses are an automated triage aid, not a novelty or "
         "patentability opinion. Not legal advice._", "",
         f"- Priority date: {d['priority_date'] or 'not set'}",
         f"- Class filter: {', '.join(d['classes']) if d['classes'] else 'none'}",
         f"- References in chart: {len(chart['references'])}", "", "## Combined view", "",
         "| Feature | Disclosed by the shortlisted references |", "|---|---|"]
    for u in chart["combined"]:
        L.append(f"| {u['feature_id']} {u['feature']} | {u['status']} |")
    refs = chart["references"]
    L += ["", "## Feature-by-reference chart", ""]
    if refs:
        L.append("| Feature | " + " | ".join(f"{r['id']} ({r['filing_date']})" for r in refs) + " |")
        L.append("|---|" + "---|" * len(refs))
        for f in chart["features"]:
            cells = []
            for r in refs:
                m = next(x for x in r["mapping"] if x["feature_id"] == f["id"])
                cells.append(f"**{m['status']}** ({m['score']:.2f}) {_cite(r, m)}".replace("|", "/"))
            L.append(f"| {f['id']} {f['text']} | " + " | ".join(cells) + " |")
    L += ["", "## References", ""]
    for r in refs:
        L.append(f"- **{r['id']}** {r['title']} - filed {r['filing_date']}, {r['assignee']} "
                 f"- reviewer: {r['decision'].replace('_', ' ')}" + (f" - note: {r['note']}" if r["note"] else ""))
    return "\n".join(L) + "\n"


def chart_csv(chart: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["feature_id", "feature", "reference_id", "reference_title", "filed", "reviewer_decision", "status",
                "score", "passage_location", "passage", "reviewer_note"])
    for r in chart["references"]:
        for m in r["mapping"]:
            w.writerow([m["feature_id"], m["feature"], r["id"], r["title"], r["filing_date"], r["decision"],
                        m["status"], f"{m['score']:.2f}", m["kind"] if m["passage"] else "", m["passage"], r["note"]])
    return buf.getvalue()
