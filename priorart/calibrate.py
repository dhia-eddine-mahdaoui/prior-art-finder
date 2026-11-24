"""Feature-mapping accuracy metric and calibration.

Gold: the generator knows which key feature of each disclosure is really disclosed in each document
(data/feature_gold.json). The extracted features of a disclosure are aligned to those gold features, and every
(feature, document) pair is a binary example: is the feature disclosed in the document? Pairs cover all labelled
documents of the query (close art, partial art, decoys, late filings) plus random background documents.

calibrate() fits the logistic scoring model and the covered / partial thresholds on the dev split only, using
leave-one-query-out cross-validation to choose the soft-match similarity tau and the regularisation, so the dev
numbers it reports are out-of-fold.  mapping_eval() scores any MapCfg on any split.
"""
from __future__ import annotations

import json
import random
from dataclasses import replace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from .evals import load_qrels
from .features import HeuristicExtractor
from .mapping import Mapper, MapCfg
from .retrieval import DATA, Index
from .text import analyze

ALIGN_MIN = 0.6


def load_gold() -> dict:
    return json.loads((DATA / "feature_gold.json").read_text(encoding="utf-8"))


def align(features, gold_features: dict[str, str]) -> dict[str, object]:
    """Map gold feature key -> extracted Feature (best overlap with the gold wording, one-to-one)."""
    gstems = {k: set(analyze(t)) for k, t in gold_features.items()}
    cand = []
    for f in features:
        fs = set(f.stems)
        for k, g in gstems.items():
            ov = len(fs & g) / max(1, len(g))
            if ov >= ALIGN_MIN:
                cand.append((ov, k, f.id))
    used_k, used_f, out = set(), set(), {}
    by_id = {f.id: f for f in features}
    for ov, k, fid in sorted(cand, reverse=True):
        if k not in used_k and fid not in used_f:
            used_k.add(k); used_f.add(fid); out[k] = by_id[fid]
    return out


def gold_pairs(index: Index, split: str, n_bg: int = 12) -> list[dict]:
    queries, qrels = load_qrels()
    gold = load_gold()
    ex = HeuristicExtractor()
    pairs = []
    for qid, q in queries.items():
        if q.get("split") != split:
            continue
        al = align(ex.extract(q["text"]), gold[qid]["features"])
        labelled = list(qrels[qid])
        rng = random.Random(qid)
        others = [d["id"] for d in index.docs if d["id"] not in qrels[qid]]
        docs = labelled + rng.sample(others, n_bg)
        present = gold[qid]["docs"]
        for did in docs:
            for key, f in al.items():
                pairs.append({"qid": qid, "feature": f, "doc": index.by_id[did], "doc_id": did, "key": key,
                              "present": int(key in present.get(did, []))})
    return pairs


def _matrix(mapper: Mapper, pairs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """One signal row per pair (best passage by a fixed heuristic) and the labels."""
    X, y = [], []
    for p in pairs:
        S, _ = mapper.signals(p["doc"], p["feature"])
        X.append(S[int(np.argmax(S[:, 1] + 0.5 * S[:, 2] + 0.3 * S[:, 3] + 0.6 * S[:, 4]))])
        y.append(p["present"])
    return np.array(X), np.array(y)


def _prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def pick_thresholds(prob: np.ndarray, y: np.ndarray, covered_precision: float = 0.95) -> tuple[float, float]:
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1 = [_prf(y, (prob >= t).astype(int))[2] for t in grid]
    t_part = float(grid[int(np.argmax(f1))])
    t_cov = None
    for t in grid:
        if t <= t_part:
            continue
        sel = prob >= t
        if sel.sum() >= 5 and (y[sel].mean() >= covered_precision):
            t_cov = float(t)
            break
    return t_part, (t_cov if t_cov is not None else min(0.95, t_part + 0.25))


def calibrate(index: Index, taus=(0.4, 0.5, 0.6, 0.7, 0.8), cs=(0.3, 1.0, 3.0, 10.0)) -> dict:
    dev = gold_pairs(index, "dev")
    qids = sorted({p["qid"] for p in dev})
    qarr = np.array([p["qid"] for p in dev])
    best = None
    for tau in taus:
        mapper = Mapper(index, MapCfg(method="calibrated", tau=tau, weights=[0] * 5))
        X, y = _matrix(mapper, dev)
        for c in cs:
            oof = np.zeros(len(y))
            for q in qids:
                te = qarr == q
                m = LogisticRegression(C=c, max_iter=1000).fit(X[~te], y[~te])
                oof[te] = m.predict_proba(X[te])[:, 1]
            ap = average_precision_score(y, oof)
            if best is None or ap > best["ap"]:
                best = {"ap": ap, "tau": tau, "C": c, "oof": oof, "y": y, "X": X}
    m = LogisticRegression(C=best["C"], max_iter=1000).fit(best["X"], best["y"])
    t_part, t_cov = pick_thresholds(best["oof"], best["y"])
    cfg = MapCfg(method="calibrated", tau=best["tau"], weights=[round(float(w), 4) for w in m.coef_[0]],
                 bias=round(float(m.intercept_[0]), 4), covered=t_cov, partial=t_part)
    return {"cfg": cfg, "dev_oof_ap": best["ap"], "C": best["C"], "n_pairs": len(dev)}


def mapping_eval(index: Index, cfg: MapCfg, split: str) -> dict:
    mapper = Mapper(index, cfg)
    pairs = gold_pairs(index, split)
    y = np.array([p["present"] for p in pairs])
    rows = [mapper.map(p["doc"], [p["feature"]])[0] for p in pairs]
    status = np.array([r["status"] for r in rows])
    scores = np.array([r["score"] for r in rows])
    cov, det = (status == "covered").astype(int), (status != "not found").astype(int)
    cp, cr, cf = _prf(y, cov)
    dp, dr, df = _prf(y, det)
    absent = y == 0
    return {"pairs": len(pairs), "disclosed_share": float(y.mean()),
            "covered_precision": cp, "covered_recall": cr,
            "detect_precision": dp, "detect_recall": dr, "detect_f1": df,
            "false_covered_rate": float(cov[absent].mean()) if absent.any() else 0.0,
            "avg_precision": float(average_precision_score(y, scores)),
            "accuracy": float(((status == "covered") == (y == 1)).mean())}


def format_mapping_table(rows: dict[str, dict]) -> str:
    L = ["| Mapping | split | pairs | covered precision | covered recall | detect F1 (covered+partial) | false-covered rate | avg precision |",
         "|---|---|---|---|---|---|---|---|"]
    for name, m in rows.items():
        L.append(f"| {name} | {m['split']} | {m['pairs']} | {m['covered_precision']:.3f} | {m['covered_recall']:.3f} | "
                 f"{m['detect_f1']:.3f} | {m['false_covered_rate']:.3f} | {m['avg_precision']:.3f} |")
    return "\n".join(L)
