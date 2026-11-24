"""Feature -> passage mapping: for each disclosure feature, find the best passage of a document and say whether it is
covered, partially covered, or not found.

Four signals per (feature, passage) pair:
  lex     IDF-weighted share of feature terms found in the passage (stemmed, thesaurus-normalised, numbers normalised)
  soft    like lex, but a missing term also counts when the passage holds a term with a similar LSA term vector
          (learned paraphrase matching, no hand-written synonyms needed)
  bigram  share of adjacent term pairs (phrases) of the feature that occur in the passage
  sem     cosine similarity of the feature and the passage in LSA space (passage-level semantic similarity)
They are combined by a logistic regression fitted on the dev split (calibrate.py), so the score is the estimated
probability that the passage discloses the feature. Two thresholds, also chosen on dev, turn it into
covered / partial / not found. Without data/tuned.json the original hand-set formula is used ("legacy").
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields

import numpy as np

from .retrieval import TUNED, Index
from .text import analyze, highlight_segments

SIGNALS = ("lex", "soft", "bigram", "sem", "dense")


@dataclass
class MapCfg:
    method: str = "legacy"                 # "legacy" hand-set formula | "calibrated" logistic model
    tau: float = 0.6                       # LSA term similarity needed for a soft match
    weights: list = field(default_factory=lambda: [0.0] * 5)
    bias: float = 0.0
    covered: float = 0.55
    partial: float = 0.28

    @classmethod
    def tuned(cls, dense: bool = False) -> "MapCfg":
        if TUNED.exists():
            raw = json.loads(TUNED.read_text(encoding="utf-8")).get("dense" if dense else "classical", {}).get("mapping")
            if raw:
                names = {f.name for f in fields(cls)}
                return cls(**{k: v for k, v in raw.items() if k in names})
        return cls()

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class Mapper:
    def __init__(self, index: Index, cfg: MapCfg | None = None):
        self.ix = index
        self.cfg = cfg or MapCfg.tuned(index.dense_docs is not None)
        self._fc: dict[str, dict] = {}
        self._inv: dict[int, str] | None = None

    # ---- per-feature context (cached) ----------------------------------
    def _feature(self, f) -> dict:
        c = self._fc.get(f.text)
        if c is None:
            ix = self.ix
            terms = analyze(f.text)
            ids = sorted(set(terms))
            w = np.array([ix.idf.get(t, ix.max_idf) for t in ids])
            c = {"ids": ids, "set": set(ids), "w": w, "wsum": float(w.sum()) or 1.0,
                 "vidx": np.array([ix.vocab.get(t, -1) for t in ids], dtype=int),
                 "big": {f"{a}_{b}" for a, b in zip(terms, terms[1:])}, "vec": ix.embed(f.text),
                 "dvec": ix.dense_embed(f.text) if ix.dense_docs is not None else None}
            if len(self._fc) > 2000:
                self._fc.clear()
            self._fc[f.text] = c
        return c

    # ---- signals ------------------------------------------------------------
    def signals(self, doc_idx: int, f) -> tuple[np.ndarray, list[dict]]:
        """Signal matrix (n_passages x 4: lex, soft, bigram, sem) for one feature against one document."""
        ix, c, tau = self.ix, self._feature(f), self.cfg.tau
        passages = ix.passages[doc_idx]
        out = np.zeros((len(passages), len(SIGNALS)))
        for j, p in enumerate(passages):
            exact = np.array([t in p["stems"] for t in c["ids"]])
            lex = float((c["w"] * exact).sum() / c["wsum"])
            sw = exact.astype(float)
            un = np.where(~exact & (c["vidx"] >= 0))[0]
            if un.size and p["tidx"].size:
                S = ix.term_vecs[c["vidx"][un]] @ ix.term_vecs[p["tidx"]].T
                best = S.max(axis=1)
                ok = best >= tau
                sw[un[ok]] = best[ok]
            soft = float((c["w"] * sw).sum() / c["wsum"])
            big = len(c["big"] & p["bigrams"]) / max(1, len(c["big"]))
            sem = max(0.0, float(c["vec"] @ p["vec"]))
            dns = max(0.0, float(c["dvec"] @ p["dvec"])) if c["dvec"] is not None else 0.0
            out[j] = (lex, soft, big, sem, dns)
        return out, passages

    def score(self, X: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        if cfg.method == "calibrated":
            w = np.zeros(X.shape[1])
            w[: len(cfg.weights)] = cfg.weights
            z = X @ w + cfg.bias
            return 1.0 / (1.0 + np.exp(-z))
        return np.maximum(X[:, 0], 0.6 * X[:, 0] + 0.4 * X[:, 3])

    def _soft_terms(self, c: dict, p: dict) -> set[str]:
        """Passage terms that matched a feature term only through LSA similarity (for highlighting)."""
        ix = self.ix
        if self._inv is None:
            self._inv = {i: t for t, i in ix.vocab.items()}
        un = np.where(np.array([t not in p["stems"] for t in c["ids"]]) & (c["vidx"] >= 0))[0]
        if not un.size or not p["tidx"].size:
            return set()
        S = ix.term_vecs[c["vidx"][un]] @ ix.term_vecs[p["tidx"]].T
        hit = (S >= self.cfg.tau).any(axis=0)
        return {self._inv[i] for i in p["tidx"][hit]}

    # ---- mapping --------------------------------------------------------------
    def map(self, doc_idx: int, features: list) -> list[dict]:
        rows = []
        cfg = self.cfg
        for f in features:
            c = self._feature(f)
            X, passages = self.signals(doc_idx, f)
            s = self.score(X)
            j = int(s.argmax())
            p, sc = passages[j], float(s[j])
            status = "covered" if sc >= cfg.covered else "partial" if sc >= cfg.partial else "not found"
            segs = []
            if status != "not found":
                exact = c["set"] & p["stems"]
                soft = self._soft_terms(c, p) - exact
                segs = highlight_segments(p["text"], exact, soft)
            rows.append({"feature_id": f.id, "feature": f.text, "status": status, "score": round(sc, 3),
                         "lexical": round(float(X[j, 0]), 3), "soft": round(float(X[j, 1]), 3),
                         "phrase": round(float(X[j, 2]), 3), "semantic": round(float(X[j, 4] if self.ix.dense_docs is not None else X[j, 3]), 3),
                         "kind": p["kind"], "passage": p["text"] if status != "not found" else "",
                         "backend": "dense" if self.ix.dense_docs is not None else "classical",
                         "segments": segs})
        return rows
