"""Hybrid retrieval over the corpus: BM25 + LSA (TF-IDF -> truncated SVD), RRF fusion,
feature-coverage rerank, class/date filters, and feature-to-passage mapping.

Runs fully offline on CPU. The "dense" side is LSA rather than a neural embedding
model, because it needs no model download.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .text import analyze, analyze_with_bigrams, split_sentences, highlight_segments

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / ".cache"
INDEX_VERSION = "1"

MODES = ("bm25", "lsa", "hybrid", "hybrid_rerank")
COVERED, PARTIAL = 0.55, 0.28


def load_corpus(path: Path | None = None) -> list[dict]:
    path = path or DATA / "corpus.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def doc_text(d: dict) -> str:
    return " ".join([d["title"], d["title"], d["abstract"], *d["claims"]])


def passages_of(d: dict) -> list[tuple[str, str]]:
    ps = [("title", d["title"])]
    ps += [("abstract", s) for s in split_sentences(d["abstract"])]
    ps += [(f"claim {c.split('.', 1)[0]}", c.split(".", 1)[1].strip()) for c in d["claims"]]
    return ps


def corpus_hash(docs: list[dict]) -> str:
    from . import text as _t
    h = hashlib.sha256((INDEX_VERSION + str(_t.USE_THESAURUS)).encode())
    for d in docs:
        h.update(json.dumps(d, sort_keys=True).encode())
    return h.hexdigest()[:16]


class Index:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75, dims: int = 96):
        self.docs = docs
        self.k1, self.b = k1, b
        self.by_id = {d["id"]: i for i, d in enumerate(docs)}
        # BM25
        toks = [analyze(doc_text(d)) for d in docs]
        self.tf = [Counter(t) for t in toks]
        self.dl = np.array([len(t) for t in toks], dtype=float)
        self.avgdl = self.dl.mean()
        df = Counter(w for c in self.tf for w in c)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.max_idf = max(self.idf.values())
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, c in enumerate(self.tf):
            for w, f in c.items():
                self.postings.setdefault(w, []).append((i, f))
        # LSA
        self.vec = TfidfVectorizer(analyzer=analyze_with_bigrams, sublinear_tf=True, min_df=1)
        X = self.vec.fit_transform([doc_text(d) for d in docs])
        self.svd = TruncatedSVD(n_components=min(dims, X.shape[0] - 1, X.shape[1] - 1), random_state=0)
        self.doc_vecs = normalize(self.svd.fit_transform(X))
        # passages for mapping
        self.passages: list[list[dict]] = []
        for d in docs:
            ps = passages_of(d)
            V = normalize(self.svd.transform(self.vec.transform([t for _, t in ps])))
            self.passages.append([{"kind": k, "text": t, "vec": V[j], "stems": set(analyze(t))} for j, (k, t) in enumerate(ps)])

    # ---- scoring -------------------------------------------------------
    def embed(self, text: str) -> np.ndarray:
        return normalize(self.svd.transform(self.vec.transform([text])))[0]

    def bm25(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.docs))
        for w, qf in Counter(analyze(query)).items():
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, f in self.postings[w]:
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return s

    def lsa(self, query: str) -> np.ndarray:
        return self.doc_vecs @ self.embed(query)

    def mask(self, classes: list[str] | None = None, before: str | None = None) -> np.ndarray:
        """Boolean candidate mask. `before` = priority date: only docs filed strictly earlier qualify."""
        m = np.ones(len(self.docs), dtype=bool)
        pref = [c.replace(" ", "").upper() for c in (classes or []) if c.strip()]
        for i, d in enumerate(self.docs):
            if before and not d["filing_date"] < before:
                m[i] = False
            if pref and not any(c.replace(" ", "").upper().startswith(p) for c in d["cpc"] for p in pref):
                m[i] = False
        return m

    @staticmethod
    def _rank(scores: np.ndarray, mask: np.ndarray, limit: int) -> list[int]:
        idx = [i for i in np.argsort(-scores, kind="stable") if mask[i]]
        return idx[:limit]

    # ---- search --------------------------------------------------------
    def search(self, query: str, mode: str = "hybrid_rerank", mask: np.ndarray | None = None,
               k: int = 10, features: list | None = None, depth: int = 40) -> list[dict]:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode}")
        mask = np.ones(len(self.docs), dtype=bool) if mask is None else mask
        b, l = self.bm25(query), self.lsa(query)
        if mode == "bm25":
            return [{"idx": i, "score": float(b[i])} for i in self._rank(b, mask, k)]
        if mode == "lsa":
            return [{"idx": i, "score": float(l[i])} for i in self._rank(l, mask, k)]
        fused: dict[int, float] = {}
        for sc in (b, l):
            for r, i in enumerate(self._rank(sc, mask, 100)):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + r + 1)
        cand = sorted(fused, key=lambda i: -fused[i])
        if mode == "hybrid":
            return [{"idx": i, "score": fused[i]} for i in cand[:k]]
        # rerank the top `depth` by how well each document covers the disclosure's features
        cand = cand[:depth]
        top = max(fused[i] for i in cand) or 1.0
        out = []
        for i in cand:
            cov = coverage(self.map_features(i, features or []))
            out.append({"idx": i, "score": 0.65 * cov + 0.35 * fused[i] / top, "coverage": cov, "fused": fused[i]})
        out.sort(key=lambda r: -r["score"])
        return out[:k]

    # ---- feature mapping ----------------------------------------------
    def map_features(self, doc_idx: int, features: list) -> list[dict]:
        rows = []
        for f in features:
            fvec = self.embed(f.text)
            fst = set(f.stems)
            wsum = sum(self.idf.get(w, self.max_idf) for w in fst) or 1.0
            best = None
            for p in self.passages[doc_idx]:
                lex = sum(self.idf.get(w, self.max_idf) for w in fst & p["stems"]) / wsum
                sem = max(0.0, float(fvec @ p["vec"]))
                s = max(lex, 0.6 * lex + 0.4 * sem)
                if best is None or s > best[0]:
                    best = (s, lex, sem, p)
            s, lex, sem, p = best
            status = "covered" if s >= COVERED else "partial" if s >= PARTIAL else "not found"
            rows.append({"feature_id": f.id, "feature": f.text, "status": status, "score": round(s, 3),
                         "lexical": round(lex, 3), "semantic": round(sem, 3), "kind": p["kind"],
                         "passage": p["text"] if status != "not found" else "",
                         "segments": highlight_segments(p["text"], fst) if status != "not found" else []})
        return rows


def coverage(mapping: list[dict]) -> float:
    return sum(r["score"] for r in mapping) / len(mapping) if mapping else 0.0


def get_index(docs: list[dict] | None = None, use_cache: bool = True) -> Index:
    """Content-hash keyed pickle cache: the index is rebuilt only when the corpus changes."""
    docs = docs if docs is not None else load_corpus()
    path = CACHE / f"index-{corpus_hash(docs)}.pkl"
    if use_cache and path.exists():
        try:
            return pickle.loads(path.read_bytes())
        except Exception:
            pass
    idx = Index(docs)
    if use_cache:
        CACHE.mkdir(exist_ok=True)
        for old in CACHE.glob("index-*.pkl"):
            old.unlink()
        path.write_bytes(pickle.dumps(idx))
    return idx
