"""Hybrid retrieval over the corpus: BM25 + LSA (TF-IDF -> truncated SVD) + per-feature BM25, fused with weighted
reciprocal rank fusion, optional pseudo-relevance feedback, feature-coverage rerank, class/date filters.

Runs fully offline on CPU. The "dense" side is LSA rather than a neural embedding model, because it needs no model
download. Fusion weights and BM25 settings are tuned on the dev split of the labelled queries (see tune.py) and
stored in data/tuned.json; without that file the untuned defaults below are used.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import re

from .text import analyze, analyze_with_bigrams, core_text, split_sentences

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / ".cache"
INDEX_VERSION = "2"
TUNED = DATA / "tuned.json"

MODES = ("bm25", "lsa", "dense", "hybrid", "hybrid_rerank")


@dataclass
class Params:
    """Retrieval settings. The defaults are the untuned baseline."""
    k1: float = 1.5
    b: float = 0.75
    dims: int = 96
    rrf_k: int = 60
    w_bm25: float = 1.0
    w_lsa: float = 1.0
    w_feat: float = 0.0          # weight of the per-feature BM25 list in the fusion
    prf_docs: int = 0            # pseudo-relevance feedback: top documents of a first fused pass
    prf_terms: int = 0           # expansion terms added to the BM25 query
    prf_beta: float = 0.0        # weight of expansion terms / centroid shift of the LSA query
    rerank_alpha: float = 0.65   # weight of feature coverage in the final rerank score
    depth: int = 40              # candidates reranked
    qtf_pow: float = 0.0         # query term weight = count ** qtf_pow (0 = binary query, 1 = raw counts)
    dense: bool = False          # use neural embeddings (optional, needs `fastembed` and a one-off model download)
    w_dense: float = 0.0         # weight of the dense list in the fusion

    @classmethod
    def tuned(cls, dense: bool | None = None) -> "Params":
        """Tuned settings for the classical (default) or the dense configuration; untuned defaults if none saved."""
        from . import dense as _d
        dense = _d.enabled_by_env() if dense is None else dense
        raw = {}
        if TUNED.exists():
            raw = json.loads(TUNED.read_text(encoding="utf-8")).get("dense" if dense else "classical", {}).get("retrieval", {})
        names = {f.name for f in fields(cls)}
        p = cls(**{k: v for k, v in raw.items() if k in names})
        p.dense = dense
        return p

    def to_dict(self) -> dict:
        return asdict(self)


def load_corpus(path: Path | None = None) -> list[dict]:
    path = path or DATA / "corpus.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def doc_text(d: dict) -> str:
    return " ".join([d["title"], d["title"], d["abstract"], *d["claims"]])


def passages_of(d: dict) -> list[tuple[str, str]]:
    """Passages as (kind, text): title, abstract sentences, and claim elements (claim 1 is split at its semicolons,
    like the rows of a claim chart)."""
    ps = [("title", d["title"])]
    ps += [("abstract", s) for s in split_sentences(d["abstract"])]
    for c in d["claims"]:
        num, body = c.split(".", 1)
        body = body.strip()
        if "comprising:" in body:
            parts = [x.strip(" .") for x in re.split(r";\s*(?:and\s+)?", body.split("comprising:", 1)[1])]
            parts = [p for p in parts if p]
            ps += [(f"claim {num}({chr(97 + i)})", p) for i, p in enumerate(parts)]
        else:
            ps.append((f"claim {num}", body))
    return ps


def corpus_hash(docs: list[dict], dims: int = 96) -> str:
    from . import text as _t
    h = hashlib.sha256((INDEX_VERSION + str(_t.USE_THESAURUS) + str(dims) + repr(_t.SYNONYM_GROUPS)).encode())
    for d in docs:
        h.update(json.dumps(d, sort_keys=True).encode())
    return h.hexdigest()[:16]


class Index:
    def __init__(self, docs: list[dict], params: Params | None = None):
        self.docs = docs
        self.params = params or Params.tuned()
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
        self.svd = TruncatedSVD(n_components=min(self.params.dims, X.shape[0] - 1, X.shape[1] - 1), random_state=0)
        self.doc_vecs = normalize(self.svd.fit_transform(X))
        # term vectors (rows of V^T, normalised) for soft term matching in the mapper
        self.term_vecs = normalize(self.svd.components_.T)
        self.vocab = self.vec.vocabulary_
        # passages for mapping
        self.passages: list[list[dict]] = []
        for d in docs:
            ps = passages_of(d)
            cores = [core_text(t) for _, t in ps]
            V = normalize(self.svd.transform(self.vec.transform(cores)))
            rows = []
            for j, (k, t) in enumerate(ps):
                terms = analyze(cores[j])
                rows.append({"kind": k, "text": t, "core": cores[j], "vec": V[j], "stems": set(terms),
                             "bigrams": {f"{a}_{b}" for a, b in zip(terms, terms[1:])},
                             "tidx": np.array(sorted({self.vocab[w] for w in terms if w in self.vocab}), dtype=int)})
            self.passages.append(rows)
        # optional neural embeddings of documents and passages
        self.dense_docs = None
        if self.params.dense:
            from .dense import get_encoder
            enc = get_encoder()
            self.dense_docs = enc.encode([" ".join([d["title"], d["abstract"], *d["claims"]]) for d in docs])
            flat = [p["core"] for rows in self.passages for p in rows]
            E = enc.encode(flat)
            k = 0
            for rows in self.passages:
                for p in rows:
                    p["dvec"] = E[k]
                    k += 1
        self.mapper = None  # built lazily from the current tuned config

    # ---- pickling: the mapper is rebuilt from the current tuned config ----
    def __getstate__(self):
        s = dict(self.__dict__)
        s["mapper"] = None
        return s

    def ensure_mapper(self):
        if self.mapper is None:
            from .mapping import Mapper
            self.mapper = Mapper(self)
        return self.mapper

    # ---- scoring -------------------------------------------------------
    def embed(self, text: str) -> np.ndarray:
        return normalize(self.svd.transform(self.vec.transform([text])))[0]

    def bm25_weighted(self, qw: dict[str, float]) -> np.ndarray:
        k1, b = self.params.k1, self.params.b
        s = np.zeros(len(self.docs))
        for w, qf in qw.items():
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, f in self.postings[w]:
                s[i] += qf * idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self.dl[i] / self.avgdl))
        return s

    def bm25(self, query: str) -> np.ndarray:
        return self.bm25_weighted(self._qw(query))

    def _qw(self, query: str) -> dict[str, float]:
        return {w: float(c) ** self.params.qtf_pow for w, c in Counter(analyze(query)).items()}

    def lsa(self, query: str) -> np.ndarray:
        return self.doc_vecs @ self.embed(query)

    def dense_embed(self, text: str) -> np.ndarray:
        from .dense import get_encoder
        return get_encoder().encode([text])[0]

    def dense(self, query: str) -> np.ndarray:
        return self.dense_docs @ self.dense_embed(query)

    def modes(self) -> list[str]:
        return [m for m in MODES if m != "dense" or self.dense_docs is not None]

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

    def feature_scores(self, features: list) -> np.ndarray:
        """Mean over features of BM25(feature) / max BM25(feature): how many features each document touches."""
        if not features:
            return np.zeros(len(self.docs))
        acc = np.zeros(len(self.docs))
        for f in features:
            s = self.bm25(f.text)
            m = s.max()
            if m > 0:
                acc += s / m
        return acc / len(features)

    def _fuse(self, lists: list[tuple[float, list[int]]]) -> dict[int, float]:
        fused: dict[int, float] = {}
        for w, ranking in lists:
            if w <= 0:
                continue
            for r, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + w / (self.params.rrf_k + r + 1)
        return fused

    def _first_pass(self, mask, b, l, fs, dn=None):
        p = self.params
        lists = [(p.w_bm25, self._rank(b, mask, 100)), (p.w_lsa, self._rank(l, mask, 100))]
        if p.w_dense > 0 and dn is not None:
            lists.append((p.w_dense, self._rank(dn, mask, 100)))
        if p.w_feat > 0 and fs is not None:
            lists.append((p.w_feat, self._rank(fs, mask, 100)))
        return self._fuse(lists)

    def _expand(self, query: str, top: list[int]) -> tuple[dict[str, float], np.ndarray]:
        """Pseudo-relevance feedback: add the highest tf-idf terms of the top documents to the BM25 query and
        shift the LSA query towards their centroid."""
        p = self.params
        qw = self._qw(query)
        qvec = self.embed(query)
        if p.prf_docs and p.prf_beta > 0 and top:
            top = top[: p.prf_docs]
            if p.prf_terms:
                sc: Counter = Counter()
                for i in top:
                    for w, f in self.tf[i].items():
                        sc[w] += (1 + math.log(f)) * self.idf[w]
                cand = [(w, s) for w, s in sc.most_common(p.prf_terms * 3) if w not in qw][: p.prf_terms]
                mx = cand[0][1] if cand else 1.0
                for w, s in cand:
                    qw[w] = p.prf_beta * s / mx
            qvec = normalize((qvec + p.prf_beta * self.doc_vecs[top].mean(axis=0))[None, :])[0]
        return qw, qvec

    # ---- search --------------------------------------------------------
    def search(self, query: str, mode: str = "hybrid_rerank", mask: np.ndarray | None = None,
               k: int = 10, features: list | None = None, depth: int | None = None) -> list[dict]:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode}")
        p = self.params
        mask = np.ones(len(self.docs), dtype=bool) if mask is None else mask
        if mode == "bm25":
            b = self.bm25(query)
            return [{"idx": i, "score": float(b[i])} for i in self._rank(b, mask, k)]
        if mode == "lsa":
            l = self.lsa(query)
            return [{"idx": i, "score": float(l[i])} for i in self._rank(l, mask, k)]
        dn = self.dense(query) if self.dense_docs is not None else None
        if mode == "dense":
            if dn is None:
                raise ValueError("dense mode needs the optional neural embeddings (PRIORART_DENSE=1)")
            return [{"idx": i, "score": float(dn[i])} for i in self._rank(dn, mask, k)]
        b, l = self.bm25(query), self.lsa(query)
        fs = self.feature_scores(features) if (p.w_feat > 0 and features) else None
        fused = self._first_pass(mask, b, l, fs, dn)
        if p.prf_docs and p.prf_beta > 0:
            top = sorted(fused, key=lambda i: -fused[i])[: p.prf_docs]
            qw, qv = self._expand(query, top)
            b, l = self.bm25_weighted(qw), self.doc_vecs @ qv
            fused = self._first_pass(mask, b, l, fs, dn)
        cand = sorted(fused, key=lambda i: (-fused[i], i))
        if mode == "hybrid":
            return [{"idx": i, "score": fused[i]} for i in cand[:k]]
        # rerank the top candidates by how well each document covers the disclosure's features
        cand = cand[: (depth or p.depth)]
        top = max(fused[i] for i in cand) or 1.0
        mapper = self.ensure_mapper()
        out = []
        for i in cand:
            cov = coverage(mapper.map(i, features or []))
            out.append({"idx": i, "score": p.rerank_alpha * cov + (1 - p.rerank_alpha) * fused[i] / top,
                        "coverage": cov, "fused": fused[i]})
        out.sort(key=lambda r: -r["score"])
        return out[:k]

    # ---- feature mapping (see mapping.py) -------------------------------
    def map_features(self, doc_idx: int, features: list) -> list[dict]:
        return self.ensure_mapper().map(doc_idx, features)


def coverage(mapping: list[dict]) -> float:
    """Expected share of the disclosure's features that the document discloses (mean calibrated score)."""
    return sum(r["score"] for r in mapping) / len(mapping) if mapping else 0.0


def get_index(docs: list[dict] | None = None, use_cache: bool = True, params: Params | None = None,
              dense: bool | None = None) -> Index:
    """Content-hash keyed pickle cache: the index is rebuilt only when the corpus or index settings change."""
    docs = docs if docs is not None else load_corpus()
    params = params or Params.tuned(dense)
    path = CACHE / f"index-{corpus_hash(docs, params.dims)}{'-dense' if params.dense else ''}.pkl"
    idx = None
    if use_cache and path.exists():
        try:
            idx = pickle.loads(path.read_bytes())
        except Exception:
            idx = None
    if idx is None:
        idx = Index(docs, params)
        if use_cache:
            CACHE.mkdir(exist_ok=True)
            for old in CACHE.glob("index-*-dense.pkl" if params.dense else "index-*[0-9a-f].pkl"):
                old.unlink()
            path.write_bytes(pickle.dumps(idx))
    idx.params = params          # query-time settings always come from the current config
    idx.mapper = None
    return idx
