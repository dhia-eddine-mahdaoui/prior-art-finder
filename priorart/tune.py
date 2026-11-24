"""Tune retrieval and mapping settings on the DEV split only and write data/tuned.json.

The test split is never read here. `python -m priorart tune [--dense]` then `python -m priorart eval` reports both.
Search: BM25 settings (k1, b, query-term weighting) first, then a seeded random search over the fusion settings
(LSA / per-feature / dense weights, RRF constant, pseudo-relevance feedback) and the LSA size, then the rerank weight.
Objective on dev: recall@10 + nDCG@10 + MAP of the hybrid list.
"""
from __future__ import annotations

import itertools
import json
import random
from dataclasses import replace

from .calibrate import calibrate
from .evals import average_precision, evaluate, load_qrels, ndcg_at_k, recall_at_k
from .features import HeuristicExtractor
from .mapping import MapCfg
from .retrieval import TUNED, Index, Params, load_corpus


def _score(ix: Index, params: Params, mode: str, prep) -> float:
    ix.params = params
    tot = 0.0
    for q, rel, feats, mask in prep:
        hits = ix.search(q["text"], mode=mode, mask=mask, k=50, features=feats)
        ranked = [ix.docs[h["idx"]]["id"] for h in hits]
        tot += recall_at_k(ranked, rel) + ndcg_at_k(ranked, rel) + average_precision(ranked, rel)
    return tot / len(prep)


def tune(dense: bool = False, samples: int = 250, seed: int = 0, log=print) -> dict:
    docs = load_corpus()
    queries, qrels = load_qrels()
    ex = HeuristicExtractor()
    dev = [(q, qrels[i]) for i, q in queries.items() if q.get("split") == "dev"]
    base = Params(dense=dense)
    ix = Index(docs, base)
    prep = [(q, rel, ex.extract(q["text"]), ix.mask(None, q["priority_date"])) for q, rel in dev]

    # 1. mapping first: the rerank uses it
    log("calibrating feature mapping on dev ...")
    cal = calibrate(ix)
    cfg: MapCfg = cal["cfg"]
    log(f"  tau={cfg.tau} covered>={cfg.covered} partial>={cfg.partial} (dev out-of-fold AP {cal['dev_oof_ap']:.3f})")

    # 2. BM25 settings (shared by the hybrid so the comparison stays fair)
    best = (-1.0, base)
    for k1, b, qp in itertools.product((0.8, 1.2, 1.5, 2.0, 3.0), (0.3, 0.5, 0.75, 0.9), (0.0, 0.5, 1.0)):
        p = replace(base, k1=k1, b=b, qtf_pow=qp)
        s = _score(ix, p, "bm25", prep)
        if s > best[0] + 1e-9:
            best = (s, p)
    bm = best[1]
    log(f"bm25: k1={bm.k1} b={bm.b} qtf_pow={bm.qtf_pow} dev objective {best[0]:.3f}")

    # 3. fusion settings
    rng = random.Random(seed)
    space = dict(rrf_k=(10, 20, 40, 60), w_lsa=(0.0, 0.3, 0.6, 1.0, 1.5, 2.0), w_feat=(0.0, 0.5, 1.0, 2.0),
                 prf_docs=(0, 3, 5, 8), prf_terms=(0, 10, 25), prf_beta=(0.3, 0.6, 1.0))
    if dense:
        space["w_dense"] = (0.0, 0.3, 0.6, 1.0, 1.5)
    cands = [bm]
    for _ in range(samples):
        cands.append(replace(bm, **{k: rng.choice(v) for k, v in space.items()}))
    scored = sorted(((_score(ix, p, "hybrid", prep), -i, p) for i, p in enumerate(cands)), key=lambda t: (t[0], t[1]), reverse=True)
    hy = scored[0][2]
    log(f"hybrid: {hy.to_dict()} dev objective {scored[0][0]:.3f}")

    # 4. LSA size: rebuild the index for each size
    best_d = (scored[0][0], hy, ix)
    for dims in (64, 128, 192):
        ix2 = Index(docs, replace(hy, dims=dims))
        s = _score(ix2, replace(hy, dims=dims), "hybrid", prep)
        if s > best_d[0] + 0.005:          # require a clear gain before paying for a bigger index
            best_d = (s, replace(hy, dims=dims), ix2)
    hy, ix = best_d[1], best_d[2]
    log(f"lsa dims={hy.dims}")

    # 5. rerank weight with the calibrated mapping
    ix.params = hy
    ix.mapper = None
    from .mapping import Mapper
    ix.mapper = Mapper(ix, cfg)
    best_a = (-1.0, hy)
    for alpha, depth in itertools.product((0.3, 0.5, 0.65, 0.8, 0.9), (20, 40)):
        p = replace(hy, rerank_alpha=alpha, depth=depth)
        s = _score(ix, p, "hybrid_rerank", prep)
        if s > best_a[0] + 1e-9:
            best_a = (s, p)
    final = best_a[1]
    log(f"rerank alpha={final.rerank_alpha} depth={final.depth} dev objective {best_a[0]:.3f}")

    out = json.loads(TUNED.read_text(encoding="utf-8")) if TUNED.exists() else {}
    out["dev_only"] = "tuned on the dev split; the test split was not read"
    out["dense" if dense else "classical"] = {"retrieval": final.to_dict(), "mapping": cfg.to_dict()}
    TUNED.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
