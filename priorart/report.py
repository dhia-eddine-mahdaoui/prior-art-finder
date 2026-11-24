"""Builds data/eval_report.json: tuned vs untuned retrieval on dev and test, mapping accuracy, per-query test results.
Rendered by the web UI's Evaluation page and quoted in the README."""
from __future__ import annotations

import json

from .calibrate import mapping_eval
from .evals import evaluate
from .mapping import MapCfg
from .retrieval import DATA, Index, Params, get_index, load_corpus

LABELS = {"bm25": "BM25 only", "lsa": "LSA only", "dense": "Neural embeddings only (optional)",
          "hybrid": "Hybrid (BM25 + LSA + per-feature, RRF)", "hybrid_rerank": "Hybrid + feature-coverage rerank"}


def build(dense: bool = False) -> dict:
    docs = load_corpus()
    tuned = get_index(docs, dense=dense)
    base = Index(docs, Params(dense=dense))
    rep = {"corpus_documents": len(docs), "dense": dense, "labels": LABELS, "settings": tuned.params.to_dict(),
           "retrieval": {}, "mapping": {}, "per_query_test": {}}
    for name, ix in (("untuned", base), ("tuned", tuned)):
        rep["retrieval"][name] = {}
        for sp in ("dev", "test"):
            r = evaluate(ix, split=sp, modes=ix.modes())
            rep["retrieval"][name][sp] = {m: v["mean"] for m, v in r.items()}
            rep["retrieval"][name][sp]["n_queries"] = next(iter(r.values()))["n_queries"]
            if name == "tuned" and sp == "test":
                rep["per_query_test"] = {m: v["per_query"] for m, v in r.items()}
    cfg = MapCfg.tuned(dense)
    for label, c, ix in (("overlap formula (untuned)", MapCfg(), base), ("calibrated", cfg, tuned)):
        rep["mapping"][label] = {sp: mapping_eval(ix, c, sp) for sp in ("dev", "test")}
    rep["mapping_config"] = cfg.to_dict()
    return rep


def save(dense: bool = False) -> dict:
    rep = build(dense)
    name = "eval_report_dense.json" if dense else "eval_report.json"
    (DATA / name).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    return rep


def markdown(rep: dict) -> str:
    L = []
    for sp in ("dev", "test"):
        for name in ("untuned", "tuned"):
            t = rep["retrieval"][name][sp]
            L += [f"\n**{sp} split, {name} settings** ({t['n_queries']} queries)\n",
                  "| Retrieval | recall@10 | nDCG@10 | MAP | MRR |", "|---|---|---|---|---|"]
            for m, lab in rep["labels"].items():
                if m in t:
                    v = t[m]
                    L.append(f"| {lab} | {v['recall@10']:.3f} | {v['ndcg@10']:.3f} | {v['map']:.3f} | {v['mrr']:.3f} |")
    L += ["\n**Feature mapping**\n", "| Mapping | split | covered precision | covered recall | detect F1 | false-covered | avg precision |",
          "|---|---|---|---|---|---|---|"]
    for name, d in rep["mapping"].items():
        for sp, m in d.items():
            L.append(f"| {name} | {sp} | {m['covered_precision']:.3f} | {m['covered_recall']:.3f} | {m['detect_f1']:.3f} | "
                     f"{m['false_covered_rate']:.3f} | {m['avg_precision']:.3f} |")
    return "\n".join(L)
