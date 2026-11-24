"""Retrieval evaluation: recall@k, MRR and nDCG@k over graded relevance labels."""
from __future__ import annotations

import json
import math

from .features import get_extractor
from .pipeline import load_samples, read_feedback, query_id
from .retrieval import DATA, MODES, Index, get_index


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    relevant = {d for d, g in rel.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def mrr(ranked: list[str], rel: dict[str, int]) -> float:
    for i, d in enumerate(ranked, 1):
        if rel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    gain = lambda g: 2 ** g - 1
    dcg = sum(gain(rel.get(d, 0)) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(gain(g) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def average_precision(ranked: list[str], rel: dict[str, int]) -> float:
    relevant = {d for d, g in rel.items() if g > 0}
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def load_qrels(with_feedback: bool = False):
    """Returns (queries, qrels): queries = {id: {text, priority_date}}; qrels = {id: {doc_id: grade}}."""
    queries = {s["id"]: {"text": s["text"], "priority_date": s["priority_date"], "split": s.get("split")} for s in load_samples()}
    qrels = {k: dict(v) for k, v in json.loads((DATA / "qrels.json").read_text()).items()}
    if with_feedback:
        for r in read_feedback():
            qid = r["query_id"]
            if qid not in queries:
                queries[qid] = {"text": r["query_text"], "priority_date": None}
            g = qrels.setdefault(qid, {})
            if r["label"] == "relevant":
                g[r["doc_id"]] = max(g.get(r["doc_id"], 0), 2)
            elif r["label"] == "not_relevant":
                g[r["doc_id"]] = 0
            else:
                g.pop(r["doc_id"], None)
    return queries, qrels


def evaluate(index: Index | None = None, with_feedback: bool = False, k: int = 10, modes=None,
             split: str | None = None) -> dict:
    """Macro-averaged retrieval metrics. split = "dev" | "test" restricts to that part of the labelled queries."""
    index = index or get_index()
    modes = modes or index.modes()
    queries, qrels = load_qrels(with_feedback)
    ext = get_extractor()
    out = {m: {"per_query": {}} for m in modes}
    for qid, q in queries.items():
        if split and q.get("split") != split:
            continue
        rel = qrels.get(qid, {})
        if not any(g > 0 for g in rel.values()):
            continue
        feats = ext.extract(q["text"])
        mask = index.mask(None, q["priority_date"])
        for m in modes:
            hits = index.search(q["text"], mode=m, mask=mask, k=max(k, 50), features=feats)
            ranked = [index.docs[h["idx"]]["id"] for h in hits]
            out[m]["per_query"][qid] = {"recall@10": recall_at_k(ranked, rel, k), "mrr": mrr(ranked, rel),
                                         "ndcg@10": ndcg_at_k(ranked, rel, k),
                                         "map": average_precision(ranked, rel)}
    for m in modes:
        pq = out[m]["per_query"]
        out[m]["mean"] = {key: sum(v[key] for v in pq.values()) / max(1, len(pq)) for key in ("recall@10", "mrr", "ndcg@10", "map")}
        out[m]["n_queries"] = len(pq)
    return out


def format_table(res: dict) -> str:
    lines = ["| Retrieval | recall@10 | nDCG@10 | MAP | MRR |", "|---|---|---|---|---|"]
    labels = {"bm25": "BM25 only", "lsa": "LSA only", "dense": "Neural embeddings only (optional)", "hybrid": "Hybrid (BM25 + LSA, RRF)",
              "hybrid_rerank": "Hybrid + feature-coverage rerank"}
    for m, r in res.items():
        v = r["mean"]
        lines.append(f"| {labels.get(m, m)} | {v['recall@10']:.3f} | {v['ndcg@10']:.3f} | {v['map']:.3f} | {v['mrr']:.3f} |")
    n = next(iter(res.values()))["n_queries"]
    lines.append("")
    lines.append(f"Macro-average over {n} queries.")
    return "\n".join(lines)
