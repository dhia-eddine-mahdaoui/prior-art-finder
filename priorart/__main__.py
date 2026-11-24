"""CLI:  python -m priorart {gen,search,eval,serve}"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="priorart", description="Prior-art finder prototype (synthetic corpus)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen", help="regenerate the synthetic corpus, sample disclosures and labels")
    s = sub.add_parser("search", help="search with a sample disclosure id or a text file")
    s.add_argument("source", help="sample id (D1..D4) or path to a text file")
    s.add_argument("--before", help="priority date YYYY-MM-DD (default: the sample's date)")
    s.add_argument("--class", dest="classes", action="append", default=[], help="class prefix filter, repeatable")
    s.add_argument("--mode", default="hybrid_rerank")
    s.add_argument("-k", type=int, default=10)
    e = sub.add_parser("eval", help="print recall@10 / MRR / nDCG@10 for each retrieval mode")
    e.add_argument("--with-feedback", action="store_true", help="include relevance marks saved from the UI")
    e.add_argument("--json", action="store_true")
    e.add_argument("--no-thesaurus", action="store_true", help="ablation: disable the synonym thesaurus")
    v = sub.add_parser("serve", help="run the web app")
    v.add_argument("--host", default="localhost")
    v.add_argument("--port", type=int, default=8765)
    a = p.parse_args(argv)

    if a.cmd == "gen":
        from .corpus_gen import write_all
        print(write_all(), "documents written to data/")
    elif a.cmd == "search":
        from .pipeline import load_samples, run_search
        samples = {x["id"]: x for x in load_samples()}
        if a.source in samples:
            text, before = samples[a.source]["text"], a.before or samples[a.source]["priority_date"]
        else:
            text, before = open(a.source, encoding="utf-8").read(), a.before
        res = run_search(text, priority_date=before, classes=a.classes, top_k=a.k, mode=a.mode)
        print("Features:")
        for f in res["features"]:
            print(f"  {f['id']}: {f['text']}")
        print(f"\nTop {len(res['results'])} of {res['candidates']} candidates (mode={res['mode']})")
        for r in res["results"]:
            s = r["summary"]
            print(f"{r['rank']:>2}. {r['id']}  {r['filing_date']}  cov={r['coverage']:.2f}  "
                  f"[{s['covered']}C/{s['partial']}P/{s['not found']}N]  {r['title']}")
    elif a.cmd == "eval":
        from . import text as _text
        from .evals import evaluate, format_table
        if a.no_thesaurus:
            _text.USE_THESAURUS = False
        res = evaluate(with_feedback=a.with_feedback)
        print(json.dumps(res, indent=2) if a.json else format_table(res))
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("priorart.app:app", host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
