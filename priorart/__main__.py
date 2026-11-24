"""CLI:  python -m priorart {gen,search,eval,tune,serve}"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="priorart", description="Prior-art finder prototype (synthetic corpus)")
    p.add_argument("--dense", action="store_true", help="use the optional neural embeddings (needs fastembed)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen", help="regenerate the synthetic corpus, sample disclosures and labels")
    s = sub.add_parser("search", help="search with a sample disclosure id or a text file")
    s.add_argument("source", help="sample id (D1..D16) or path to a text file")
    s.add_argument("--before", help="priority date YYYY-MM-DD (default: the sample's date)")
    s.add_argument("--class", dest="classes", action="append", default=[], help="class prefix filter, repeatable")
    s.add_argument("--mode", default="hybrid_rerank")
    s.add_argument("-k", type=int, default=10)
    e = sub.add_parser("eval", help="retrieval metrics (dev and test) and feature-mapping accuracy")
    e.add_argument("--split", choices=("dev", "test", "all"), default=None, help="default: dev and test tables")
    e.add_argument("--with-feedback", action="store_true", help="include relevance marks saved from the UI")
    e.add_argument("--json", action="store_true")
    e.add_argument("--no-thesaurus", action="store_true", help="ablation: disable the synonym thesaurus")
    e.add_argument("--untuned", action="store_true", help="use the untuned default settings")
    t = sub.add_parser("tune", help="tune retrieval + mapping on the dev split and write data/tuned.json")
    t.add_argument("--samples", type=int, default=250)
    sub.add_parser("report", help="write data/eval_report.json (tuned vs untuned, dev vs test, mapping) and print it")
    v = sub.add_parser("serve", help="run the web app")
    v.add_argument("--host", default="localhost")
    v.add_argument("--port", type=int, default=8765)
    a = p.parse_args(argv)
    if a.dense:
        os.environ["PRIORART_DENSE"] = "1"

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
        from .calibrate import format_mapping_table, mapping_eval
        from .evals import evaluate, format_table
        from .mapping import MapCfg
        from .retrieval import Index, Params, get_index, load_corpus
        if a.no_thesaurus:
            _text.USE_THESAURUS = False
        dense = a.dense
        ix = Index(load_corpus(), Params(dense=dense)) if a.untuned else get_index(dense=dense)
        splits = [a.split] if a.split else ["dev", "test"]
        out = {}
        for sp in splits:
            res = evaluate(ix, with_feedback=a.with_feedback, split=None if sp == "all" else sp, modes=ix.modes())
            out[sp] = res
            if not a.json:
                print(f"\n### {sp} split\n\n" + format_table(res))
        if not a.json and not a.with_feedback and not a.no_thesaurus:
            cfg = MapCfg() if a.untuned else MapCfg.tuned(dense)
            rows = {}
            for sp in [s for s in splits if s != "all"]:
                for name, c in (("legacy overlap", MapCfg()), ("calibrated", cfg)):
                    rows[f"{name} / {sp}"] = {**mapping_eval(ix, c, sp), "split": sp}
            print("\n### Feature mapping accuracy (is the feature disclosed in the document?)\n")
            print(format_mapping_table({k.split(" / ")[0]: v for k, v in rows.items()}))
        if a.json:
            print(json.dumps(out, indent=2))
    elif a.cmd == "tune":
        from .tune import tune
        tune(dense=a.dense, samples=a.samples)
        print("wrote data/tuned.json")
    elif a.cmd == "report":
        from .report import markdown, save
        print(markdown(save(dense=a.dense)))
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("priorart.app:app", host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
