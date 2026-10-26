"""Ask one question about one document and pretty-print the grounded answer. Logic: docscout.pipeline.

Usage:
  .venv/Scripts/python -m scripts.ask "What is the population of Douglas?" --doc jjvg0227
  .venv/Scripts/python -m scripts.ask "question" --doc DOC_ID --k 5 --demo     # llm.demo_model (gpt-oss-120b)
  .venv/Scripts/python -m scripts.ask "question" --doc DOC_ID --json           # full QueryResult as JSON
  .venv/Scripts/python -m scripts.ask "question" --doc p60_281 --store demo    # demo-PDF store (auto-detected)
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from typing import List, Optional

from docscout.config import get_config, use_store


def _snip(text: str, n: int = 160) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 3] + "..."


def main(argv: Optional[List[str]] = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--doc", required=True, help="document id (see data/processed/docs/)")
    ap.add_argument("--k", type=int, default=int(cfg["retrieval"].get("final_top_k", 5)), help="evidence items")
    ap.add_argument("--demo", action="store_true", help=f"answer with llm.demo_model ({cfg['llm'].get('demo_model')})")
    ap.add_argument("--json", action="store_true", help="print the full QueryResult as JSON")
    ap.add_argument("--slots", choices=["off", "conditional", "both"], default=None,
                    help="modality slot reservation (default: answer.slots in the config)")
    ap.add_argument("--store", choices=["eval", "demo"], default=None,
                    help="where DOC_ID lives (default: demo if it is in the demo store, else eval)")
    args = ap.parse_args(argv)
    store = args.store
    if store is None:
        demo_cfg = use_store("demo")
        f = demo_cfg["paths"]["components_file"]
        store = "demo" if f.exists() and f'"doc_id": "{args.doc}"' in f.read_text(encoding="utf-8") else "eval"
    cfg = use_store(store)

    from docscout.pipeline import DocScoutPipeline

    pipe = DocScoutPipeline(cfg=cfg, model=cfg["llm"].get("demo_model") if args.demo else None)
    if pipe.doc_sizes.get(args.doc, 0) == 0:
        print(f"unknown document {args.doc!r}: no components. Ingest it and rebuild the indexes first.")
        return 1
    pipe.warm_up()
    res = pipe.answer(args.question, args.doc, top_k=args.k, slots=args.slots)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=1, default=str))
        return 0

    m = res.meta
    cls = m["classification"]
    print(f"Q: {res.query}")
    print(f"doc {args.doc} [{cfg['store']} store] ({m['doc_components']} components) | type {res.query_type} "
          f"({cls.get('source')}, conf {float(cls.get('confidence', 0)):.2f}) | branch weights {m['branch_weights']}")
    print(f"\nANSWER: {res.answer}")
    print(f"citations: {', '.join(m['citations']) or '-'}"
          + (f" | dropped invalid citations: {m['answer_validation']['invalid_citations']}"
             if m['answer_validation']['invalid_citations'] else "")
          + (" | no valid citation -> not found" if m['answer_validation']['no_valid_citation'] else ""))
    print("\nEVIDENCE")
    for e in res.evidence:
        c = e.component
        print(f"[{e.meta['eid']}] {c.type:<9} p{c.page:<3} {c.id}")
        print(textwrap.indent(_snip(c.text or c.text_form), "      "))
        if e.localized_cells:
            print(f"      localized cells ({e.meta.get('localize_flag')}):")
            for r, col, v in e.localized_cells[:8]:
                print(f"        {_snip(r, 40)} | {_snip(col, 50)} = {v}")
        elif e.meta.get("localize_flag"):
            print(f"      localization: {e.meta['localize_flag']}")
        for j, x in enumerate(e.context, start=1):
            print(f"      ctx{j} {x.type} p{x.page}: {_snip(x.text or x.text_form, 90)}")
        print(f"      why: {e.why}")
    t = m["timings_ms"]
    print(f"\ntimings ms: " + ", ".join(f"{k} {v:.0f}" for k, v in t.items()))
    llm = m["llm"]
    print(f"LLM model {m['model']} | calls {llm['calls']} cache_hits {llm['cache_hits']} tokens {llm['total_tokens']} "
          f"| prompt est {m['prompt'].get('est_tokens')} tokens, ids {len(m['prompt'].get('ids', []))} "
          f"| reranked list: {len(m['reranked'])} components")
    return 0


if __name__ == "__main__":
    sys.exit(main())
