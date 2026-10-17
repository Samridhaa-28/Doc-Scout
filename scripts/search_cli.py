"""Search the saved indexes.

Prints rank, score, component id, type, page and the first 120 chars of text_form.

Usage:
  .venv/Scripts/python -m scripts.search_cli "population in poverty" --branch table --retriever bm25 --doc jjvg0227
  .venv/Scripts/python -m scripts.search_cli "bar graph" --branch figure --retriever clip --doc ttwl0228 --k 3
  .venv/Scripts/python -m scripts.search_cli "query" --branch text --retriever all --target COMPONENT_ID --show-encoded
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

from docscout.config import get_config
from docscout.indexing.build_all import load_index, retrievers_for, stale_indexes, stale_warning
from docscout.indexing.dense_index import DenseIndex
from docscout.schema import Component, iter_jsonl


def main(argv: Optional[List[str]] = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--branch", default="all", choices=["all", *cfg["branches"].keys()])
    ap.add_argument("--retriever", default="bm25", choices=["all", "bm25", "dense_bge", "dense_minilm", "clip"])
    ap.add_argument("--doc", default=None, help="restrict to one document (exact doc_id)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--target", default=None, help="also print the rank of this component id")
    ap.add_argument("--show-encoded", action="store_true", help="print the exact query string sent to dense models")
    args = ap.parse_args(argv)

    stale = stale_indexes(cfg)
    if stale:
        print(stale_warning(stale))

    comps: Dict[str, Component] = {r["id"]: Component.from_dict(r) for r in iter_jsonl(cfg["paths"]["components_file"])}
    branches = list(cfg["branches"]) if args.branch == "all" else [args.branch]
    all_docs_ok = True
    for branch in branches:
        rets = retrievers_for(branch) if args.retriever == "all" else [args.retriever]
        for ret in rets:
            if ret not in retrievers_for(branch):
                print(f"[{branch}/{ret}] not available for this branch")
                continue
            idx = load_index(branch, ret, cfg, warn=False)
            if idx is None:
                print(f"[{branch}/{ret}] index missing. Run: .venv/Scripts/python -m scripts.build_indexes")
                continue
            print(f"\n=== {branch}/{ret}  query={args.query!r}" + (f"  doc={args.doc}" if args.doc else "") + " ===")
            if args.show_encoded and isinstance(idx, DenseIndex):
                print(f"encoded query string: {idx.query_text(args.query)!r}")
            full = idx.search(args.query, top_k=max(len(idx), 1), doc_id=args.doc)
            for rank, (cid, score) in enumerate(full[: args.k], start=1):
                c = comps.get(cid)
                snippet = c.text_form[:120].replace("\n", " ") if c else "?"
                print(f"{rank:>3}  {score:8.4f}  {cid:<28} {c.type if c else '?':<9} p{c.page if c else '?':<4} {snippet}")
            if not full:
                print("  (no results)")
            if args.doc:
                ok = all(comps[cid].doc_id == args.doc for cid, _ in full)
                all_docs_ok &= ok
                print(f"  doc filter: {len(full)} results, all from {args.doc}: {ok}")
            if args.target:
                ranks = [i for i, (cid, _) in enumerate(full, start=1) if cid == args.target]
                print(f"  rank of {args.target}: {ranks[0] if ranks else 'not retrieved'} (of {len(full)} results)")
    return 0 if all_docs_ok else 1


if __name__ == "__main__":
    sys.exit(main())
