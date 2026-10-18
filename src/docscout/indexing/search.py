"""Interactive search over the saved indexes (used by scripts/search_cli.py).

Prints rank, score, component id, type, page and the first 120 chars of text_form.
"""
from __future__ import annotations

from typing import Dict, Optional

from docscout.config import get_config
from docscout.indexing.build_all import load_index, retrievers_for, stale_indexes, stale_warning
from docscout.indexing.dense_index import DenseIndex
from docscout.schema import Component, iter_jsonl


def search_report(query: str, branch: str = "all", retriever: str = "bm25", doc: Optional[str] = None, k: int = 5,
                  target: Optional[str] = None, show_encoded: bool = False, cfg: Optional[Dict] = None) -> int:
    """Search one or all branches with one or all retrievers and print the top-k.

    With `doc`, verifies every result comes from that document; with `target`, prints that component's rank.
    Returns 0, or 1 if the doc filter leaked a result from another document.
    """
    cfg = cfg or get_config()
    stale = stale_indexes(cfg)
    if stale:
        print(stale_warning(stale))

    comps: Dict[str, Component] = {r["id"]: Component.from_dict(r) for r in iter_jsonl(cfg["paths"]["components_file"])}
    branches = list(cfg["branches"]) if branch == "all" else [branch]
    all_docs_ok = True
    for b in branches:
        rets = retrievers_for(b) if retriever == "all" else [retriever]
        for ret in rets:
            if ret not in retrievers_for(b):
                print(f"[{b}/{ret}] not available for this branch")
                continue
            idx = load_index(b, ret, cfg, warn=False)
            if idx is None:
                print(f"[{b}/{ret}] index missing. Run: .venv/Scripts/python -m scripts.build_indexes")
                continue
            print(f"\n=== {b}/{ret}  query={query!r}" + (f"  doc={doc}" if doc else "") + " ===")
            if show_encoded and isinstance(idx, DenseIndex):
                print(f"encoded query string: {idx.query_text(query)!r}")
            full = idx.search(query, top_k=max(len(idx), 1), doc_id=doc)
            for rank, (cid, score) in enumerate(full[:k], start=1):
                c = comps.get(cid)
                snippet = c.text_form[:120].replace("\n", " ") if c else "?"
                print(f"{rank:>3}  {score:8.4f}  {cid:<28} {c.type if c else '?':<9} p{c.page if c else '?':<4} {snippet}")
            if not full:
                print("  (no results)")
            if doc:
                ok = all(comps[cid].doc_id == doc for cid, _ in full)
                all_docs_ok &= ok
                print(f"  doc filter: {len(full)} results, all from {doc}: {ok}")
            if target:
                ranks = [i for i, (cid, _) in enumerate(full, start=1) if cid == target]
                print(f"  rank of {target}: {ranks[0] if ranks else 'not retrieved'} (of {len(full)} results)")
    return 0 if all_docs_ok else 1
