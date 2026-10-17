"""Build all branch indexes (BM25, dense BGE, dense MiniLM, CLIP for figures) from components.jsonl.

Existing up-to-date indexes are loaded, not rebuilt. If components.jsonl changed since an index
was built (fingerprint = line count + sha256), the script warns and exits: run with --rebuild.

Usage:
  .venv/Scripts/python -m scripts.build_indexes
  .venv/Scripts/python -m scripts.build_indexes --rebuild
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import List, Optional

from docscout.config import get_config
from docscout.indexing.build_all import build_all, stale_warning
from docscout.schema import iter_jsonl


def main(argv: Optional[List[str]] = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="rebuild every index from scratch")
    args = ap.parse_args(argv)

    comp_path = cfg["paths"]["components_file"]
    if not comp_path.exists():
        print(f"{comp_path} not found. Run scripts.ingest first.")
        return 1
    rows = list(iter_jsonl(comp_path))
    docs = sorted({r["doc_id"] for r in rows})
    print(f"components.jsonl: {len(rows)} components, {len(docs)} documents {docs}")
    print("by type:", dict(Counter(r["type"] for r in rows)))

    res = build_all(rebuild=args.rebuild, cfg=cfg)
    print(f"fingerprint: lines={res['fingerprint']['lines']} sha256={res['fingerprint']['sha256'][:16]}...")
    if res["stale"]:
        print(stale_warning(res["stale"]))
        return 2

    print(f"\n{'branch':<8}{'retriever':<14}{'n':>5}  {'action':<22}{'secs':>9}{'size':>12}")
    total = 0.0
    for r in res["rows"]:
        size = f"{r.get('size_bytes', 0) / 1024:.1f} KB"
        print(f"{r['branch']:<8}{r['retriever']:<14}{r['n']:>5}  {r['action']:<22}{r['secs']:>9.3f}{size:>12}")
        total += r["secs"]
    print(f"total: {total:.1f}s | indexes in {cfg['paths']['indexes_dir'].relative_to(cfg['repo_root'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
