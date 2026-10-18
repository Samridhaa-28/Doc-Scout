"""Search the saved indexes. Logic: docscout.indexing.search.

Usage:
  .venv/Scripts/python -m scripts.search_cli "population in poverty" --branch table --retriever bm25 --doc jjvg0227
  .venv/Scripts/python -m scripts.search_cli "bar chart" --branch figure --retriever clip --doc demo-report --k 3
  .venv/Scripts/python -m scripts.search_cli "query" --branch text --retriever all --target COMPONENT_ID --show-encoded
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from docscout.config import get_config
from docscout.indexing.search import search_report


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
    a = ap.parse_args(argv)
    return search_report(a.query, a.branch, a.retriever, a.doc, a.k, a.target, a.show_encoded, cfg)


if __name__ == "__main__":
    sys.exit(main())
