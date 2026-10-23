"""Build all branch indexes (BM25, dense BGE, dense MiniLM, CLIP for figures). Logic: docscout.indexing.build_all.

Up-to-date indexes are loaded, not rebuilt. If components.jsonl changed since an index was built,
the script warns and exits with code 2: run with --rebuild.

Usage:
  .venv/Scripts/python -m scripts.build_indexes
  .venv/Scripts/python -m scripts.build_indexes --rebuild
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from docscout.config import use_store

from docscout.indexing.build_all import build_report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", choices=["eval", "demo"], default="eval",
                    help="eval: MP-DocVQA store (default); demo: separate store for the demo PDFs")
    ap.add_argument("--rebuild", action="store_true", help="rebuild every index from scratch")
    args = ap.parse_args(argv)
    use_store(args.store)
    return build_report(rebuild=args.rebuild)


if __name__ == "__main__":
    sys.exit(main())
