"""Draw component bboxes on each page of a document. Logic: docscout.ingestion.visualize.

Output: data/processed/debug/{doc_id}/page_{n}.png
Usage: .venv/Scripts/python -m scripts.visualize_components --doc snbx0223 [--verbose]
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from docscout.config import get_config
from docscout.ingestion.visualize import render_doc


def main(argv: Optional[List[str]] = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--verbose", action="store_true", help="one line per page")
    args = ap.parse_args(argv)
    written = render_doc(args.doc, cfg)
    if not written:
        print(f"No components for {args.doc}. Run scripts.ingest first.")
        return 1
    for page, n, path in written if args.verbose else []:
        print(f"page {page}: {n} components -> {path.relative_to(cfg['repo_root']).as_posix()}")
    folder = (cfg["paths"]["debug_dir"] / args.doc).relative_to(cfg["repo_root"]).as_posix()
    print(f"{args.doc}: {len(written)} pages, {sum(n for _, n, _ in written)} components drawn -> {folder}/page_<n>.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
