"""Download an MP-DocVQA validation subset (page images + questions). Logic: docscout.dataset.mpdocvqa.

Usage:
  .venv/Scripts/python -m scripts.download_data --limit 5
  .venv/Scripts/python -m scripts.download_data --limit 150
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from docscout.config import get_config
from docscout.dataset.mpdocvqa import download_subset


def main(argv: Optional[List[str]] = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=cfg["dataset"]["subset_size"], help="number of documents")
    ap.add_argument("--dataset", default=cfg["dataset"]["hf_name"])
    ap.add_argument("--split", default=cfg["dataset"]["split"])
    args = ap.parse_args(argv)
    return download_subset(args.limit, args.dataset, args.split, cfg)


if __name__ == "__main__":
    sys.exit(main())
