"""Validate data/processed/components.jsonl. Logic: docscout.ingestion.validate.

Usage: .venv/Scripts/python -m scripts.check_components [--per-doc]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from docscout.ingestion.validate import report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=Path, default=None, help="components.jsonl (default: config paths.components_file)")
    ap.add_argument("--per-doc", action="store_true", help="also print one line per document")
    args = ap.parse_args(argv)
    return report(args.path, per_doc=args.per_doc)


if __name__ == "__main__":
    sys.exit(main())
