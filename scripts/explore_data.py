"""Print stats for the downloaded MP-DocVQA subset. Logic: docscout.dataset.explore.

Usage: .venv/Scripts/python -m scripts.explore_data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from docscout.config import get_config
from docscout.dataset.explore import explore


def main(argv: Optional[List[str]] = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=Path, default=cfg["paths"]["questions_file"])
    ap.add_argument("--images", type=Path, default=cfg["paths"]["mpdocvqa_images_dir"])
    args = ap.parse_args(argv)
    return explore(args.questions, args.images, cfg)


if __name__ == "__main__":
    sys.exit(main())
