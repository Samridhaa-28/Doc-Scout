"""Select the evaluation documents and write data/raw/mpdocvqa/eval_docs.txt.

Rule: walk the 150 downloaded documents in their saved order (first appearance in the split),
always keep documents already ingested, skip documents with more than --max-pages saved pages,
and stop once the total reaches about --target-pages pages.

Usage: .venv/Scripts/python -m eval.select_eval_docs --target-pages 250 --max-pages 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from docscout.config import get_config
from docscout.schema import iter_jsonl


def main(argv: Optional[List[str]] = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-pages", type=int, default=250)
    ap.add_argument("--max-pages", type=int, default=20)
    ap.add_argument("--out", type=Path, default=cfg["paths"]["mpdocvqa_dir"] / "eval_docs.txt")
    args = ap.parse_args(argv)

    questions = list(iter_jsonl(cfg["paths"]["questions_file"]))
    order = list(dict.fromkeys(q["doc_id"] for q in questions))  # saved order of the 150 documents
    pages: Dict[str, int] = {}
    for d in order:
        meta = json.loads((cfg["paths"]["mpdocvqa_images_dir"] / d / "meta.json").read_text(encoding="utf-8"))
        pages[d] = meta["num_pages"]
    comp_path = cfg["paths"]["components_file"]
    ingested = {r["doc_id"] for r in iter_jsonl(comp_path)} if comp_path.exists() else set()
    ingested &= set(order)
    # documents whose ingest failed with the current MP-DocVQA backend are replaced by the next eligible ones
    from docscout.ingestion.runner import failed_doc_ids
    from docscout.ingestion.parser import pdf_backend_for

    failed = failed_doc_ids(cfg, pdf_backend_for("mpdocvqa", cfg)) - ingested

    selected = [d for d in order if d in ingested]  # always keep, in saved order
    total = sum(pages[d] for d in selected)
    skipped_big, skipped_failed = [], []
    for d in order:
        if total >= args.target_pages:
            break
        if d in selected:
            continue
        if d in failed:
            skipped_failed.append(d)
            continue
        if pages[d] > args.max_pages:
            skipped_big.append(d)
            continue
        selected.append(d)
        total += pages[d]
    selected = [d for d in order if d in set(selected)]  # keep saved order

    args.out.write_text("\n".join(selected) + "\n", encoding="utf-8")
    pp = [pages[d] for d in selected]
    qs = [q for q in questions if q["doc_id"] in set(selected)]
    multi = [q for q in qs if pages[q["doc_id"]] >= 2]
    print(f"wrote {args.out}")
    print(f"documents: {len(selected)} (already ingested kept: {sorted(ingested)})")
    print(f"pages: {sum(pp)} | pages/doc min={min(pp)} mean={mean(pp):.2f} max={max(pp)}")
    print(f"single-page documents: {sum(1 for p in pp if p == 1)}")
    print(f"questions kept: {len(qs)} | on documents with >=2 pages: {len(multi)}")
    print(f"skipped (> {args.max_pages} pages) while selecting: {len(skipped_big)} {skipped_big}")
    print(f"skipped (ingest failed): {len(skipped_failed)} {skipped_failed}")
    print(f"remaining to ingest: {len([d for d in selected if d not in ingested])} docs, "
          f"{sum(pages[d] for d in selected if d not in ingested)} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
