"""Print stats for the downloaded MP-DocVQA subset.

Usage: .venv/Scripts/python -m scripts.explore_data
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from docscout.config import get_config
from docscout.schema import read_jsonl


def main(argv: Optional[List[str]] = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=Path, default=cfg["paths"]["questions_file"])
    ap.add_argument("--images", type=Path, default=cfg["paths"]["mpdocvqa_images_dir"])
    args = ap.parse_args(argv)

    if not args.questions.exists():
        print(f"{args.questions} not found. Run: .venv/Scripts/python -m scripts.download_data --limit 5")
        return 1

    questions = read_jsonl(args.questions)
    metas: Dict[str, dict] = {}
    for meta_path in sorted(args.images.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["n_png"] = len(list(meta_path.parent.glob("page_*.png")))
        metas[meta["doc_id"]] = meta

    pages = [m["num_pages"] for m in metas.values()]
    print("=== MP-DocVQA subset ===")
    print(f"documents (with images): {len(metas)}")
    print(f"documents referenced by questions: {len({q['doc_id'] for q in questions})}")
    if pages:
        print(f"pages per doc: min={min(pages)} mean={sum(pages) / len(pages):.2f} max={max(pages)} "
              f"(total pages: {sum(pages)})")
        mismatched = [d for d, m in metas.items() if m["n_png"] != m["num_pages"]]
        print(f"docs where png count != num_pages: {len(mismatched)}")
    print(f"questions: {len(questions)} (per doc: {len(questions) / max(len(metas), 1):.2f})")

    missing = [q["question_id"] for q in questions if "answer_page_idx" not in q or not q.get("answers")]
    print(f"questions missing answer_page_idx or answers: {len(missing)}")
    out_of_range = [q["question_id"] for q in questions if not 0 <= q["answer_page_idx"] < len(q["page_ids"])]
    print(f"questions with answer_page_idx out of range: {len(out_of_range)}")
    bad_map = [q["question_id"] for q in questions if q["page_ids"][q["answer_page_idx"]] != q["answer_page_id"]]
    no_img = [q["question_id"] for q in questions
              if not (args.images / q["doc_id"] / f"page_{q['answer_page']}.png").exists()]
    print(f"questions whose answer_page_id != page_ids[answer_page_idx]: {len(bad_map)}")
    print(f"questions whose answer page image is missing: {len(no_img)}")

    windows = [len(q["page_ids"]) for q in questions]
    print(f"question page window (len(page_ids)): min={min(windows)} mean={sum(windows) / len(windows):.2f} "
          f"max={max(windows)}")

    print("\nanswer_page_idx distribution (0-based index into the question's page_ids -> count):")
    for idx, n in sorted(Counter(q["answer_page_idx"] for q in questions).items()):
        print(f"  {idx:>2}: {n:>4} {'#' * min(n, 60)}")
    # position of the answer page among the doc's saved pages (what retrieval sees)
    pos = Counter(metas[q["doc_id"]]["page_numbers"].index(q["answer_page"]) + 1 for q in questions)
    print("answer page position within the saved document (1 = first saved page -> count):")
    for p, n in sorted(pos.items()):
        print(f"  {p:>2}: {n:>4} {'#' * min(n, 60)}")
    single = sum(1 for q in questions if metas[q["doc_id"]]["num_pages"] == 1)
    print(f"questions on single-page docs (page retrieval trivial): {single}/{len(questions)}")

    print("\nquestion types:")
    if any("question_types" in q for q in questions):
        qt = Counter(t for q in questions for t in q.get("question_types", []))
        for t, n in qt.most_common():
            print(f"  {t:<20} {n}")
    else:
        print("  not available in this dataset (no question_types field)")

    ex = questions[0]
    meta = metas.get(ex["doc_id"], {})
    print("\n=== One full example ===")
    print(json.dumps(ex, indent=2, ensure_ascii=False))
    if meta:
        print(f"doc meta: num_pages={meta['num_pages']} page_numbers={meta['page_numbers']} "
              f"page_sizes={meta['page_sizes']}")
        img = args.images / ex["doc_id"] / f"page_{ex['answer_page']}.png"
        print(f"answer page: page_ids[{ex['answer_page_idx']}] = {ex['answer_page_id']} -> page {ex['answer_page']} "
              f"-> {img.relative_to(cfg['repo_root'])} exists={img.exists()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
