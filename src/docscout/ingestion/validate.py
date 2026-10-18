"""Validate data/processed/components.jsonl.

Checks: unique ids (line count vs unique-id count), id format, non-empty text_form, page >= 1,
bbox (and extra regions) inside the page, page size = manifest size, table/figure files exist,
and for MP-DocVQA: every component page is in meta.json, every meta.json page has components unless the
parser recorded it as processed-but-empty (blank page; reported as info), and every answer_page has components.

Command line: .venv/Scripts/python -m scripts.check_components
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from docscout.config import get_config
from docscout.schema import Component, iter_jsonl


def duplicate_ids(ids: List[str]) -> Dict[str, int]:
    """Ids that occur more than once -> count."""
    return {i: n for i, n in Counter(ids).items() if n > 1}


def bbox_inside(c: Component) -> bool:
    boxes = [c.bbox] + [tuple(r["bbox"]) for r in c.meta.get("regions", [])]
    return all(0 <= x0 <= x1 <= c.page_width and 0 <= y0 <= y1 <= c.page_height for x0, y0, x1, y1 in boxes)


def validate(path: Path, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run every check; returns counts and lists of failures (empty lists = pass)."""
    cfg = cfg or get_config()
    root = cfg["repo_root"]
    comps = [Component.from_dict(r) for r in iter_jsonl(path)]
    by_doc: Dict[str, List[Component]] = defaultdict(list)
    for c in comps:
        by_doc[c.doc_id].append(c)
    res: Dict[str, Any] = {
        "lines": len(comps),
        "unique_ids": len({c.id for c in comps}),
        "duplicate_ids": duplicate_ids([c.id for c in comps]),
        "bad_id_format": [c.id for c in comps if not c.id.startswith(f"{c.doc_id}_p{c.page}_{c.type}_")],
        "empty_text_form": [c.id for c in comps if not c.text_form.strip()],
        "bad_page": [c.id for c in comps if c.page < 1],
        "bbox_outside": [c.id for c in comps if not bbox_inside(c)],
        "missing_files": [c.id for c in comps
                          if (c.table_path and not (root / c.table_path).exists())
                          or (c.image_path and not (root / c.image_path).exists())],
        "size_mismatch": [], "page_mismatch": [], "answer_page_missing": [], "blank_pages": [], "docs": {},
    }
    q_path: Path = cfg["paths"]["questions_file"]
    questions = list(iter_jsonl(q_path)) if q_path.exists() else []
    for doc_id, dcomps in sorted(by_doc.items()):
        man = json.loads((cfg["paths"]["docs_dir"] / f"{doc_id}.json").read_text(encoding="utf-8"))
        sizes = {p["page"]: (p["width"], p["height"]) for p in man["pages"]}
        res["size_mismatch"] += [c.id for c in dcomps if sizes.get(c.page) != (c.page_width, c.page_height)]
        pages = sorted({c.page for c in dcomps})
        info: Dict[str, Any] = {"source": man["source"], "components": len(dcomps), "pages": pages}
        if man["source"] == "mpdocvqa":
            meta = json.loads((cfg["paths"]["mpdocvqa_images_dir"] / doc_id / "meta.json").read_text(encoding="utf-8"))
            # Fail on a component page that does not exist, or a page that is missing WITHOUT being recorded by
            # the parser as processed-but-empty (blank pages legitimately have no components).
            recorded_empty = set(man.get("parse_stats", {}).get("pages_without_components", []))
            unknown = sorted(set(pages) - set(meta["page_numbers"]))
            lost = sorted(set(meta["page_numbers"]) - set(pages) - recorded_empty)
            if unknown or lost:
                res["page_mismatch"].append((doc_id, {"unknown_pages": unknown, "lost_pages": lost}))
            if recorded_empty:
                res["blank_pages"].append((doc_id, sorted(recorded_empty)))
            answers = sorted({q["answer_page"] for q in questions if q["doc_id"] == doc_id})
            missing = [a for a in answers if a not in pages]
            if missing:
                res["answer_page_missing"].append((doc_id, missing))
            info.update({"meta_page_numbers": meta["page_numbers"], "answer_pages": answers})
        res["docs"][doc_id] = info
    return res


def report(path: Optional[Path] = None, cfg: Optional[Dict[str, Any]] = None, per_doc: bool = False) -> int:
    """Run validate() and print a summary (per-document lines only with per_doc=True). Returns 1 if a check fails."""
    cfg = cfg or get_config()
    path = Path(path) if path else cfg["paths"]["components_file"]
    if not path.exists():
        print(f"{path} not found")
        return 1
    r = validate(path, cfg)
    print(f"lines: {r['lines']} | unique ids: {r['unique_ids']} | duplicates: {len(r['duplicate_ids'])}")
    print(f"documents: {len(r['docs'])}")
    if per_doc:
        for doc_id, info in r["docs"].items():
            print(f"  {doc_id}: {info}")
    failed = False
    for key in ("duplicate_ids", "bad_id_format", "empty_text_form", "bad_page", "bbox_outside", "missing_files",
                "size_mismatch", "page_mismatch", "answer_page_missing"):
        n = len(r[key])
        failed |= n > 0
        print(f"{'FAIL' if n else 'ok  '} {key}: {n}" + (f" e.g. {list(r[key])[:3]}" if n else ""))
    print(f"info blank_pages (processed by the parser, no components): {r['blank_pages']}")
    print("ALL CHECKS PASSED" if not failed else "SOME CHECKS FAILED")
    return 1 if failed else 0
