"""Build page-level and component-level qrels (ranx format) for the evaluation documents.

Page qrels     : question -> {"{doc_id}_p{answer_page}": 1}. answer_page in questions.jsonl is already the
                 1-based original page number (= page_ids[answer_page_idx] '_p{k}' + 1 = Component.page).
Component qrels: components ON the answer page whose text (text_form, text, or any table cell / header
                 from the parquet file) contains a gold answer.
                 Normalization: lowercase, keep only [a-z0-9]. Answers longer than 3 normalized chars match
                 with rapidfuzz partial_ratio >= 90; shorter answers need an exact match with a whole
                 normalized word or a whole table cell (a plain substring test would match "5" almost anywhere).

Usage: .venv/Scripts/python -m eval.build_qrels
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from rapidfuzz import fuzz

from docscout.config import get_config
from docscout.schema import Component, iter_jsonl

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_NUMERIC_RE = re.compile(r"^[\s$€£¥+\-]*\d[\d\s,.\-/:]*[%]?\s*$")


def norm(text: str) -> str:
    """Lowercase and strip everything except letters and digits."""
    return _NON_ALNUM.sub("", str(text).lower())


def is_numeric_answer(answer: str) -> bool:
    """'0.28', '$1,234', '26.9%', '1975', '12/03/1998' -> True; 'University of California' -> False."""
    return bool(_NUMERIC_RE.match(str(answer)))


def page_key(doc_id: str, page: int) -> str:
    return f"{doc_id}_p{page}"


@lru_cache(maxsize=None)
def table_cells(table_path: str) -> Tuple[str, ...]:
    """All cell values and column headers of a saved table (raw strings)."""
    import pandas as pd

    p = Path(table_path)
    p = p if p.is_absolute() else get_config()["repo_root"] / p
    if not p.exists():
        return ()
    df = pd.read_parquet(p)
    vals = [str(c) for c in df.columns] + [str(v) for v in df.astype(str).values.ravel().tolist()]
    return tuple(v for v in vals if v.strip())


def component_texts(c: Component) -> Tuple[List[str], List[str]]:
    """(long texts to fuzzy-match, atomic units for exact short-answer match: words + table cells)."""
    texts = [c.text_form, c.text]
    units = [w for t in texts for w in str(t).split()]
    if c.type == "table" and c.table_path:
        cells = list(table_cells(c.table_path))
        texts += cells
        units += cells
    return texts, units


def contains_answer(a: str, t: str, threshold: int) -> bool:
    """Does normalized text t (approximately) CONTAIN normalized answer a?

    partial_ratio aligns the shorter string inside the longer one, so it is only meaningful when the text
    is at least as long as the answer; otherwise ('0' vs '(910) 741-5327' would score 100) we require
    the whole text to match the whole answer (ratio).
    """
    if not t:
        return False
    if len(t) >= len(a):
        return fuzz.partial_ratio(a, t) >= threshold
    return fuzz.ratio(a, t) >= threshold


def matches(answer: str, texts: Iterable[str], units: Iterable[str], threshold: int = 90) -> bool:
    a = norm(answer)
    if not a:
        return False
    if len(a) > 3:
        return any(contains_answer(a, norm(t), threshold) for t in texts if t)
    return any(norm(u) == a for u in units)


def relevant_components(question: dict, page_comps: List[Component], threshold: int = 90) -> List[str]:
    out = []
    for c in page_comps:
        texts, units = component_texts(c)
        if any(matches(ans, texts, units, threshold) for ans in question["answers"]):
            out.append(c.id)
    return out


def load_eval_docs(cfg) -> List[str]:
    path = cfg["paths"]["mpdocvqa_dir"] / "eval_docs.txt"
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=int, default=int(cfg["eval"]["fuzzy_match_threshold"]))
    args = ap.parse_args(argv)

    eval_docs = load_eval_docs(cfg)
    eval_set = set(eval_docs)
    questions = [q for q in iter_jsonl(cfg["paths"]["questions_file"]) if q["doc_id"] in eval_set]
    comps = [Component.from_dict(r) for r in iter_jsonl(cfg["paths"]["components_file"]) if r["doc_id"] in eval_set]
    by_page: Dict[Tuple[str, int], List[Component]] = defaultdict(list)
    for c in comps:
        by_page[(c.doc_id, c.page)].append(c)
    ingested = {c.doc_id for c in comps}
    missing_docs = sorted(eval_set - ingested)
    if missing_docs:
        print(f"WARNING: {len(missing_docs)} eval documents are not ingested: {missing_docs}")

    num_pages = {d: json.loads((cfg["paths"]["mpdocvqa_images_dir"] / d / "meta.json").read_text())["num_pages"]
                 for d in eval_docs}
    qrels_page: Dict[str, Dict[str, int]] = {}
    qrels_comp: Dict[str, Dict[str, int]] = {}
    info: Dict[str, dict] = {}
    for q in questions:
        qid = q["question_id"]
        qrels_page[qid] = {page_key(q["doc_id"], q["answer_page"]): 1}
        page_comps = by_page.get((q["doc_id"], q["answer_page"]), [])
        rel = relevant_components(q, page_comps, args.threshold)
        if rel:
            qrels_comp[qid] = {cid: 1 for cid in rel}
        info[qid] = {
            "doc_id": q["doc_id"], "num_pages": num_pages[q["doc_id"]],
            "numeric": all(is_numeric_answer(a) for a in q["answers"]),
            "table_page": any(c.type == "table" for c in page_comps),
            "n_relevant": len(rel), "n_page_components": len(page_comps),
        }

    out_dir: Path = cfg["paths"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qrels_page.json").write_text(json.dumps(qrels_page, indent=1), encoding="utf-8")
    (out_dir / "qrels_component.json").write_text(json.dumps(qrels_comp, indent=1), encoding="utf-8")
    (out_dir / "qrels_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")

    def cov(qids: List[str]) -> str:
        n = len(qids)
        k = sum(1 for qid in qids if qid in qrels_comp)
        return f"{k}/{n} = {100 * k / n:.1f}%" if n else "n/a"

    qids = list(info)
    print(f"eval documents: {len(eval_docs)} (ingested {len(ingested)}) | questions: {len(questions)}")
    print(f"page qrels: {len(qrels_page)} questions")
    print(f"component qrels: {len(qrels_comp)} questions with >=1 relevant component; "
          f"{len(questions) - len(qrels_comp)} excluded from component metrics (kept in page metrics)")
    print(f"relevant components per covered question: mean "
          f"{sum(len(v) for v in qrels_comp.values()) / max(len(qrels_comp), 1):.2f}")
    print("\n=== Coverage (questions with >=1 relevant component) ===")
    print(f"  all questions                : {cov(qids)}")
    print(f"  numeric answers              : {cov([q for q in qids if info[q]['numeric']])}")
    print(f"  text answers                 : {cov([q for q in qids if not info[q]['numeric']])}")
    print(f"  answer page has a table      : {cov([q for q in qids if info[q]['table_page']])}")
    print(f"  answer page has no table     : {cov([q for q in qids if not info[q]['table_page']])}")
    print(f"  docs with >=2 pages          : {cov([q for q in qids if info[q]['num_pages'] >= 2])}")
    print(f"  answer page has no components: {sum(1 for q in qids if info[q]['n_page_components'] == 0)} questions")
    print(f"\nanswer kinds: {dict(Counter('numeric' if info[q]['numeric'] else 'text' for q in qids))} | "
          f"table pages: {sum(info[q]['table_page'] for q in qids)}")
    print(f"wrote {', '.join(p for p in ('qrels_page.json', 'qrels_component.json', 'qrels_info.json'))} to "
          f"{out_dir.relative_to(cfg['repo_root'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
