"""Dev/test split of the evaluation questions BY DOCUMENT (no document in both splits).

The evaluation documents (eval_docs.txt, in saved order) are shuffled with random.Random(seed) and the first
round(dev_fraction * n_docs) go to dev; the rest are test. Every question follows its document.
Dev is used only for tuning; every main table is reported on test.

Output: eval/results/split.json = {seed, dev_fraction, dev: {doc_ids, question_ids}, test: {...}}

Usage: .venv/Scripts/python -m eval.split
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from docscout.config import get_config
from docscout.schema import iter_jsonl
from eval.build_qrels import load_eval_docs


def split_by_document(doc_ids: Sequence[str], question_docs: Mapping[str, str], dev_fraction: float,
                      seed: int) -> Dict[str, Dict[str, List[str]]]:
    """{"dev": {"doc_ids", "question_ids"}, "test": {...}}; question_docs = {question_id: doc_id}."""
    docs = list(doc_ids)
    random.Random(seed).shuffle(docs)
    n_dev = round(dev_fraction * len(docs))
    dev_docs, test_docs = set(docs[:n_dev]), set(docs[n_dev:])
    out = {}
    for name, ds in (("dev", dev_docs), ("test", test_docs)):
        out[name] = {"doc_ids": sorted(ds), "question_ids": [q for q, d in question_docs.items() if d in ds]}
    return out


def split_file(cfg: Optional[dict] = None) -> Path:
    cfg = cfg or get_config()
    return cfg["repo_root"] / cfg["eval"]["split_file"]


def load_split(cfg: Optional[dict] = None) -> Dict[str, Dict[str, List[str]]]:
    """Read split.json (run `python -m eval.split` first)."""
    return json.loads(split_file(cfg).read_text(encoding="utf-8"))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    cfg = get_config()
    results_dir: Path = cfg["paths"]["results_dir"]
    qrels_page = json.loads((results_dir / "qrels_page.json").read_text(encoding="utf-8"))
    info = json.loads((results_dir / "qrels_info.json").read_text(encoding="utf-8"))
    docs = load_eval_docs(cfg)
    dset = set(docs)
    qdocs = {q["question_id"]: q["doc_id"] for q in iter_jsonl(cfg["paths"]["questions_file"])
             if q["doc_id"] in dset and q["question_id"] in qrels_page}
    seed, frac = int(cfg["eval"]["seed"]), float(cfg["eval"]["dev_fraction"])
    sp = split_by_document(docs, qdocs, frac, seed)
    overlap = set(sp["dev"]["doc_ids"]) & set(sp["test"]["doc_ids"])
    assert not overlap, overlap
    out = {"seed": seed, "dev_fraction": frac, **sp}
    f = split_file(cfg)
    f.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name in ("dev", "test"):
        qs = sp[name]["question_ids"]
        multi = sum(1 for q in qs if info[q]["num_pages"] >= 2)
        print(f"{name:<5}: {len(sp[name]['doc_ids']):>2} documents, {len(qs):>3} questions, "
              f"{multi:>3} from documents with >= 2 pages")
    print(f"total: {len(qdocs)} questions | document overlap: {len(overlap)} | seed {seed}, dev fraction {frac}")
    print(f"wrote {f.relative_to(cfg['repo_root'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
