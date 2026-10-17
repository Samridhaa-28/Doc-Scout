"""Download an MP-DocVQA validation subset (page images + questions) via HF streaming.

How MP-DocVQA is laid out (verified by inspecting the data, see P1 report):
  * one row = one question; it carries a contiguous window of 1..20 page images
    (image_1..image_20) given by `page_ids`, e.g. ['snbx0223_p3', 'snbx0223_p4'];
  * rows are NOT grouped by document, and different questions about the same
    document can carry different page windows.

Strategy (two streaming passes, never downloads the whole 14 GB split):
  1. Metadata pass: stream only the text columns of the split (no images) and
     cache them in data/raw/mpdocvqa/{split}_index.jsonl. Select the first N
     documents in order of first appearance.
  2. Image pass: stream full rows from the start until the row where the N-th
     selected document first appears; save every page of every selected doc seen
     so far. A document's page set = union of the windows seen in that prefix.
  3. Keep every question of a selected doc whose answer page was saved.

Output:
  data/raw/mpdocvqa/images/{doc_id}/page_{i}.png  i = ORIGINAL 1-based page number
                                                  (page id '{doc_id}_p{k}' -> page_{k+1}.png)
  data/raw/mpdocvqa/images/{doc_id}/meta.json     saved page ids / numbers / sizes
  data/raw/mpdocvqa/questions.jsonl               rewritten on every run (deterministic)

Usage:
  .venv/Scripts/python -m scripts.download_data --limit 5
  .venv/Scripts/python -m scripts.download_data --limit 150
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

from docscout.config import get_config
from docscout.schema import iter_jsonl, write_jsonl

# Candidate source names for each target field. The real names are resolved
# against the dataset's printed features at runtime; nothing is assumed.
FIELD_CANDIDATES: Dict[str, List[str]] = {
    "question_id": ["questionId", "question_id", "qid"],
    "question": ["question"],
    "doc_id": ["doc_id", "docId", "document_id"],
    "page_ids": ["page_ids", "pageIds"],
    "answer_page_idx": ["answer_page_idx", "answer_page", "answerPageIdx"],
    "answers": ["answers", "answer"],
    "question_types": ["question_types", "question_type", "questionTypes"],
}
REQUIRED = ["question_id", "question", "doc_id", "page_ids", "answer_page_idx", "answers"]
PAGE_ID_RE = re.compile(r"_p(\d+)$")


def parse_listish(value: Any) -> List[Any]:
    """Parse a list stored either natively or as a string like "['a', 'b']"."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("["):
            try:
                return list(json.loads(s))
            except json.JSONDecodeError:
                return list(ast.literal_eval(s))
        return [s] if s else []
    return [value]


def page_number(page_id: str) -> int:
    """'{doc_id}_p{k}' (k is 0-based) -> original 1-based page number k+1."""
    m = PAGE_ID_RE.search(page_id)
    if not m:
        raise ValueError(f"unexpected page id: {page_id!r}")
    return int(m.group(1)) + 1


def resolve_fields(feature_names: List[str]) -> Dict[str, Optional[str]]:
    """Map target field -> real column name (None if absent). Fails if a required field is missing."""
    mapping: Dict[str, Optional[str]] = {}
    for target, cands in FIELD_CANDIDATES.items():
        mapping[target] = next((c for c in cands if c in feature_names), None)
    missing = [t for t in REQUIRED if mapping[t] is None]
    if missing:
        raise SystemExit(f"Required fields not found in dataset features: {missing}. Features: {feature_names}")
    return mapping


def image_columns(features: Any) -> List[str]:
    """Return Image-typed columns sorted by their numeric suffix (image_1, image_2, ...)."""
    from datasets import Image

    cols = [name for name, feat in features.items() if isinstance(feat, Image)]
    return sorted(cols, key=lambda n: int(re.search(r"(\d+)$", n).group(1)) if re.search(r"(\d+)$", n) else 0)


def save_png(img_field: Dict[str, Any], out_path: Path) -> List[int]:
    """Write an undecoded HF image ({'bytes', 'path'}) as PNG. Returns [width, height]."""
    from PIL import Image as PILImage

    data = img_field.get("bytes")
    if data is None and img_field.get("path"):
        data = Path(img_field["path"]).read_bytes()
    with PILImage.open(io.BytesIO(data)) as im:
        size = list(im.size)
        if im.format == "PNG":
            out_path.write_bytes(data)
        else:
            im.save(out_path, format="PNG")
    return size


def load_index(args: argparse.Namespace, mapping: Dict[str, Optional[str]], index_path: Path) -> List[Dict[str, Any]]:
    """Pass 1: metadata of every question in the split (text columns only), cached on disk."""
    if index_path.exists():
        rows = list(iter_jsonl(index_path))
        print(f"Metadata pass: using cache {index_path.name} ({len(rows)} rows)")
        return rows
    from datasets import load_dataset

    cols = [c for c in mapping.values() if c]
    ds = load_dataset(args.dataset, split=args.split, streaming=True, columns=cols)
    rows = []
    for row_idx, r in enumerate(tqdm(ds, desc="metadata pass (no images)", unit="row")):
        rec = {"row": row_idx, **{t: r[s] for t, s in mapping.items() if s}}
        rec["page_ids"] = [str(p) for p in parse_listish(rec["page_ids"])]
        rec["answers"] = [str(a) for a in parse_listish(rec["answers"])]
        rec["answer_page_idx"] = int(rec["answer_page_idx"])
        rec["question_id"] = str(rec["question_id"])
        rec["doc_id"] = str(rec["doc_id"])
        if "question_types" in rec:
            rec["question_types"] = parse_listish(rec["question_types"])
        rows.append(rec)
    write_jsonl(index_path, rows)
    print(f"Metadata pass: {len(rows)} rows cached to {index_path.name}")
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=cfg["dataset"]["subset_size"], help="number of documents")
    ap.add_argument("--dataset", default=cfg["dataset"]["hf_name"])
    ap.add_argument("--split", default=cfg["dataset"]["split"])
    args = ap.parse_args(argv)

    from datasets import Image, load_dataset

    out_dir: Path = cfg["paths"]["mpdocvqa_dir"]
    img_root: Path = cfg["paths"]["mpdocvqa_images_dir"]
    q_path: Path = cfg["paths"]["questions_file"]
    img_root.mkdir(parents=True, exist_ok=True)

    # 0) Inspect the real features first.
    print(f"Loading {args.dataset} split={args.split} (streaming=True)")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    features = ds.features
    print("\n=== Dataset features (real) ===")
    for name, feat in features.items():
        print(f"  {name}: {feat}")
    mapping = resolve_fields(list(features.keys()))
    img_cols = image_columns(features)
    print("\n=== Field mapping (target <- source) ===")
    for t, s in mapping.items():
        print(f"  {t:<16} <- {s if s else '(not in dataset)'}")
    print(f"  page images      <- {img_cols[0]} .. {img_cols[-1]} ({len(img_cols)} columns), aligned with page_ids")

    # 1) Metadata pass + document selection.
    index = load_index(args, mapping, out_dir / f"{args.split}_index.jsonl")
    doc_order = list(dict.fromkeys(r["doc_id"] for r in index))
    selected = doc_order[: args.limit]
    selected_set: Set[str] = set(selected)
    stop_row = next(r["row"] for r in index if r["doc_id"] == selected[-1])
    print(f"\nSplit has {len(index)} questions over {len(doc_order)} documents.")
    print(f"Selected first {len(selected)} documents; image pass streams rows 0..{stop_row}.")

    # 2) Image pass: stream full rows up to stop_row; save pages not yet on disk.
    pages: Dict[str, Dict[str, List[int]]] = {d: {} for d in selected}  # doc -> page_id -> size
    for d in selected:
        meta_path = img_root / d / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            pages[d] = {p: s for p, s in zip(meta["page_ids"], meta["page_sizes"])}

    needed: Dict[str, Set[str]] = {d: set() for d in selected}
    for r in index:
        if r["row"] > stop_row:
            break
        if r["doc_id"] in selected_set:
            needed[r["doc_id"]] |= set(r["page_ids"])
    missing_total = sum(len(needed[d] - set(pages[d])) for d in selected)
    already = sum(1 for d in selected if needed[d] <= set(pages[d]))
    print(f"Resume: {already}/{len(selected)} docs already complete on disk; {missing_total} pages to download.")

    new_pages = 0
    if missing_total:
        for col in img_cols:  # keep raw bytes; avoids decode/re-encode of PNGs
            ds = ds.cast_column(col, Image(decode=False))
        for row_idx, row in enumerate(tqdm(ds, total=stop_row + 1, desc="image pass", unit="row")):
            if row_idx > stop_row:
                break
            doc_id = str(row[mapping["doc_id"]])
            if doc_id not in selected_set:
                continue
            page_ids = [str(p) for p in parse_listish(row[mapping["page_ids"]])]
            assert page_ids == index[row_idx]["page_ids"], f"row order mismatch at row {row_idx}"
            todo = [(i, p) for i, p in enumerate(page_ids) if p not in pages[doc_id]]
            if not todo:
                continue
            doc_dir = img_root / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)
            for i, pid in todo:
                field = row.get(img_cols[i])
                if field is None:
                    print(f"  WARN {doc_id}: {img_cols[i]} empty for page {pid}")
                    continue
                pages[doc_id][pid] = save_png(field, doc_dir / f"page_{page_number(pid)}.png")
                new_pages += 1
            ordered = sorted(pages[doc_id], key=page_number)
            meta = {
                "doc_id": doc_id,
                "page_ids": ordered,
                "page_numbers": [page_number(p) for p in ordered],
                "page_sizes": [pages[doc_id][p] for p in ordered],
                "num_pages": len(ordered),
                "source": f"{args.dataset}:{args.split}",
            }
            (doc_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 3) Questions: every question of a selected doc whose answer page is saved.
    records: List[Dict[str, Any]] = []
    dropped = Counter()
    for r in index:
        if r["doc_id"] not in selected_set:
            continue
        ans_pid = r["page_ids"][r["answer_page_idx"]]
        if ans_pid not in pages[r["doc_id"]]:
            dropped["answer page not saved"] += 1
            continue
        rec: Dict[str, Any] = {
            "question_id": r["question_id"],
            "question": r["question"],
            "doc_id": r["doc_id"],
            "page_ids": r["page_ids"],              # this question's original window
            "answer_page_idx": r["answer_page_idx"],  # 0-based index into page_ids (as in the dataset)
            "answer_page_id": ans_pid,
            "answer_page": page_number(ans_pid),    # 1-based page number = page_{n}.png
            "answers": r["answers"],
        }
        if "question_types" in r:
            rec["question_types"] = r["question_types"]
        records.append(rec)
    write_jsonl(q_path, records)

    total_pages = sum(len(p) for p in pages.values())
    print(f"\nNew pages saved: {new_pages}. Docs: {sum(1 for p in pages.values() if p)} | pages on disk: {total_pages}")
    print(f"Questions written: {len(records)} -> {q_path.relative_to(cfg['repo_root'])}")
    print(f"Questions dropped: {dict(dropped) or 0}")
    if not mapping["question_types"]:
        print("Note: dataset has no question-type field; question_types not written.")
    if records:
        print("First question record:", json.dumps(records[0], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
