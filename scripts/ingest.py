"""Parse -> serialize -> graph for MP-DocVQA documents or demo PDFs.

Appends to data/processed/components.jsonl, skipping documents already processed,
saves graphs to data/processed/graphs/{doc_id}.pkl and a manifest to data/processed/docs/{doc_id}.json.

Usage:
  .venv/Scripts/python -m scripts.ingest --source mpdocvqa --limit 3
  .venv/Scripts/python -m scripts.ingest --source mpdocvqa --docs snbx0223 zxfk0226
  .venv/Scripts/python -m scripts.ingest --source pdf
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

from tqdm import tqdm

from docscout.config import get_config
from docscout.graph.builder import build_graph, save_graph
from docscout.ingestion.parser import build_converter, parse_mpdocvqa, parse_pdf, safe_doc_id
from docscout.ingestion.serializer import serialize_components
from docscout.schema import COMPONENT_TYPES, iter_jsonl, read_components, write_components


class _OcrLogFilter(logging.Filter):
    """Surface Docling's OCR-engine selection messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "OCR" in msg or "ocr" in record.name.lower()


def processed_doc_ids(path: Path) -> Set[str]:
    return {r["doc_id"] for r in iter_jsonl(path)} if path.exists() else set()


def drop_docs(path: Path, doc_ids: Set[str]) -> int:
    """Remove the given docs' lines from components.jsonl. Returns the number of lines removed."""
    if not path.exists():
        return 0
    comps = read_components(path)
    keep = [c for c in comps if c.doc_id not in doc_ids]
    if len(keep) != len(comps):
        write_components(path, keep)
    return len(comps) - len(keep)


def clean_doc_outputs(doc_id: str, cfg: Dict) -> Dict[str, int]:
    """Delete a document's derived files (tables, figures, rendered pages, graph, manifest, debug overlays).

    Doc ids contain no '_', so the glob '{doc_id}_*' only matches this document's component files.
    """
    p = cfg["paths"]
    removed = {"tables": 0, "figures": 0, "pages": 0, "other": 0}
    for key, folder in (("tables", p["tables_dir"]), ("figures", p["figures_dir"])):
        for f in folder.glob(f"{doc_id}_*"):
            f.unlink()
            removed[key] += 1
    for folder_key, count_key in (("pages_dir", "pages"), ("debug_dir", "other")):
        d = p[folder_key] / doc_id
        if d.is_dir():
            for f in d.iterdir():
                f.unlink()
                removed[count_key] += 1
            d.rmdir()
    for f in (p["graphs_dir"] / f"{doc_id}.pkl", p["docs_dir"] / f"{doc_id}.json"):
        if f.exists():
            f.unlink()
            removed["other"] += 1
    return removed


def main(argv: Optional[List[str]] = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["mpdocvqa", "pdf"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="max documents to process (0 = all)")
    ap.add_argument("--docs", nargs="*", help="explicit doc ids (mpdocvqa) or PDF file names (pdf)")
    ap.add_argument("--force", action="store_true", help="re-process documents already in components.jsonl")
    ap.add_argument("--reserialize", action="store_true",
                    help="no parsing: recompute text_form and graphs for components already in components.jsonl")
    args = ap.parse_args(argv)

    comp_path: Path = cfg["paths"]["components_file"]
    if args.reserialize:
        comps = read_components(comp_path)
        by_doc: Dict[str, list] = {}
        for c in comps:
            by_doc.setdefault(c.doc_id, []).append(c)
        for doc_id, dcomps in by_doc.items():
            serialize_components(dcomps)
            save_graph(build_graph(dcomps), doc_id)
        write_components(comp_path, [c for d in by_doc.values() for c in d])
        print(f"Re-serialized {len(comps)} components and rebuilt {len(by_doc)} graphs.")
        return 0

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ocr_handler = logging.StreamHandler(sys.stdout)
    ocr_handler.addFilter(_OcrLogFilter())
    ocr_handler.setFormatter(logging.Formatter("[docling] %(message)s"))
    dl = logging.getLogger("docling")
    dl.setLevel(logging.INFO)
    dl.addHandler(ocr_handler)
    dl.propagate = False

    comp_path.parent.mkdir(parents=True, exist_ok=True)

    # Work list: (doc_id, input path or None)
    if args.source == "mpdocvqa":
        img_root: Path = cfg["paths"]["mpdocvqa_images_dir"]
        if args.docs:
            work = [(d, None) for d in args.docs]
        else:
            q_path: Path = cfg["paths"]["questions_file"]
            order = list(dict.fromkeys(r["doc_id"] for r in iter_jsonl(q_path)))
            work = [(d, None) for d in order if (img_root / d / "meta.json").exists()]
    else:
        pdf_dir: Path = cfg["paths"]["pdfs_dir"]
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if args.docs:
            pdfs = [p for p in pdfs if p.name in args.docs or p.stem in args.docs]
        if not pdfs:
            print(f"No PDFs found in {pdf_dir}. Add demo PDFs there first.")
            return 1
        work = [(safe_doc_id(p.stem), p) for p in pdfs]

    done = processed_doc_ids(comp_path)
    if args.force:
        done -= {d for d, _ in work}  # old lines/files are removed per document just before re-parsing
    todo = [(d, p) for d, p in work if d not in done]
    skipped = len(work) - len(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(work)} candidate docs, {skipped} already processed, processing {len(todo)}.")
    if not todo:
        return 0

    t0 = time.perf_counter()
    converter = build_converter(cfg)
    print(f"Converter ready in {time.perf_counter() - t0:.1f}s (models load lazily on first document); "
          f"OCR: RapidOCR, backend={cfg['parsing'].get('ocr_backend')}")

    totals: Counter = Counter()
    total_pages = 0
    total_secs = 0.0
    rows: List[Dict] = []
    for doc_id, pdf in tqdm(todo, desc="ingest", unit="doc"):
        # Remove any old output of this document first (forced re-parse, or leftovers of a crashed run).
        n_lines = drop_docs(comp_path, {doc_id})
        removed = clean_doc_outputs(doc_id, cfg)
        if n_lines or any(removed.values()):
            tqdm.write(f"{doc_id}: removed {n_lines} old components.jsonl lines, old files {removed}")
        t = time.perf_counter()
        if args.source == "mpdocvqa":
            comps, stats = parse_mpdocvqa(doc_id, converter, cfg)
        else:
            comps, stats = parse_pdf(pdf, doc_id, converter, cfg)
        t_parse = time.perf_counter() - t
        serialize_components(comps)
        G = build_graph(comps)
        if comps:
            save_graph(G, doc_id)
            write_components(comp_path, comps, append=True)
        secs = time.perf_counter() - t
        n_pages = stats.get("pages_total", 0)
        counts = Counter(c.type for c in comps)
        totals.update(counts)
        total_pages += n_pages
        total_secs += secs
        row = {"doc_id": doc_id, "pages": n_pages, "secs": round(secs, 1), "parse_secs": round(t_parse, 1),
               "sec_per_page": round(secs / max(n_pages, 1), 2), "status": stats.get("status"),
               "pages_with_components": stats.get("pages_with_components"),
               "pages_without_components": stats.get("pages_without_components"),
               "errors": stats.get("errors"), "item_errors": stats.get("item_errors", 0),
               "captions_docling": stats.get("captions_linked_docling", 0),
               "captions_nearest": stats.get("captions_linked_nearest", 0),
               "graph": f"{G.number_of_nodes()} nodes / {G.number_of_edges()} edges",
               **{t_: counts.get(t_, 0) for t_ in COMPONENT_TYPES}}
        rows.append(row)
        tqdm.write(json.dumps(row))

    print("\n=== Per document ===")
    hdr = f"{'doc_id':<22}{'pages':>6}{'para':>6}{'head':>6}{'table':>6}{'fig':>6}{'cap':>6}{'secs':>8}{'s/page':>8}  status"
    print(hdr)
    for r in rows:
        print(f"{r['doc_id']:<22}{r['pages']:>6}{r['paragraph']:>6}{r['heading']:>6}{r['table']:>6}{r['figure']:>6}"
              f"{r['caption']:>6}{r['secs']:>8}{r['sec_per_page']:>8}  {r['status']}")
    print("\n=== Totals ===")
    print(f"docs: {len(rows)} | pages: {total_pages} | components: {sum(totals.values())} | "
          + " ".join(f"{t_}={totals.get(t_, 0)}" for t_ in COMPONENT_TYPES))
    print(f"time: {total_secs:.1f}s total, {total_secs / max(total_pages, 1):.2f} s/page "
          f"(includes first-document model loading)")
    print(f"components appended to {comp_path.relative_to(cfg['repo_root'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
