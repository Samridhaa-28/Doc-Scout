"""Parse -> serialize -> graph for MP-DocVQA documents or demo PDFs.

Appends to data/processed/components.jsonl, skipping documents already processed,
saves graphs to data/processed/graphs/{doc_id}.pkl and a manifest to data/processed/docs/{doc_id}.json.

Isolation (default): documents are parsed in a child worker process. If the worker dies (for example a
native crash inside a PDF library) or exceeds --doc-timeout, the parent records the document in
data/processed/ingest_failures.jsonl, starts a fresh worker and continues. The parent is the only
process that writes components.jsonl. Failed documents are skipped on later runs (see --retry-failed).

Command line (scripts/ingest.py):
  .venv/Scripts/python -m scripts.ingest --source mpdocvqa --limit 3
  .venv/Scripts/python -m scripts.ingest --source mpdocvqa --docs snbx0223 zxfk0226
  .venv/Scripts/python -m scripts.ingest --source mpdocvqa --docs-file data/raw/mpdocvqa/eval_docs.txt --max-minutes 12
  .venv/Scripts/python -m scripts.ingest --source pdf
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import queue
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from docscout.config import get_config
from docscout.graph.builder import build_graph, save_graph
from docscout.ingestion.parser import build_converter, parse_mpdocvqa, parse_pdf, pdf_backend_for, safe_doc_id
from docscout.ingestion.serializer import serialize_components
from docscout.schema import COMPONENT_TYPES, Component, iter_jsonl, read_components, write_components, write_jsonl


class _OcrLogFilter(logging.Filter):
    """Surface Docling's OCR-engine selection messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "OCR" in msg or "ocr" in record.name.lower()


def _setup_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_OcrLogFilter())
    handler.setFormatter(logging.Formatter("[docling] %(message)s"))
    dl = logging.getLogger("docling")
    dl.setLevel(logging.INFO)
    dl.addHandler(handler)
    dl.propagate = False


def processed_doc_ids(path: Path) -> Set[str]:
    return {r["doc_id"] for r in iter_jsonl(path)} if path.exists() else set()


def failed_doc_ids(cfg: Dict, backend: Optional[str] = None) -> Set[str]:
    """Documents recorded in ingest_failures.jsonl (only those that failed with `backend`, if given)."""
    path = failures_path(cfg)
    if not path.exists():
        return set()
    return {r["doc_id"] for r in iter_jsonl(path) if backend is None or r.get("pdf_backend") == backend}


def failures_path(cfg: Dict) -> Path:
    return cfg["paths"]["processed_dir"] / "ingest_failures.jsonl"


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


# ------------------------------------------------------------------ parsing (in-process or in a worker)

def parse_one(source: str, doc_id: str, pdf: Optional[str], converter, cfg: Dict) -> Tuple[List[Component], Dict, float]:
    """Parse + serialize one document. Returns (components, stats, parse seconds)."""
    t = time.perf_counter()
    if source == "mpdocvqa":
        comps, stats = parse_mpdocvqa(doc_id, converter, cfg)
    else:
        comps, stats = parse_pdf(Path(pdf), doc_id, converter, cfg)
    t_parse = time.perf_counter() - t
    serialize_components(comps)
    return comps, stats, t_parse


def _worker_main(source: str, backend: str, task_q, result_q) -> None:
    """Child process: build the converter once, then parse documents from task_q until None."""
    _setup_logging()
    cfg = get_config()
    converter = build_converter(cfg, backend=backend)
    result_q.put(("ready", None, None))
    while True:
        task = task_q.get()
        if task is None:
            return
        doc_id, pdf = task
        try:
            comps, stats, t_parse = parse_one(source, doc_id, pdf, converter, cfg)
            result_q.put(("ok", doc_id, ([c.to_dict() for c in comps], stats, t_parse)))
        except Exception:  # noqa: BLE001 - report Python errors to the parent
            result_q.put(("error", doc_id, traceback.format_exc()))


class Worker:
    """A spawned child process that parses documents; restarted after a crash."""

    def __init__(self, source: str, backend: str, target=None) -> None:
        self.source, self.backend = source, backend
        self.target = target or _worker_main  # injectable for tests (must be importable: spawn)
        self.ctx = mp.get_context("spawn")
        self.proc = None
        self.start()

    def start(self) -> None:
        self.task_q, self.result_q = self.ctx.Queue(), self.ctx.Queue()
        self.proc = self.ctx.Process(target=self.target, args=(self.source, self.backend, self.task_q, self.result_q),
                                     daemon=True)
        self.proc.start()

    def run(self, doc_id: str, pdf: Optional[str], timeout_s: float) -> Tuple[str, Any]:
        """Returns ('ok', (comps, stats, t_parse)) | ('error', traceback) | ('crash', exitcode) | ('timeout', secs)."""
        self.task_q.put((doc_id, pdf))
        t0 = time.perf_counter()
        while True:
            try:
                kind, did, payload = self.result_q.get(timeout=2)
            except queue.Empty:
                if not self.proc.is_alive():
                    code = self.proc.exitcode
                    self.start()
                    return "crash", code
                if time.perf_counter() - t0 > timeout_s:
                    self.proc.terminate()
                    self.proc.join(10)
                    self.start()
                    return "timeout", round(time.perf_counter() - t0)
                continue
            if kind == "ready":
                continue
            if did != doc_id:
                continue
            if kind == "ok":
                dicts, stats, t_parse = payload
                return "ok", ([Component.from_dict(d) for d in dicts], stats, t_parse)
            return "error", payload

    def close(self) -> None:
        if self.proc is not None and self.proc.is_alive():
            self.task_q.put(None)
            self.proc.join(30)
            if self.proc.is_alive():
                self.proc.terminate()


def run_ingest(source: str, docs: Optional[List[str]] = None, docs_file: Optional[Path] = None, limit: int = 0,
               max_minutes: float = 0, stop_on_error: bool = False, force: bool = False, retry_failed: bool = False,
               isolate: bool = True, doc_timeout: float = 30, pdf_backend: Optional[str] = None,
               reserialize: bool = False, verbose: bool = False, cfg: Optional[Dict] = None) -> int:
    """Ingest documents (see module docstring). Returns 0 on success, 3 if stopped by stop_on_error, 1 on bad input."""
    cfg = cfg or get_config()
    args = SimpleNamespace(source=source, docs=list(docs) if docs else None, docs_file=docs_file, limit=limit,
                           max_minutes=max_minutes, stop_on_error=stop_on_error, force=force,
                           retry_failed=retry_failed, no_isolate=not isolate, doc_timeout=doc_timeout,
                           pdf_backend=pdf_backend, reserialize=reserialize)

    if source == "pdf" and cfg.get("store", "eval") == "eval":
        print("Demo PDFs go to their own store so the MP-DocVQA eval corpus (and its BM25 statistics) is unchanged: "
              "rerun with --store demo.")
        return 1
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

    _setup_logging()
    comp_path.parent.mkdir(parents=True, exist_ok=True)

    # Work list: (doc_id, input path or None)
    if args.docs_file:
        file_docs = [ln.strip() for ln in args.docs_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        args.docs = (args.docs or []) + file_docs
    if args.source == "mpdocvqa":
        img_root: Path = cfg["paths"]["mpdocvqa_images_dir"]
        if args.docs:
            work: List[Tuple[str, Optional[str]]] = [(d, None) for d in args.docs]
        else:
            order = list(dict.fromkeys(r["doc_id"] for r in iter_jsonl(cfg["paths"]["questions_file"])))
            work = [(d, None) for d in order if (img_root / d / "meta.json").exists()]
    else:
        pdfs = sorted(cfg["paths"]["pdfs_dir"].glob("*.pdf"))
        if args.docs:
            pdfs = [p for p in pdfs if p.name in args.docs or p.stem in args.docs]
        if not pdfs:
            print(f"No PDFs found in {cfg['paths']['pdfs_dir']}. Add demo PDFs there first.")
            return 1
        work = [(safe_doc_id(p.stem), str(p)) for p in pdfs]

    done = processed_doc_ids(comp_path)
    if args.force:
        done -= {d for d, _ in work}  # old lines/files are removed per document just before re-parsing
    backend = args.pdf_backend or pdf_backend_for(args.source, cfg)
    failed_before = set() if (args.retry_failed or args.force) else failed_doc_ids(cfg, backend)
    todo = [(d, p) for d, p in work if d not in done and d not in failed_before]
    skipped_failed = [d for d, _ in work if d in failed_before and d not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(work)} candidate docs, {len(work) - len(todo) - len(skipped_failed)} already processed, "
          f"{len(skipped_failed)} skipped as previously failed {skipped_failed or ''}, processing {len(todo)}.")
    print(f"PDF backend: {backend} | OCR: RapidOCR, backend={cfg['parsing'].get('ocr_backend')} | "
          f"isolation: {'off' if args.no_isolate else 'child process'}")
    if not todo:
        return 0

    t0 = time.perf_counter()
    worker = None if args.no_isolate else Worker(args.source, backend)
    converter = build_converter(cfg, backend=backend) if args.no_isolate else None

    totals: Counter = Counter()
    total_pages = 0
    total_secs = 0.0
    rows: List[Dict] = []
    failures: List[Dict] = []
    stop = False
    budget_hit = False
    try:
        for doc_id, pdf in tqdm(todo, desc="ingest", unit="doc"):
            if args.max_minutes and (time.perf_counter() - t0) / 60 >= args.max_minutes:
                budget_hit = True
                break
            # Remove any old output of this document first (forced re-parse, or leftovers of a crashed run).
            n_lines = drop_docs(comp_path, {doc_id})
            removed = clean_doc_outputs(doc_id, cfg)
            if n_lines or any(removed.values()):
                tqdm.write(f"{doc_id}: removed {n_lines} old components.jsonl lines, old files {removed}")
            t = time.perf_counter()
            if worker is None:
                comps, stats, t_parse = parse_one(args.source, doc_id, pdf, converter, cfg)
                kind = "ok"
            else:
                kind, payload = worker.run(doc_id, pdf, args.doc_timeout * 60)
                if kind == "ok":
                    comps, stats, t_parse = payload
            secs = time.perf_counter() - t
            if kind != "ok":
                fail = {"doc_id": doc_id, "source": args.source, "pdf_backend": backend, "reason": kind,
                        "detail": str(payload)[-2000:], "secs": round(secs, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                write_jsonl(failures_path(cfg), [fail], append=True)
                clean_doc_outputs(doc_id, cfg)
                failures.append(fail)
                tqdm.write(f"FAILED {doc_id}: {kind} ({str(payload)[-300:]}) after {secs:.0f}s -> recorded in "
                           f"{failures_path(cfg).name}; continuing with a fresh worker")
                if args.stop_on_error:
                    stop = True
                    break
                continue
            G = build_graph(comps)
            if comps:
                save_graph(G, doc_id)
                write_components(comp_path, comps, append=True)
            n_pages = stats.get("pages_total", 0)
            counts = Counter(c.type for c in comps)
            totals.update(counts)
            total_pages += n_pages
            total_secs += secs
            row = {"doc_id": doc_id, "pages": n_pages, "secs": round(secs, 1), "parse_secs": round(t_parse, 1),
                   "sec_per_page": round(secs / max(n_pages, 1), 2), "status": stats.get("status"),
                   "pdf_backend": stats.get("pdf_backend"),
                   "pages_with_components": stats.get("pages_with_components"),
                   "pages_without_components": stats.get("pages_without_components"),
                   "errors": stats.get("errors"), "item_errors": stats.get("item_errors", 0),
                   "captions_docling": stats.get("captions_linked_docling", 0),
                   "captions_nearest": stats.get("captions_linked_nearest", 0),
                   "graph": f"{G.number_of_nodes()} nodes / {G.number_of_edges()} edges",
                   **{t_: counts.get(t_, 0) for t_ in COMPONENT_TYPES}}
            rows.append(row)
            tqdm.write(json.dumps(row) if verbose else
                       f"{doc_id}: {n_pages} pages, {len(comps)} components, {secs:.0f}s, {row['status']}")
            sys.stdout.flush()
            if args.stop_on_error and (stats.get("status") != "success" or stats.get("item_errors", 0) or not comps):
                stop = True
                break
    finally:
        if worker is not None:
            worker.close()

    if verbose:
        print("\n=== Per document ===")
        print(f"{'doc_id':<22}{'pages':>6}{'para':>6}{'head':>6}{'table':>6}{'fig':>6}{'cap':>6}{'secs':>8}{'s/page':>8}  status")
        for r in rows:
            print(f"{r['doc_id']:<22}{r['pages']:>6}{r['paragraph']:>6}{r['heading']:>6}{r['table']:>6}{r['figure']:>6}"
                  f"{r['caption']:>6}{r['secs']:>8}{r['sec_per_page']:>8}  {r['status']}")
    print("\n=== Totals ===")
    print(f"docs: {len(rows)} | pages: {total_pages} | components: {sum(totals.values())} | "
          + " ".join(f"{t_}={totals.get(t_, 0)}" for t_ in COMPONENT_TYPES))
    print(f"time: {total_secs:.1f}s total, {total_secs / max(total_pages, 1):.2f} s/page")
    print(f"failures this run: {len(failures)} {[ (f['doc_id'], f['reason']) for f in failures ]}")
    done_now = processed_doc_ids(comp_path) & {d for d, _ in work}
    print(f"progress: {len(done_now)}/{len(work)} documents done | this run: {len(rows)} docs in "
          f"{(time.perf_counter() - t0) / 60:.1f} min" + (" | time budget reached, rerun to resume" if budget_hit else ""))
    if stop:
        print("STOPPED (--stop-on-error): see the failed document above")
        return 3
    return 0
