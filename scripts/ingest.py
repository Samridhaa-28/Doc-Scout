"""Parse -> serialize -> graph for MP-DocVQA documents or demo PDFs. Logic: docscout.ingestion.runner.

Documents are parsed in an isolated child process (crash-safe); failures go to data/processed/ingest_failures.jsonl.

Usage:
  .venv/Scripts/python -m scripts.ingest --source mpdocvqa --docs-file data/raw/mpdocvqa/eval_docs.txt --max-minutes 12
  .venv/Scripts/python -m scripts.ingest --source mpdocvqa --docs snbx0223 zxfk0226
  .venv/Scripts/python -m scripts.ingest --source pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from docscout.ingestion.runner import run_ingest


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["mpdocvqa", "pdf"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="max documents to process (0 = all)")
    ap.add_argument("--docs", nargs="*", help="explicit doc ids (mpdocvqa) or PDF file names (pdf)")
    ap.add_argument("--docs-file", type=Path, help="text file with one doc id per line (e.g. eval_docs.txt)")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="time budget for this run: no new document is started after it (0 = no limit); rerun to resume")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="stop at the first document that fails, crashes, or yields no components")
    ap.add_argument("--force", action="store_true", help="re-process documents already in components.jsonl")
    ap.add_argument("--retry-failed", action="store_true", help="also retry documents listed in ingest_failures.jsonl")
    ap.add_argument("--no-isolate", action="store_true", help="parse in this process (no crash isolation)")
    ap.add_argument("--doc-timeout", type=float, default=30, help="minutes before a document is abandoned (isolated mode)")
    ap.add_argument("--pdf-backend", choices=["pypdfium2", "docling_parse"], default=None,
                    help="override config parsing.pdf_backend for this run")
    ap.add_argument("--reserialize", action="store_true",
                    help="no parsing: recompute text_form and graphs for components already in components.jsonl")
    ap.add_argument("--verbose", action="store_true", help="print the full JSON row per document")
    a = ap.parse_args(argv)
    return run_ingest(a.source, docs=a.docs, docs_file=a.docs_file, limit=a.limit, max_minutes=a.max_minutes,
                      stop_on_error=a.stop_on_error, force=a.force, retry_failed=a.retry_failed,
                      isolate=not a.no_isolate, doc_timeout=a.doc_timeout, pdf_backend=a.pdf_backend,
                      reserialize=a.reserialize, verbose=a.verbose)


if __name__ == "__main__":
    sys.exit(main())
