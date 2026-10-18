# DocScout-X progress

## Starting state (2026-10-17)
- Git repo on branch `main` with no commits yet. Files present: `CLAUDE.md`, `docs/PHASES.md`.
- `.venv` (Python 3.10.0, pip 21.2.3, nothing else installed) existed in the repo root. Its files had been staged in the git index. Before P1 they were unstaged with `git rm -r --cached .venv`; no files were deleted from disk.
- No data, no code.

<!-- Phase summaries are appended below as each phase is approved. -->

## P1 (approved, 2026-10-17): setup, data, shared schema
Full report: [docs/reports/P1_REPORT.md](reports/P1_REPORT.md)

### What exists now
- Repo skeleton (src layout, package `docscout`, an `__init__.py` in every package of the CLAUDE.md tree), `pyproject.toml`, `requirements.txt`, `requirements.lock.txt` (its first line adds the PyTorch CPU index), `.gitignore`, `.env.example`, `README.md` (Windows setup).
- `.venv`: Python 3.10.0 with torch 2.14.1+cpu and every library from CLAUDE.md. `scripts.check_env` prints "all good".
- `configs/config.yaml`: paths, dataset, parsing, models, branches, retrieval, fusion (branch weights per query type), classifier, feedback, llm, eval, api.
- `src/docscout/config.py`, `src/docscout/schema.py`.
- `scripts/check_env.py`, `scripts/download_data.py`, `scripts/explore_data.py`.
- Tests: `test_schema.py`, `test_config.py`, `test_download_utils.py` (13 passing).

### Key interfaces
- `docscout.config`:
  - `get_config() -> dict` returns a deep copy; paths are absolute `Path` objects.
  - `get_path(key)`; `get_secret(name)` (reads `.env`); `REPO_ROOT`.
- `docscout.schema`:
  - `Component(id, doc_id, page, type, text, text_form, bbox, page_width, page_height, order, section_id, table_path, image_path, meta)`
  - `Evidence(component, scores, context, localized_cells, why, meta)`
  - `QueryResult(query, query_type, answer, evidence, meta)`
  - `make_component_id(doc_id, page, type, n) -> "{doc_id}_p{page}_{type}_{n}"`
  - `write_components` / `read_components` / `write_jsonl` / `read_jsonl` / `iter_jsonl`
- MP-DocVQA on disk:
  - `data/raw/mpdocvqa/images/{doc_id}/page_{n}.png`, where `n` is the original 1-based page number.
  - `images/{doc_id}/meta.json`: `{doc_id, page_ids, page_numbers, page_sizes, num_pages, source}`.
  - `data/raw/mpdocvqa/questions.jsonl`: `{question_id, question, doc_id, page_ids, answer_page_idx (0-based into page_ids), answer_page_id, answer_page (1-based), answers}`.
  - `data/raw/mpdocvqa/val_index.jsonl`: metadata cache for the whole split.

### Data counts
- At approval: 5 documents, 16 pages, 43 questions (`download_data --limit 5`).
- Projected for `--limit 150` (from the metadata cache): 680 of 5,187 rows streamed, 1,060 pages, 947 of 1,178 questions kept, 7.07 pages per document, 41 single-page documents.
- The split has 5,187 questions over 927 documents.

### Decisions
- **Dataset:** `lmms-lab/MP-DocVQA`, `val` split, first-N-documents selection (by first appearance in the stream). `answer_page` = original 1-based page number. Page numbers can have gaps, so always read `meta.json`.
- **Single-page documents:** keep them. In P4, report page-level metrics on all questions AND on questions from documents with >=2 saved pages; the >=2-pages figure is the main page-level number. Component metrics use all questions.
- **Retrieval scope:** search the whole saved document, not the question's `page_ids` window.
- **Question types:** join with `lmms-lab/DocVQA` val on `questionId` in P5.
- **Avast SSL workaround** (`SSL_CERT_FILE` pointing at a certifi + Windows-roots bundle) is local to this machine only; nothing in the repo depends on it.

## P2 (approved for MP-DocVQA, 2026-10-17): parsing, serialization, graph
Full report: [docs/reports/P2_REPORT.md](reports/P2_REPORT.md)

### What exists now
- `src/docscout/ingestion/parser.py`: Docling (layout + TableFormer + **RapidOCR on the onnxruntime backend**, config `parsing.ocr_backend`). MP-DocVQA PNGs are joined into one PDF (`data/processed/mpdocvqa_pdfs/`). Docling items map to paragraph / heading / table / figure / caption. Tables are saved to parquet, figures are cropped to PNG. Captions are linked (Docling refs first, nearest caption as fallback; "Page N" markers relabelled). Each document gets a page manifest.
- `src/docscout/ingestion/serializer.py`: `text_form` per type (table = title above + caption + columns + row headers + first 3 rows; figure = caption + nearest paragraph + inner text), truncated to about 512 tokens, never empty.
- `src/docscout/graph/builder.py`: DiGraph Document → Page → Section(heading, page) → Block. Edge `rel` values: `parent_child`, `sibling`, `caption_of`. Saved as pickle, with helpers.
- `scripts/ingest.py` (`--source mpdocvqa|pdf`, `--limit`, `--docs`, `--force` with per-document cleanup, `--reserialize`), `scripts/visualize_components.py`, `scripts/check_components.py`.
- Tests: 28 passing.

### Key interfaces
- `components.jsonl`: one `Component` per line, unique ids.
  - `bbox` is top-left, in pixels of the displayed page image (MP-DocVQA: original PNG; PDF: rendered at `parsing.image_dpi`).
  - `meta`: `docling_label`, `docling_label_original`, `regions`, `caption_id` / `caption_of`, `n_rows` / `n_cols`, `inner_text`.
- `parse_mpdocvqa(doc_id)` and `parse_pdf(path)` return `(components, stats)`. `serialize_components(comps)`; `build_graph(comps)`; `load_graph(doc_id)`, `parent_section`, `siblings`, `caption_of`, `linked_component`, `path_to_root`; `load_manifest(doc_id)`.
- Files:
  - `data/processed/{tables,figures}/{component_id}.{parquet,png}`
  - `data/processed/graphs/{doc_id}.pkl`
  - `data/processed/docs/{doc_id}.json`

### Data state
- **Ingested: 3 MP-DocVQA documents** (`jjvg0227`, `yxvw0217`, `ttwl0228`): 11 pages, 108 components (85 paragraphs, 10 headings, 6 tables, 6 figures, 1 caption).
- **Full ingest pending:** about 5 h at 17.8 s/page, resumable: `.venv/Scripts/python -m scripts.ingest --source mpdocvqa`.
- **Demo-PDF test pending** (moved into P3): `data/raw/pdfs/` was empty.

### Decisions
- OCR backend is onnxruntime: 27% faster than torch, with identical text and bboxes on all 108 components.
- Page headers and footers are kept as paragraphs (DocVQA answers are often in them).
- Multi-region Docling items keep every box in `meta.regions`.

## P3 (approved, 2026-10-18): indexes (BM25, dense, CLIP)
Full report: [docs/reports/P3_REPORT.md](reports/P3_REPORT.md)

### What exists now
- `src/docscout/indexing/`:
  - `base.py`: `BaseIndex` interface, `file_fingerprint`
  - `bm25_index.py`: tokenizer + BM25Okapi
  - `dense_index.py`: sentence-transformers + FAISS IndexFlatIP; the BGE instruction is applied to queries only; models load from the cache first
  - `visual_index.py`: CLIP clip-ViT-B-32 over figure crops
  - `build_all.py`: layout, build/load, staleness
- `scripts/build_indexes.py` (`--rebuild`; exit 2 when stale) and `scripts/search_cli.py` (`--branch --retriever --doc --k --target --show-encoded`).
- 10 indexes in `data/indexes/{text,table,figure}/{bm25,dense_bge,dense_minilm}` plus `figure/clip`, built over 4 MP-DocVQA documents (111 components; `pybv0228` was added to test staleness).
- Tests: 36 passing.

### Key interfaces
- `load_index(branch, retriever)` → `idx.search(query, top_k, doc_id=None) -> [(component_id, score)]`. The doc filter is exact (per-document rows). `DenseIndex.encode_query()` and `search_vector()` let a query be encoded once and reused.
- Score scales: BM25 ≥ 0 and unbounded; dense cosine about 0.2–0.9; CLIP about 0.2–0.35.
- Each index's `meta.json` stores the fingerprint of `components.jsonl` (lines + sha256). After more ingest, run `build_indexes --rebuild`.

### Limitations and decisions
- **CLIP "bar graph" limitation:** Docling splits some scanned charts into tiny noisy fragments, and dark full-page covers get detected as figures. CLIP scores cluster within about 0.02, so "bar graph" ranked a black cover page first. Clear images work ("line graph of temperature", "company logo"). Not tuned.
- **Demo PDF deferred to P6/P7:** `data/raw/pdfs/` is still empty.
- Model downloads: BGE 418.7 MB and CLIP 578.5 MB (MiniLM was already cached).

## P4 (approved, 2026-10-18): eval set, qrels, baselines + folder structure
Full report: [docs/reports/P4_REPORT.md](reports/P4_REPORT.md) · results: [eval/results/summary.md](../eval/results/summary.md)

### Evaluation set
- `data/raw/mpdocvqa/eval_docs.txt`, from `eval.select_eval_docs`: 150 documents in saved order, the 4 already-ingested ones kept, documents with > 20 pages skipped, stopping at ~250 pages.
  - Result: **49 documents, 250 pages, 293 questions** (246 on documents with ≥ 2 pages), 12 single-page documents, pages per document 1 / 5.10 / 20.
- All 49 are ingested: 2,976 components (2,380 paragraphs, 339 headings, 96 tables, 120 figures, 41 captions). `check_components` passes; one blank page is reported as info (`hqgb0228` p19).
- The 10 indexes are rebuilt on it. The full 150-document ingest is still not done (not needed for evaluation).

### Decisions
- **pypdfium2 + isolation.** docling-parse crashed natively (0xC0000005 in `pdf_parsers.pyd`) after 7 documents in one process; the crash was not reproducible in a fresh process.
  - MP-DocVQA joined PDFs (image-only) now use the **pypdfium2** backend (`parsing.pdf_backend`). Demo PDFs keep docling_parse.
  - Every document is parsed in a **spawned child worker**. A crash or timeout is logged to `data/processed/ingest_failures.jsonl`, the worker restarts and the run continues.
- **Qrels:**
  - Page relevance = `answer_page`.
  - Component relevance: the text, `text_form` or a table cell on the answer page contains a gold answer. Normalized to `[a-z0-9]`; `partial_ratio ≥ 90` only when the text is at least as long as the answer, else `ratio`; short answers (≤ 3 chars) need an exact word or cell.
  - Coverage 94.9%. This is genuine for extractive DocVQA answers; an audit removed a matching bug.
- **Metrics:**
  - Page metrics on all questions and on documents with ≥ 2 pages (the main number); component metrics on the 278 questions that have a relevant component.
  - Random baseline = mean(1/num_pages).
  - **BM25 page runs stay unpadded:** pages with no term match are not ranked.

### Key numbers (page level, documents with ≥ 2 pages; component level on 278 questions)
| system | page R@1 | page R@5 | page MRR | comp R@1 | comp R@10 | ms/query |
|---|---|---|---|---|---|---|
| random | 0.246 | 0.770 | 0.473 | - | - | - |
| bm25_all | 0.593 | 0.793 | 0.679 | 0.380 | 0.633 | 0.2 |
| dense_minilm_all | 0.537 | 0.902 | 0.696 | 0.244 | 0.729 | 9.4 |
| dense_bge_all | 0.606 | 0.907 | 0.738 | 0.331 | 0.726 | 35.4 |

### Folder tidy-up (after P4)
`scripts/` now holds only thin entry points (argparse + one call); the logic moved into the package and `eval/`. The commands are unchanged except `select_eval_docs`.

| was | now |
|---|---|
| `scripts/download_data.py` (logic) | `src/docscout/dataset/mpdocvqa.py` → `download_subset()` |
| `scripts/explore_data.py` (logic) | `src/docscout/dataset/explore.py` → `explore()` |
| `scripts/ingest.py` (worker, isolation, cleanup, failures, run loop) | `src/docscout/ingestion/runner.py` → `run_ingest()`, `Worker`, `failed_doc_ids()`, `clean_doc_outputs()`, `drop_docs()` |
| `scripts/check_components.py` (validation) | `src/docscout/ingestion/validate.py` → `validate()`, `report()`, `duplicate_ids()` |
| `scripts/visualize_components.py` (drawing) | `src/docscout/ingestion/visualize.py` → `render_doc()`, `draw_page()` |
| `scripts/build_indexes.py` (report) | `src/docscout/indexing/build_all.py` → `build_report()` |
| `scripts/search_cli.py` (search loop) | `src/docscout/indexing/search.py` → `search_report()` |
| `scripts/check_env.py` (library list) | `src/docscout/env_check.py` → `check_env()` |
| `scripts/select_eval_docs.py` | `eval/select_eval_docs.py` (run as `python -m eval.select_eval_docs`) |

### Pending
- **Demo PDF:** deferred again (not added in P4).
- `.env` with `LLM_API_KEY` exists locally (gitignored) for P5.
