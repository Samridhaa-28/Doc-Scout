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

## P5 (approved, 2026-10-20): classifier, hybrid fusion, rerank, ablation
Full report: [docs/reports/P5_REPORT.md](reports/P5_REPORT.md) · results: [eval/results/summary.md](../eval/results/summary.md) · plot: `eval/results/ablation.png`

### What exists now
- `src/docscout/reasoning/llm_client.py`: provider-agnostic `LLMClient` (Groq implemented): JSON mode, temperature 0, sha256 disk cache (`data/processed/llm_cache.jsonl`), backoff on 429/5xx, minimum call gap (`llm.min_interval_s`), counters for calls / cache hits / tokens.
- `src/docscout/query/classifier.py`: word-boundary rules first; the LLM decides when the rules are unsure (both or neither group match).
- `src/docscout/retrieval/`: `retrievers.py` (scores every component of the document with BM25 + BGE, + CLIP for figures; `Candidate` keeps raw / norm / fused / rrf / rerank / final), `normalize.py` (min-max per (branch, retriever) list, z-score option), `fusion.py` (weighted sum, RRF k=60, hard routing).
- `src/docscout/rerank/cross_encoder.py`: ms-marco-MiniLM-L-6-v2 on the top 30; the rest keep their fused order.
- `eval/split.py` (document-level dev/test split, **committed** as `eval/results/split.json`), `eval/eval_classifier.py`, `eval/tune_fusion.py` (dev only), `eval/plot_results.py`, and `eval/run_eval.py` with 8 systems, per-split metrics, p90 latency and ranx paired t-tests.
- Tests: 87 passing.

### Key interfaces
- `get_client() -> LLMClient`; `client.complete_json(prompt, system)`; `client.stats`.
- `classify(query, client=None, use_llm=None) -> {"type", "confidence", "source"}`; `weights_for(type)`.
- `HybridRetriever(load, comp_meta, cfg).retrieve(query, doc_id, branches=None) -> List[Candidate]`.
- `rank_candidates(cands, qtype, method, routing, retriever_weights, branch_weights, rrf_k, normalization)`.
- `CrossEncoderReranker(cfg=cfg).rerank(query, ranked, texts)`. The P6 feedback boost goes on `c.final` after rerank.
- `eval.split.load_split()`. Result JSON: `{system, splits: {test, dev, all}, latency_ms_median, latency_ms_p90, runs (full)}`.
- Tuned config (dev only): `fusion.retriever_weights` bm25 0.7 / dense 0.3 / clip 0.3; `branch_weights` textual {0.4, 0.375, 0.225}, numerical {0.3, 0.6, 0.1}, visual {0.25, 0.15, 0.6} (text / table / figure).

### Split
Split by document, seed 42, 20% of documents to dev:
- dev: 10 documents, 76 questions
- test: 39 documents, 217 questions (176 on documents with ≥ 2 pages, 209 with a relevant component)

### TEST results (page level on documents with ≥ 2 pages; component level)
| system | page R@1 | page MRR | comp R@1 | comp MRR | median / p90 ms |
|---|---|---|---|---|---|
| random | 0.270 | 0.501 | - | - | - |
| bm25_all | 0.636 | 0.718 | 0.417 | 0.552 | 0.2 / 0.5 |
| dense_minilm_all | 0.568 | 0.725 | 0.259 | 0.465 | 8.9 / 11.8 |
| dense_bge_all | 0.631 | 0.756 | 0.362 | 0.544 | 35.6 / 40.4 |
| hybrid_weighted_soft | 0.659 | 0.788 | 0.432 | 0.599 | 54.3 / 60.2 |
| hybrid_rrf_soft | 0.659 | 0.786 | 0.398 | 0.581 | 54.8 / 60.5 |
| hybrid_weighted_hard | 0.602 | 0.696 | 0.360 | 0.473 | 36.1 / 43.3 |
| **hybrid_weighted_soft_rerank** | **0.733** | **0.828** | **0.488** | **0.653** | 762.7 / 1850.9 |
| hybrid_weighted_soft_goldtype (analysis) | 0.670 | 0.795 | 0.449 | 0.607 | 57.2 / 66.4 |

Paired t-test against dense_bge_all (p < 0.05):
- soft: page MRR p = 0.077 (not significant), component MRR p = 0.003
- soft + rerank: page MRR p = 0.0002, component MRR p < 0.0001

Classifier (293 questions, DocVQA question_types, 100% join), accuracy / macro-F1:
- rules-only: 0.614 / 0.521
- llm-only: 0.747 / 0.644
- combined: 0.751 / 0.652

A cached rerun makes 0 LLM calls.

### Decisions
- **LLM = `openai/gpt-oss-20b` for all evaluation.** The requested `llama-3.1-8b-instant` returns 404 on this Groq key, which has no Llama chat models. gpt-oss-20b is the small, fast option with JSON mode; it runs with `reasoning_effort: low`, and the account limits are 1,000 requests/day and 8,000 tokens/min. P6 will add an optional demo-only model setting (gpt-oss-120b).
- **P6 pipeline default = `hybrid_weighted_soft_rerank`.**
- **Split committed:** `.gitignore` has `!eval/results/split.json`.
- **Normalization unchanged (min-max), no post-hoc changes. Known limitation: lone table.**
  - Min-max gives the best table 1.0 from both retrievers, so a numerical query puts it at exactly w_table = 0.6, above any paragraph (≤ 0.3).
  - This hurts on DEV: page MRR 0.586 for hybrid vs 0.693 for BGE; e.g. 49172, where the table on p9 beats the answer paragraph on p10. On TEST the effect is small (13 of 46 such top-1 tables are on the wrong page).
  - Rerank mostly repairs it.
- Hard routing is strict: an empty ranking when the document has no component in the predicted branch (34/217 test questions).
- The Avast TLS workaround (`SSL_CERT_FILE`) is still needed on this machine for Groq and HF.

### Pending
- Demo PDF still missing (`data/raw/pdfs/`).

## P6 (approved, 2026-10-23): localization, graph context, grounded answers, provenance, pipeline
Full report: [docs/reports/P6_REPORT.md](reports/P6_REPORT.md) · evidence: `docs/reports/p6_evidence/` · results: `eval/results/summary.md` (ANLS section)

### What exists now
- `src/docscout/pipeline.py`: `DocScoutPipeline.answer(query, doc_id, session_id=None, top_k=5, feedback_adjust=None, *, slots=None, ...) -> QueryResult`. It runs 10 steps (classify → retrieve → normalize → fuse → rerank → feedback hook → localize → expand → answer → provenance). `meta.reranked` is the full list for "show next".
- `localization/table_locator.py` (`locate`), `context/graph_expander.py` (`expand`, `expand_evidence`), `reasoning/prompts.py` + `answerer.py` (numbered evidence, prompt cap 2,500 est. tokens, JSON + citation validation), `evidence/provenance.py` (why-line), `retrieval/slots.py` (modality slot reservation).
- `scripts/ask.py` (`--demo` = gpt-oss-120b, `--store`), `eval/eval_answers.py`, `eval/eval_locator.py`.
- A separate demo store (`stores.demo`, `--store demo`) holds the Census P60-281 PDF; the eval corpus and the P5 numbers are unchanged.
- Tests: 133 passing.

### Results (TEST, gpt-oss-20b)
- ANLS: (a) top-1 0.486 [0.424, 0.546]; (b) top-3 + context + cells 0.646 [0.585, 0.707]; oracle (40 q) 0.791. (b) − (a) = +0.161 [+0.103, +0.218].
- Not found 38.2% / 15.7% / 15.0%; 0 invalid citations; the cached rerun makes 0 LLM calls.
- Locator cell hit rate: clean v1 estimate 34/62 pipeline, 38/62 oracle (the shipped v3 has two TEST-informed rules: 32/62, 38/62).
- Slot reservation: type-conditional adopted by a pre-registered non-inferiority rule on DEV (−0.005 [−0.015, 0.000], no benefit); TEST once 0.637 vs 0.646 (−0.009 [−0.023, 0.000]). Default `answer.slots: conditional`; the eval scripts default to `--slots off`.

### Pending
- Demo ask.py comparison with/without slots at k=5 (deferred: Groq daily token budget).
- The demo visual question fails at k=5 without slots (the cross-encoder pushes Figure 2 to about rank 25).
- Groq has a rolling daily token budget (~200k/day observed); the client waits out Retry-After (`llm.max_retry_after_s`).
