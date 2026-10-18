## PHASE REPORT: P4 Ground truth, metrics, first baselines

### Resolution of the earlier block
| Issue | What I changed (file: one line) | Result |
|---|---|---|
| Native crash (0xC0000005 in docling-parse `pdf_parsers.pyd`) while ingesting `zrhw0228` | **Option 2:** `src/docscout/ingestion/parser.py`: `build_converter(cfg, backend=...)` with `PyPdfiumDocumentBackend`; `configs/config.yaml`: `parsing.pdf_backend: {mpdocvqa: pypdfium2, pdf: docling_parse}` | pypdfium2 parses `zrhw0228` (10/10 pages, 105 components). On `yxvw0217` it gives the same 18 component ids and types. The differences are 1–2 px of bbox jitter, one reading-order swap of two footer blocks, and slightly different OCR text ("DesignWrite" vs docling_parse's "DesignW rite") |
| A native crash kills the whole run | **Option 1:** `scripts/ingest.py`: each document is parsed in a spawned child worker. If the worker dies or times out, the document is recorded in `data/processed/ingest_failures.jsonl` (per backend), a fresh worker starts, and the run continues | `test_worker_survives_child_crash`: worker killed with exit code 3 → reported as `('crash', 3)`, restarted, next document OK |
| Is the crash reproducible? | Re-ran `zrhw0228` with docling_parse in an isolated worker | **Not reproducible:** it succeeded (10 pages, 96.5 s). The original crash came after 7 documents in one long-lived process, so it is intermittent |
| Consistency | Re-ingested all 12 documents parsed so far with pypdfium2 (`--force`) | All 49 eval documents now record `pdf_backend: pypdfium2` |

### 0. Summary in plain words
- **Evaluation set:** 49 MP-DocVQA documents / 250 pages / 293 questions, selected by your rule and written to `eval_docs.txt`.
- **Ingest:** all 49 documents ingested in 6 resumable chunks (about 60 min in total) with crash isolation and the pypdfium2 backend: no failures, 2,976 components, all checks pass. Indexes rebuilt over them.
- **Qrels:** page qrels and component qrels built from `answer_page` and gold-answer matching (including parquet table cells).
- **Coverage is 94.9%,** above the expected 60–85%. I audited it and fixed one matching bug; what remains is genuine (DocVQA answers are copied from the page).
- **Baselines:** three baselines evaluated with ranx at page and component level. On multi-page documents (main number), page Recall@1 is BM25 0.593 / MiniLM 0.537 / BGE 0.606, against a random baseline of 0.246.
- BGE ≥ MiniLM on 14 of 15 metrics. BM25 has the best component Recall@1 (0.380).
- 48 tests pass.

### 1. Files created/changed
**Committed and pushed in `e89a2d0 phase-3: indexes (bm25, dense, clip)`:** `docs/PROGRESS.md` ("P3 (approved)").

**Not committed (P4):**
- `scripts/select_eval_docs.py`: eval-document selection (saved order, keep already-ingested, skip > 20 pages, stop at ~250 pages, skip documents that failed ingest) → `eval_docs.txt`
- `eval/build_qrels.py`: page and component qrels in ranx format, coverage by answer kind and table/non-table page, `qrels_info.json`
- `eval/metrics.py`: `evaluate_rankings` (ranx), `pages_from_components` (max), `anls`, `anls_single`
- `eval/run_eval.py`: systems `bm25_all`, `dense_minilm_all`, `dense_bge_all`; page runs from component runs; metrics on all, ≥ 2-page and component subsets; random baseline; latency; `--failures`; `{system}.json` and `summary.md`
- `tests/test_eval.py` (11 tests), `tests/test_ingest_isolation.py` (1 test)

**Earlier-phase files changed** (approved in your reply; reasons in section 2):
- `src/docscout/ingestion/parser.py` (P2): selectable PDF backend (`pdf_backend_for`, `build_converter(..., backend)`); the backend is recorded in each manifest's `parse_stats`
- `scripts/ingest.py` (P2): child-process isolation, `--docs-file`, `--max-minutes`, `--stop-on-error`, `--pdf-backend`, `--retry-failed`, `--no-isolate`, `--doc-timeout`, failure log, progress line
- `scripts/check_components.py` (P2): blank pages processed by the parser count as information, not a page mismatch (fix 2)
- `configs/config.yaml`: `parsing.pdf_backend`

### 2. Design decisions and any deviations from the architecture (and why)
1. **Evaluation set:** the 150 documents in saved order; the 4 ingested documents are kept; documents with > 20 saved pages are skipped (`snbx0223`, `ylvj0223`, `psyn0081`); stop at ≥ 250 pages. This gives 49 documents and exactly 250 pages. The script also skips documents that failed ingest (none did).
2. **PDF backend per source:** pypdfium2 for MP-DocVQA (image-only joined PDFs, where docling-parse's text-cell extraction adds nothing and it crashed once), docling_parse for digital demo PDFs. The backend is recorded per document.
3. **Isolation:**
   - One persistent spawned worker builds the converter once and receives documents through a queue; the parent detects a dead worker or timeout (default 30 min).
   - The parent is the only process that writes `components.jsonl`.
   - Failures are logged per backend, so a document that crashed with docling_parse isn't skipped when pypdfium2 is used.
   - Cost: the converter loads once per run (about 20–40 s), not once per document.
4. **Page qrels:** `{qid: {"{doc_id}_p{answer_page}": 1}}`. `answer_page` is already 1-based and equals `Component.page`; worked example in section 4.
5. **Component qrels:** components on the answer page whose `text_form`, `text` or any parquet cell/header contains a gold answer.
   - Normalization: lowercase, `[a-z0-9]` only.
   - Answers longer than 3 characters: the text must contain the answer. That means rapidfuzz `partial_ratio ≥ 90` when the text is at least as long as the answer, else `ratio ≥ 90` (fix 3).
   - Answers of ≤ 3 characters: exact match with a whole word or a whole table cell.
   - The 15 questions with no relevant component stay in the page metrics and are excluded from the component metrics.
6. **Answer kinds** (the dataset has no question types): numeric if every gold answer looks like a number, date, amount or percentage (`is_numeric_answer`), otherwise text. Table page = the answer page has at least one table component.
7. **Systems:**
   - Search within the question's document, `top_k = 1000` per branch, i.e. the whole document.
   - `bm25_all` merges the three branch BM25 lists by max score. Each component belongs to exactly one branch, so this is a union sorted by raw BM25 score.
   - The dense systems encode the query once (BGE with its query instruction) and search all three branches with the same vector.
8. **Page runs:** page score = max of its components' scores. Pages without a retrieved component are not in the run and count as misses. This is deliberate and not padded; see section 8.
9. **Random baseline:** expected R@1 = mean(1/num_pages); also expected R@5 and R@10 = mean(min(k, n)/n) and MRR = mean(H_n/n).

### 3. How to run (exact commands, in order, as a teammate would on a fresh clone)
```powershell
.venv/Scripts/python -m scripts.select_eval_docs --target-pages 250 --max-pages 20
.venv/Scripts/python -m scripts.ingest --source mpdocvqa --docs-file data/raw/mpdocvqa/eval_docs.txt --max-minutes 12   # repeat until "49/49"
.venv/Scripts/python -m scripts.check_components
.venv/Scripts/python -m scripts.build_indexes --rebuild
.venv/Scripts/python -m eval.build_qrels
.venv/Scripts/python -m eval.run_eval --systems bm25_all dense_minilm_all dense_bge_all --failures 3
.venv/Scripts/python -m pytest
```

### 4. Actual output (real terminal output from this session, trimmed)

**Evaluation set:**
```
documents: 49 (already ingested kept: ['jjvg0227', 'pybv0228', 'ttwl0228', 'yxvw0217'])
pages: 250 | pages/doc min=1 mean=5.10 max=20
single-page documents: 12
questions kept: 293 | on documents with >=2 pages: 246
skipped (> 20 pages) while selecting: 3 ['snbx0223', 'ylvj0223', 'psyn0081']
```
**pypdfium2 verification:**
```
yxvw0217: backend=pypdfium2 status=success pages=2 with_components=2 errors=[] comps=18 {'paragraph': 14, 'heading': 2, 'table': 2} 44.5s (22.2 s/page)
zrhw0228: backend=pypdfium2 status=success pages=10 with_components=10 errors=[] comps=105 {...} 91.2s (9.1 s/page)
yxvw0217 by content vs docling_parse: the same 18 components up to 1-2 px bbox jitter; one footer pair swapped in reading order
  ('Page 3 of 13' <-> 'DesignWrite, 1 Inc.'); OCR text 'DesignWrite' (pypdfium2) vs 'DesignW rite' (docling_parse)
```
**Isolated re-run with docling_parse on `zrhw0228`:** `"status": "success", "pdf_backend": "docling_parse", "pages": 10, "secs": 96.5` (crash not reproduced).

**Ingest chunks** (all with `pdf_backend: pypdfium2`, isolation on):
```
re-ingest (--force) 12 docs : docs: 12 | pages: 32 | time: 351.1s, 10.97 s/page | failures this run: 0 | progress: 12/12
chunk A                     : docs: 6  | pages: 62 | 15.4 min | failures this run: 0 | progress: 18/49 documents done
chunk B                     : docs: 5  | pages: 45 | 12.3 min | failures this run: 0 | progress: 23/49 documents done
chunk C                     : docs: 13 | pages: 57 | 12.3 min | failures this run: 0 | progress: 36/49 documents done
chunk D                     : docs: 10 | pages: 50 | 12.9 min | failures this run: 0 | progress: 46/49 documents done
chunk E                     : docs: 3  | pages: 4  | 1.3 min  | failures this run: 0 | progress: 49/49 documents done
```
Total 250 pages in about 60 min, roughly 14.4 s/page on average (chunk averages 11–20 s/page).

**check_components** (after fix 2):
```
lines: 2976 | unique ids: 2976 | duplicates: 0
ok   duplicate_ids / bad_id_format / empty_text_form / bad_page / bbox_outside / missing_files / size_mismatch / page_mismatch / answer_page_missing : 0
info blank_pages (processed by the parser, no components): [('hqgb0228', [19])]
ALL CHECKS PASSED
pdf_backend per document: {'pypdfium2': 49} | components by type: paragraph 2380, heading 339, table 96, figure 120, caption 41
```
**build_indexes --rebuild** (2,976 components):
```
text    bm25           2760  built      0.037    579.7 KB
text    dense_bge      2760  built    114.944  16666.4 KB
text    dense_minilm   2760  built     14.918   8386.4 KB
table   bm25             96  built      0.011     88.6 KB
table   dense_bge        96  built     24.110    579.8 KB
table   dense_minilm     96  built      2.816    291.8 KB
figure  bm25            120  built      0.004     48.5 KB
figure  dense_bge       120  built     15.910    724.8 KB
figure  dense_minilm    120  built      1.317    364.7 KB
figure  clip            120  built      7.913    484.7 KB
total: 182.0s
```
**Page index conversion, worked by hand** (question 49168):
```
question       : 49168 'What time is the ‘coffee break’?'
answers        : ['11.14 to 11.39 a.m.', '11:14 to 11:39 a.m.']
page_ids       : ['txpp0227_p7', 'txpp0227_p8', 'txpp0227_p9']
answer_page_idx: 2 (0-based index into page_ids)
page_ids[2]    : txpp0227_p9  -> k = 9 (0-based original page) -> 1-based page = k + 1 = 10
answer_page    : 10 | matches: True
saved pages    : [8, 9, 10]
image path     : data/raw/mpdocvqa/images/txpp0227/page_10.png exists: True
page qrel id   : txpp0227_p10 (components on it have ids txpp0227_p10_<type>_<n>)
```
**build_qrels** (final, after fix 3):
```
eval documents: 49 (ingested 49) | questions: 293
page qrels: 293 questions
component qrels: 278 questions with >=1 relevant component; 15 excluded from component metrics (kept in page metrics)
relevant components per covered question: mean 1.40
=== Coverage (questions with >=1 relevant component) ===
  all questions                : 278/293 = 94.9%
  numeric answers              : 76/77 = 98.7%
  text answers                 : 202/216 = 93.5%
  answer page has a table      : 115/117 = 98.3%
  answer page has no table     : 163/176 = 92.6%
  docs with >=2 pages          : 233/246 = 94.7%
  answer page has no components: 0 questions
answer kinds: {'numeric': 77, 'text': 216} | table pages: 117
```
**Coverage audit** (relevant pairs by how they matched, after fix 3):
```
{'exact_substring': 298, 'short_word_or_cell': 77, 'fuzzy_only': 14}
fuzzy_only samples: gold 'University of California' -> 'UNIVERSITYOFÇALIEORNIA, SAN DIEGO' | 'Appendix 1 - Results of ...' -> 'Appendix I - Results of ...'
                    'Sustainability Updates' -> 'Sustainabilit Updates' | '3-28-2001' -> '28-2001' (OCR dropped the '3'; borderline)
```
**run_eval** (293 questions; 246 on documents with ≥ 2 pages; 278 with a relevant component):
```
random page Recall@1 (expected): all = 0.367, >=2 pages = 0.246
[bm25_all] median 0.3 ms/query
  page (>=2 pages): recall@1=0.593 recall@5=0.793 recall@10=0.833 mrr=0.679 ndcg@10=0.716
  page (all)      : recall@1=0.625 recall@5=0.792 recall@10=0.826 mrr=0.696 ndcg@10=0.727
  component       : recall@1=0.380 recall@5=0.588 recall@10=0.633 mrr=0.522 ndcg@10=0.530
[dense_minilm_all] median 9.4 ms/query
  page (>=2 pages): recall@1=0.537 recall@5=0.902 recall@10=0.963 mrr=0.696 ndcg@10=0.760
  page (all)      : recall@1=0.611 recall@5=0.918 recall@10=0.969 mrr=0.745 ndcg@10=0.798
  component       : recall@1=0.244 recall@5=0.593 recall@10=0.729 mrr=0.448 ndcg@10=0.502
[dense_bge_all] median 35.4 ms/query
  page (>=2 pages): recall@1=0.606 recall@5=0.907 recall@10=0.976 mrr=0.738 ndcg@10=0.795
  page (all)      : recall@1=0.669 recall@5=0.922 recall@10=0.980 mrr=0.780 ndcg@10=0.828
  component       : recall@1=0.331 recall@5=0.633 recall@10=0.726 mrr=0.514 ndcg@10=0.551
```
**summary.md** (`eval/results/summary.md`):
```
# Retrieval results (within-document search)

Evaluation set: 49 documents from `eval_docs.txt`, 293 questions. Page metrics on documents with >= 2 pages: 246 questions (main page-level number). Component metrics: 278 questions with >= 1 relevant component.

## Page level: documents with >= 2 pages (main)
| system | R@1 | R@5 | R@10 | MRR | nDCG@10 | median ms/query |
|---|---|---|---|---|---|---|
| random (expected) | 0.246 | 0.770 | 0.938 | 0.473 | - | - |
| bm25_all | 0.593 | 0.793 | 0.833 | 0.679 | 0.716 | 0.2 |
| dense_bge_all | 0.606 | 0.907 | 0.976 | 0.738 | 0.795 | 35.4 |
| dense_minilm_all | 0.537 | 0.902 | 0.963 | 0.696 | 0.760 | 9.4 |

## Page level: all questions
| system | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| random (expected) | 0.367 | 0.807 | 0.948 | 0.557 | - |
| bm25_all | 0.625 | 0.792 | 0.826 | 0.696 | 0.727 |
| dense_bge_all | 0.669 | 0.922 | 0.980 | 0.780 | 0.828 |
| dense_minilm_all | 0.611 | 0.918 | 0.969 | 0.745 | 0.798 |

## Component level (questions with >= 1 relevant component)
| system | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| bm25_all | 0.380 | 0.588 | 0.633 | 0.522 | 0.530 |
| dense_bge_all | 0.331 | 0.633 | 0.726 | 0.514 | 0.551 |
| dense_minilm_all | 0.244 | 0.593 | 0.729 | 0.448 | 0.502 |
```
**Why BM25's page R@5 and R@10 are near or below random** (diagnostic, ≥ 2-page questions):
```
bm25_all: >=2-page questions 246 | empty page ranking 29 | ranks only some pages 137 | on the 217 non-empty: recall@1=0.673 recall@5=0.899 recall@10=0.945
dense_bge_all: >=2-page questions 246 | empty page ranking 0 | ranks only some pages 12 | on the 246 non-empty: recall@1=0.606 recall@5=0.907 recall@10=0.976
```
**One BM25 failure** (from `run_eval --systems bm25_all --failures 3`; `bm25_all: 92 component-level failures (no relevant component in top 10), 59 of them with < 3 retrieved components`):
```
--- bm25_all failure: question 49187 (doc txpp0227, answer page 10)
Q: What time is “introductory remarks” session?
gold answers: ['11:39 to 11:44 a.m.', '11.39 to 11.44 a.m.']
relevant components: ['txpp0227_p10_paragraph_2'] | first relevant rank: not retrieved
  top1 16.765 txpp0227_p10_paragraph_5 [paragraph, p10] '"Introductory Remarks" Lee A. Waller, TRRF Vice Presi-'
  top2 8.317 txpp0227_p9_table_0 [table, p9] 'Table: MONDAY, MAY 15\nRows: 8:15 to 8:56 a.m.; 8:58 a.m.; 8:58 to 9:03 a.m.; ...'
  top3 6.845 txpp0227_p10_paragraph_4 [paragraph, p10] 'TRRF GENERAL SESSION (PART I) Presiding: Lee A. Waller TRRF Vice President'
  relevant: txpp0227_p10_paragraph_2 '11:39 to 11:44 a.m.'
```
**Why it failed:** the agenda is laid out as two columns (time | session). Docling made each time cell a separate paragraph, `"11:39 to 11:44 a.m."`, which shares no words with the question. BM25 correctly finds the session label "Introductory Remarks" (rank 1, on the right page, so it's a page-level hit) but can never reach the time next to it. P6's graph context (siblings of the top hit) is designed for exactly this case. MiniLM shows the same failure (relevant at rank 14).

### 5. Tests
`48 passed` (36 earlier + 12 new):
- `test_eval.py::test_anls_exact_and_case_whitespace`, `test_anls_partial_and_threshold`, `test_anls_mean`: ANLS (DocVQA definition, threshold 0.5)
- `test_eval.py::test_pages_from_components_uses_max`: page score = max of its components
- `test_eval.py::test_merge_max_keeps_highest_score`
- `test_eval.py::test_evaluate_rankings_with_ranx`: ranx R@1, R@5 and MRR on a hand-computed case (a query without a run counts as a miss)
- `test_eval.py::test_random_baseline`: mean(1/num_pages)
- `test_eval.py::test_norm_and_numeric_kind`
- `test_eval.py::test_matching_long_fuzzy_and_short_exact`: OCR typo via fuzzy match; "5" matches a word but not "1,500"
- `test_eval.py::test_short_text_does_not_match_long_answer`: **regression for fix 3** (`'0'` vs `'(910) 741-5327'`, `'A'` vs `'Joyce Bagby'`, a partial time range)
- `test_eval.py::test_relevant_components_uses_table_cells`: answer found only in a parquet cell
- `test_ingest_isolation.py::test_worker_survives_child_crash`: a child killed with `os._exit(3)` → `('crash', 3)`, restarted, a Python error is reported, the next document succeeds
- The existing data test (`test_processed_components_valid`) now runs on all 2,976 components

### 6. Acceptance criteria and reviewer checks
- PASS: eval set selected and reported. 49 documents / 250 pages / 293 questions / 12 single-page documents / pages per document 1, 5.10, 20.
- PASS: eval set ingested in ~15-min chunks with progress. 6 runs, `49/49 documents done`, 0 failures.
- PASS: `check_components` all pass. 2,976 lines = 2,976 unique ids; `ALL CHECKS PASSED` (after fix 2).
- PASS: `build_indexes --rebuild`. 10 indexes, 182 s; next run: 10 up to date.
- PASS: qrels built and coverage printed. Section 4: 94.9% overall, by numeric/text and table/non-table.
- PASS with explanation: coverage "expect roughly 60–85%". Measured 94.9%, after removing a matching bug that had inflated it to 96.2% (fix 3). An audit shows 298/389 pairs are exact substrings, 77 are exact short words or cells, and 14 are fuzzy OCR variants. DocVQA answers are extractive and we keep all page text, so high coverage is expected for this setup.
- PASS: summary.md has 3 rows with page and component metrics (plus the random row and a ≥ 2-page table).
- PASS: page Recall@1 clearly beats 1/avg_pages. ≥ 2 pages: random 0.246 vs BM25 0.593 / MiniLM 0.537 / BGE 0.606. All questions: random 0.367 vs 0.625 / 0.611 / 0.669.
- PASS for dense, PARTIAL for BM25: page Recall@5 well above random. ≥ 2 pages: BGE 0.907 and MiniLM 0.902 vs random 0.770; BM25 0.793, barely above, and its R@10 of 0.833 is below random 0.938. Cause: BM25 returns no component for 29/246 questions and covers only some pages for 137 more (on non-empty runs, R@5 = 0.899). Short documents (5.1 pages on average) also make random R@5 and R@10 high. Not tuned; see section 8.
- PASS: page index conversion shown by hand (question 49168: idx 2 → `txpp0227_p9` → page 10 = `answer_page`).
- PASS: BGE ≥ MiniLM in most metrics. 14/15 (all page metrics in both subsets, component R@1/R@5/MRR/nDCG@10); exception: component R@10, 0.726 vs 0.729.
- PASS: questions without a relevant component excluded from component metrics and kept in page metrics (15 questions; reported).
- PASS: one BM25 failure with question, gold, top-3 and the reason (question 49187).
- SKIPPED (folder still empty): ingest a PDF if one appears in `data/raw/pdfs/`.
- PASS: `pytest`: `48 passed`.

### 7. Interfaces for the next phase
- **Systems:** `eval.run_eval.make_systems(cache, top_k)` returns `{name: fn(question, doc_id) -> [(component_id, score)]}`. P5 adds hybrid systems there. `IndexCache.get(branch, retriever)` loads each index once.
- **Metrics:**
  - `eval.metrics.evaluate_rankings(qrels, runs, metrics) -> {metric: mean}`
  - `pages_from_components(ranking, comp_page)`
  - `anls(preds, golds)` (used in P6)
- **Result files** (`eval/results/`):
  - `qrels_page.json` / `qrels_component.json`: ranx format `{qid: {id: 1}}`
  - `qrels_info.json`: `{qid: {doc_id, num_pages, numeric, table_page, n_relevant, n_page_components}}`, useful for P5's dev split and error analysis
  - `{system}.json`: `{metrics: {page_all, page_multi, component}, latency_ms_median, runs: top-50 per qid}`
  - `baseline.json`, `summary.md` (regenerated from every result file)
- **Ingest:**
  - `ingest --docs-file`, `--max-minutes` (resumable chunks)
  - isolation on by default; failures in `data/processed/ingest_failures.jsonl`
  - `parsing.pdf_backend` per source
- **Eval set:** `data/raw/mpdocvqa/eval_docs.txt` (49 documents). P5 should take its 20% dev split from these 293 questions.

### 8. Known limitations / open questions
1. **BM25 page rankings are incomplete.** Pages without a BM25 term match are never ranked (29/246 questions return nothing, 137 partially), so BM25 page R@10 is below random. I evaluated it as is (no padding). Hybrid fusion in P5 combines BM25 with dense scores and covers this. If you prefer, unranked pages could be appended at the end in document order. That would favour early pages, which hold 360/947 answers in the full subset, so I didn't do it.
2. **The coverage criterion (60–85%) doesn't fit this data.** With extractive DocVQA answers and full OCR text, about 95% is genuine. One borderline OCR case remains (`28-2001` for `3-28-2001`).
3. **Component-level gaps come from layout fragmentation.** Values often sit in separate components from their labels (agenda times, budget amounts such as `'$ 975.00$'` next to "Other Expense ...."). That's why component R@1 is only 0.33–0.38. P5 reranking and P6 graph context target this.
4. **The docling-parse crash is intermittent and was not reproduced.** It's avoided for MP-DocVQA by pypdfium2 and contained by isolation. Demo PDFs still use docling_parse, but they're protected by isolation too.
5. **Model choice:** BGE costs 35 ms/query and MiniLM 9 ms on CPU, so both are fine for an interactive demo.
6. A 20-page cap was used in selection. The 3 skipped long documents may behave differently.

### 9. What I need from you before the next phase
1. **Review the numbers** in `eval/results/summary.md` and decide on limitation 1: keep BM25 page runs as is (my recommendation), or append unranked pages in document order.
2. **A working `.env` with `LLM_API_KEY`** (Groq) for P5's LLM classifier. It's not created yet; I won't create it or ask for the key in a file other than `.env`.
3. **Demo PDF:** still missing (`data/raw/pdfs/` is empty). It's deferred to P6/P7, as you said.
4. P4 is **not committed**. Approve it and give a commit message when you're ready.

### 10. Fixes made during this phase
| # | Problem found (how) | Change (file: one line) | Rerun result |
|---|---|---|---|
| 1 | Native crash in docling-parse stopped the ingest after 7 new documents (event log: `pdf_parsers.cp310-win_amd64.pyd`, 0xC0000005) | `parser.py`: pypdfium2 backend for MP-DocVQA (`parsing.pdf_backend`); `ingest.py`: child-process isolation with a failure log and worker restart | 49/49 documents, 0 failures; isolation unit test passes; crash not reproducible in an isolated run |
| 2 | `check_components` reported `page_mismatch: hqgb0228` (found by running it after the ingest). Page 19 is a blank back cover (image opened: grey page, only a faint "Source:" footer), which Docling processed and recorded as having no components | `scripts/check_components.py`: fail only on unknown pages, or on missing pages not recorded by the parser; blank pages listed as `info` | `ALL CHECKS PASSED`, `info blank_pages: [('hqgb0228', [19])]` |
| 3 | Coverage of 96.2% was suspicious. An audit showed 65 fuzzy-only matches, most of them false: `partial_ratio` aligns the shorter string, so a component text `'0'` or `'A'` scored 100 against answers like `'(910) 741-5327'` or `'Joyce Bagby'` | `eval/build_qrels.py`: `contains_answer()` uses `partial_ratio` only when the text is ≥ the answer's length, else full `ratio`; regression test `test_short_text_does_not_match_long_answer` | Coverage 96.2% → 94.9%; relevant per question 1.56 → 1.40; fuzzy-only pairs 65 → 14, and the samples are genuine OCR variants |
| 4 | The BM25 failure printout showed failures with 0–1 retrieved components, so there was no top-3 to inspect | `eval/run_eval.py`: `show_failures` lists failures with ≥ 3 retrieved components first, and prints how many have fewer than 3 | Question 49187 shown with full top-3; `92 failures, 59 with < 3 retrieved` |
| 5 | `summary.md` placeholder "–" (en dash) showed as `â€“` in Windows PowerShell | `eval/run_eval.py`: ASCII `-` placeholders | summary rows show `- \| -` |
