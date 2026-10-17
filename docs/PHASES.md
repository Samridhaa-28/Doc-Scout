# DocScout-X build phases

Do one phase at a time, following CLAUDE.md. Each phase has three parts:
- **Tasks** to build
- **Acceptance criteria**
- **Reviewer checks**: the team will verify these, so check them yourself and report each one in section 6 of the report.

Put the outputs listed under **Evidence** in the report.

---

## P1: Setup, data and shared schema (plan tasks T0.1–T0.3)

### Tasks
1. Create the repo files:
   - `pyproject.toml`: src layout, package `docscout`, `requires-python = ">=3.10"`.
   - `requirements.txt`: the libraries from CLAUDE.md, unpinned, with a comment that torch CPU is installed first.
   - `.gitignore`: .venv, .env, \_\_pycache\_\_, *.pyc, data/raw, data/processed, data/indexes, eval/results/*.json, .ipynb_checkpoints.
   - `.env.example`: LLM_PROVIDER, LLM_API_KEY, LLM_MODEL.
   - `README.md`: Windows setup steps.
2. Install everything into `.venv` in this order:
   1. torch CPU
   2. requirements.txt
   3. `pip install -e .`
   
   Then write `requirements.lock.txt` with `pip freeze`.
3. Create the full folder skeleton with `__init__.py` files so every later module has a home. Also create:
   - `data/raw/pdfs/` (for demo PDFs)
   - `docs/reports/`
   - `docs/PROGRESS.md`, containing a "Starting state" section
4. `configs/config.yaml` with ALL keys later phases need:
   - paths
   - parsing: ocr on/off, image dpi
   - models: bge, minilm, clip, cross-encoder, bge query instruction
   - retrieval: top_k per retriever, rerank_top_n=30
   - fusion: method, retriever weights, branch weights per query type for textual/numerical/visual
   - feedback: step, cap
   - llm: provider, model, temperature, cache path
   - eval: k values, subset size
5. `src/docscout/config.py`: load the YAML once, resolve paths relative to the repo root, load .env, and expose `get_config()`.
6. `src/docscout/schema.py`: dataclasses
   - `Component(id, doc_id, page, type, text, text_form, bbox, page_width, page_height, order, section_id, table_path, image_path, meta)`
   - `Evidence(component, scores dict, context list, localized cells, why)`
   - `QueryResult(query, query_type, answer, evidence list)`
   
   Include to_dict/from_dict and JSONL read/write helpers. Component.id format: `"{doc_id}_p{page}_{type}_{n}"`.
7. `scripts/check_env.py`: import every library, print versions and the torch device, and finish with "all good".
8. `scripts/download_data.py`:
   - FIRST inspect the dataset's real features and print them. Do not assume field names.
   - Use streaming if possible so the full dataset is not downloaded.
   - Save a subset of N documents from the validation split (default 150, `--limit`):
     - page images to `data/raw/mpdocvqa/images/{doc_id}/page_{i}.png`
     - `data/raw/mpdocvqa/questions.jsonl` with question_id, question, doc_id, page_ids, answer_page_idx, answers, and question_types if present
   - Skip documents that are already saved.
9. `scripts/explore_data.py`: print the number of docs, pages per doc (min/mean/max), the number of questions, the distribution of answer_page_idx and of question types, and one full example.
10. Tests: `tests/test_schema.py` (JSONL round trip) and `tests/test_config.py`.

### Acceptance criteria
- `pip install -e .` works and check_env prints "all good".
- `download_data --limit 5` saves 5 documents and their questions. Give me the full-run command (~150 docs); don't run it yet.
- explore_data prints sensible stats, and pytest passes.

### Reviewer checks
- The folder tree matches CLAUDE.md, and every package has `__init__.py`.
- config.yaml has fusion branch weights and feedback keys.
- download_data printed the real dataset features, and the field mapping is correct.
- questions.jsonl has answer_page_idx and answers for every row.
- The average number of pages per document is > 1 (otherwise it is the wrong dataset).
- `.venv` and `data/` are not tracked by git (check `git status`).

### Evidence
- Output of check_env, `download_data --limit 5` and explore_data
- The first 2 lines of questions.jsonl
- The pytest summary line

---

## P2: Parsing, text serialization, document graph (plan tasks T1–T3)

### Tasks
1. `src/docscout/ingestion/parser.py`, using the Docling DocumentConverter with OCR on.
   - Inputs:
     - Input A: a digital PDF from `data/raw/pdfs/`.
     - Input B: an MP-DocVQA document, which is a folder of page images. Combine the images into one multi-page PDF (PyMuPDF or Pillow) and parse that, so both inputs go through the same path.
   - Map Docling items to Component types:
     - paragraph: text and list items
     - heading: section headers and titles
     - table
     - figure: pictures
     - caption
   - For each item store:
     - page (1-based)
     - bbox converted to TOP-LEFT origin in page-pixel coordinates at the configured DPI
     - page_width/height
     - reading order
   - Tables: export to a pandas DataFrame and save it to `data/processed/tables/{component_id}.parquet`. Put a plain-text version in `text`.
   - Figures: crop from the rendered page image and save to `data/processed/figures/{component_id}.png`.
   - Link captions to their table/figure. Use Docling caption references first, with the nearest caption below/above as the fallback. Store the link in meta.
   - section_id = id of the most recent heading in reading order.
   - Return `list[Component]`. Never crash on one bad page; log it and continue.
2. `src/docscout/ingestion/serializer.py`: fill text_form exactly as described in CLAUDE.md, truncated to ~512 tokens.
3. `src/docscout/graph/builder.py`:
   - Build a networkx DiGraph per document: Document -> Page -> Section -> Block, sibling edges in reading order, and caption-of edges. Each edge has a "rel" attribute.
   - Save with pickle to `data/processed/graphs/{doc_id}.pkl`.
   - Helpers: load_graph, parent_section, siblings(n), caption_of, linked_component, path_to_root.
4. `scripts/ingest.py --source mpdocvqa|pdf --limit N`:
   - Run parse -> serialize -> graph.
   - Append to `data/processed/components.jsonl`, skipping documents that are already processed.
   - Print per-doc and total counts by type, and timing per page.
5. `scripts/visualize_components.py --doc DOC_ID`: draw bboxes coloured by type, with component ids, on each page. Save to `data/processed/debug/{doc_id}/page_{i}.png`.
6. Tests:
   - serializer on a hand-made table
   - graph builder on fake components
   - bbox inside page bounds

If `data/raw/pdfs/` is empty, ask me for a demo PDF before doing the PDF part.

### Acceptance criteria
- `ingest --source mpdocvqa --limit 3` and `ingest --source pdf` on one demo PDF both finish.
- Every component has a non-empty text_form, a valid page and a bbox inside the page.
- The debug overlays line up with the real paragraphs and tables. Open 2 overlay images and describe what you see.
- For one table, print its section heading, caption and neighbouring paragraphs from the graph.
- Report the average seconds per page and estimate the time for the full ~150 docs.

### Reviewer checks
- The bbox overlays align. The most common bug is a bottom-left vs top-left origin mix-up.
- Tables exist as parquet files and look like real tables, not one column of text.
- Captions are linked to the right table/figure.
- The graph has a Page level and rel attributes on its edges.
- The seconds per page are reasonable.

### Evidence
- ingest output (counts by type + timing)
- Paths of 2 debug overlay images: one MP-DocVQA page, and one demo PDF page with a table
- One components.jsonl line for a table and one for a paragraph
- The graph printout for one table
- The pytest summary

---

## P3: Indexes: BM25, dense, CLIP (plan tasks T4–T6)

### Tasks
1. A common interface for every index:
   - `build(components)`, `save(dir)`, `load(dir)`
   - `search(query, top_k, doc_id=None) -> list[tuple[component_id, float]]`
   
   doc_id filtering must be exact and efficient. Keep row ids per doc. For doc-filtered dense search, compute dot products with numpy over that doc's rows.
2. `src/docscout/indexing/bm25_index.py`: rank_bm25 BM25Okapi over text_form. The tokenizer:
   - lowercases
   - keeps numbers and words like "2023"
   - splits on punctuation
   - removes a small stopword list
3. `src/docscout/indexing/dense_index.py`:
   - sentence-transformers, with the model name from config
   - normalized embeddings in a FAISS IndexFlatIP
   - BGE query instruction prefix applied to queries only, not documents
   - batch encoding with a progress bar
   - store embeddings + id map
4. `src/docscout/indexing/visual_index.py`: CLIP clip-ViT-B-32 via sentence-transformers. Encode the figure crops, and encode text queries with the same model.
5. Branches:
   - text = paragraph, heading, caption
   - table = table
   - figure = figure
   
   For each branch build BM25 + dense(BGE) + dense(MiniLM). The figure branch also gets CLIP. Save under `data/indexes/{branch}/{retriever}/`.
6. `src/docscout/indexing/build_all.py` + `scripts/build_indexes.py`: build everything from components.jsonl, print counts and build time, and support a `--rebuild` flag.
7. `scripts/search_cli.py "query" --branch --retriever --doc --k`: print rank, score, component id, type, page, and the first 120 chars of text_form.
8. Tests:
   - a tiny synthetic corpus where BM25 must rank the keyword match first and dense must rank a paraphrase first
   - the doc filter returns only that doc

### Acceptance criteria
- build_indexes finishes on all processed docs, and the indexes load again without rebuilding.
- search_cli shows:
  - a table query finding a table via BM25
  - a paraphrased query found by BGE but not BM25
  - a "chart" query returning a figure via CLIP (on a demo PDF)
- Index sizes and build time are reported.

### Reviewer checks
- The BGE query prefix is used only for queries.
- The doc filter works: all results come from the requested document.
- The paraphrase example shows dense beating BM25.
- Indexes reload from disk, with no rebuild on every run.

### Evidence
- build_indexes output
- 3–4 search_cli outputs: BM25 table query, BM25 vs BGE on a paraphrase, CLIP figure query
- The pytest summary

---

## P4: Ground truth, metrics, first baselines (plan task T7)

### Tasks
1. `eval/build_qrels.py`:
   - Page qrels: for each question, the answer page has relevance 1. The answer page comes from answer_page_idx, mapped to our 1-based page numbers.
   - Component qrels: components on the answer page whose text (or table cell text) contains any gold answer get relevance 1.
     - Normalize first: lowercase, strip punctuation/spaces.
     - Use rapidfuzz partial_ratio >= 90 for answers longer than 3 characters, and exact match otherwise.
   - Save `qrels_page.json` and `qrels_component.json` in ranx format.
   - Print coverage = % of questions with at least one relevant component, broken down by question type.
   - Questions with no relevant component stay in the page metrics but are excluded from the component metrics. Report how many.
2. `eval/metrics.py`: wrappers around ranx for recall@1/5/10, mrr and ndcg@10. Add an ANLS function (standard DocVQA definition, threshold 0.5) with unit tests.
3. `eval/run_eval.py`:
   - A "system" is a function `(question, doc_id) -> ranked list of (component_id, score)`.
   - Build page runs from component runs (page score = max).
   - Evaluate any registered systems with `--systems` and `--limit`.
   - Save `eval/results/{system}.json` and update `eval/results/summary.md` (one row per system, with page and component metrics).
4. Register these baselines, all searching within the question's doc_id:
   - `bm25_all`: BM25 over all three branches, scores merged by max
   - `dense_minilm_all`
   - `dense_bge_all`
5. Tests for ANLS and for page-from-component aggregation.

### Acceptance criteria
- qrels are built and coverage is printed.
- summary.md has 3 rows with page and component metrics.
- The numbers look plausible: page Recall@5 is well above random for multi-page docs.

### Reviewer checks
- The page index conversion is right (0-based dataset vs 1-based pages). Show one example worked by hand.
- Component coverage is printed. Expect roughly 60–85%.
- Random-baseline sanity: page Recall@1 clearly beats 1/avg_pages. Print both numbers.
- BGE ≥ MiniLM in most metrics.

### Evidence
- build_qrels output (coverage)
- summary.md contents
- One question where BM25 failed: the question, the gold answer and the top-3 retrieved texts

---

## P5: Query classifier, normalization, fusion, reranking, full ablation (plan tasks T8–T11)

Needs `.env` with a working LLM key. If it is missing, ask me.

### Tasks
1. `src/docscout/reasoning/llm_client.py`:
   - provider-agnostic client, with groq implemented and the interface easy to swap
   - reads .env
   - JSON-mode helper
   - retries with backoff
   - on-disk cache keyed by (model, prompt), so evaluation reruns cost nothing
2. `src/docscout/query/classifier.py`:
   - `classify(query) -> {"type": textual|numerical|visual, "confidence": float, "source": rules|llm}`
   - Rules run first:
     - numbers, how much/many, total, percent, amount, date, year -> numerical
     - chart, graph, figure, diagram, image, logo, photo -> visual
   - The LLM handles the rest, or cases where the rules are unsure.
   - `weights_for(type)` reads branch weights from config.
   - Map the DocVQA question types to our 3 labels.
   - Add `eval/eval_classifier.py`, reporting accuracy and a confusion matrix for rules-only, llm-only and combined.
3. `src/docscout/retrieval/retrievers.py`: run BM25 + dense(BGE) per branch (+ CLIP for figures) within doc_id, returning candidates with the raw score per retriever.
4. `src/docscout/retrieval/normalize.py`: min-max per retriever result list (handle all-equal scores), with z-score as an option.
5. `src/docscout/retrieval/fusion.py`: `weighted_sum(candidates, retriever_weights, branch_weights)` and `rrf(k=60)`. Keep the per-stage scores on each candidate for provenance later.
6. `src/docscout/rerank/cross_encoder.py`: rerank the top-N (from config) with the cross-encoder on (query, text_form). Keep the cross-encoder score.
7. Register these systems in run_eval:
   - `hybrid_weighted_soft`
   - `hybrid_rrf_soft`
   - `hybrid_weighted_hard` (only the predicted branch)
   - `hybrid_weighted_soft_rerank`
   
   Run the full ablation and regenerate summary.md.
8. `eval/plot_results.py`: grouped bar chart (Recall@5 and MRR, page and component) for all systems, saved to `eval/results/ablation.png`. Readable labels, one colour per metric.
9. Report the median latency per query for each system.

### Acceptance criteria
- The classifier accuracy table is printed.
- summary.md shows all 7 systems: hybrid beats single retrievers, rerank improves component-level metrics, and soft >= hard routing.
- ablation.png is saved.
- If a result contradicts expectations, explain why rather than tuning on the test set. Use a 20% dev split of questions for any weight tuning.

### Reviewer checks
- Normalization happens before the weighted sum, but RRF uses ranks only.
- Hard and soft routing are both reported.
- Weights are tuned only on the dev split.
- Rerank latency is acceptable on CPU (top-30).
- LLM calls are cached: show that a rerun is fast.

### Evidence
- Classifier accuracy + confusion matrix
- Full summary.md
- ablation.png path
- Latency numbers

---

## P6: Value localization, graph context, grounded answers, provenance, pipeline (plan tasks T12–T16)

### Tasks
1. `src/docscout/localization/table_locator.py`: `locate(query, table_df) -> sub-DataFrame + list of (row_label, column, value)`.
   - Match query terms to column headers and row labels with rapidfuzz (years, units, row names).
   - Fall back to rows containing the query's numbers/keywords.
   - If nothing matches, return the whole table with a flag.
   - Only used for numerical queries on table components.
2. `src/docscout/context/graph_expander.py`: `expand(component_id, graph, max_siblings=2)` returns the context components (section heading, siblings, caption or linked figure/table) plus a path string like "Page 4 -> Section 'Revenue' -> Table 7".
3. `src/docscout/reasoning/prompts.py` and `answerer.py`:
   - Build a prompt with the question, the query type and numbered evidence [E1], [E2]... Show localized cells as a small markdown table.
   - Instruct the LLM to answer only from the evidence, cite [E#], and give a short answer for DocVQA-style questions; otherwise say "Not found in the retrieved evidence".
   - Return the answer + cited ids.
4. `src/docscout/evidence/provenance.py`: build the "why retrieved" line from:
   - query type + weights
   - per-stage scores (bm25, dense, clip, fused, rerank)
   - matched terms
   - graph path
   - page
5. `src/docscout/pipeline.py`: class `DocScoutPipeline` with `answer(query, doc_id, session_id=None, top_k=5, feedback_adjust=None) -> QueryResult`.
   - Steps, in order:
     1. classify
     2. retrieve
     3. normalize
     4. fuse
     5. rerank
     6. feedback_adjust hook (identity for now)
     7. localize (numerical)
     8. expand
     9. answer
     10. provenance
   - Load models and indexes once.
   - Also return the full reranked list (needed by "show next" in P7).
6. `scripts/ask.py "question" --doc DOC_ID`: pretty-print the answer, evidence, localized cells and why lines.
7. `eval/eval_answers.py`: ANLS on the eval subset for (a) top-1 evidence only and (b) top-3 + graph context. Add both to summary.md.
8. `tests/test_pipeline_smoke.py`: run one question end to end, with the LLM mocked.

### Acceptance criteria
- ask.py works on a numerical question (showing localized cells), a textual question and a visual question on a demo PDF.
- Every answer cites evidence or says not found.
- ANLS is reported for both settings.

### Reviewer checks
- The LLM never sees the whole document, only evidence.
- Localization returns specific cells for a numerical question.
- The why-line is readable and contains real scores and a graph path.
- The pipeline returns the full reranked list.
- ANLS (b) ≥ (a), or the difference is explained.

### Evidence
- 3 ask.py outputs (numerical, textual, visual)
- ANLS numbers
- The pytest summary

---

## P7: Backend API, session relevance feedback, demo UI (plan tasks T17–T19)

### Tasks
1. `src/docscout/feedback/session_store.py`: an in-memory dict keyed by session_id. It holds the preference counts per (query_type, component_type) and, per query_id, the reranked list + a cursor. Functions:
   - `adjust(candidates, session_id, query_type)`: adds step * count (capped; values from config) to the rerank score and re-sorts.
   - `record(session_id, query_id, component_id, liked)`
   - `next_item(session_id, query_id)`: moves the cursor and returns the next evidence, or None ("no further matches").
2. Plug `adjust()` into the pipeline's feedback_adjust hook.
3. `api/main.py` (FastAPI). Endpoints:
   - `POST /session`: returns a new uuid
   - `GET /documents`
   - `POST /ingest`: upload PDF -> parse -> graph -> add to the indexes and persist; return doc_id and counts
   - `POST /query {session_id, doc_id, query}`: returns answer, query_id and the evidence list with why/page/bbox
   - `POST /feedback {session_id, query_id, component_id, liked}`
   - `POST /next {session_id, query_id}`
   - `GET /page_image?doc_id&page&highlight=component_id`: returns a PNG with the highlighted bbox(es) (PyMuPDF render + Pillow rectangle)
   
   Load the pipeline once at startup. Turn CORS on.
4. `ui/app.py` (Streamlit, talks to the API):
   - sidebar with the session id, a document picker and PDF upload
   - question box
   - answer card
   - evidence cards with a type badge, page, snippet, localized cells table, expandable "why retrieved" and graph path, and thumbs up / thumbs down buttons (thumbs down shows the next result in place)
   - page image with the highlight for the selected evidence
   - a small "session preferences" panel showing the counts, so the effect is visible in the demo
5. A README section: how to start the API (`uvicorn api.main:app --reload`) and the UI (`streamlit run ui/app.py`).
6. Tests with the FastAPI TestClient for /session, /query (pipeline mocked), /feedback and /next.

### Acceptance criteria
- Upload a new demo PDF from the UI, ask a numerical question, and see the highlighted table and cells.
- Thumbs down replaces the result with the next one.
- After 2-3 thumbs up on tables, tables rank higher for the next numerical query (visible in the preferences panel).
- All endpoints work from /docs.

You can't click the UI yourself. Verify the API with real HTTP calls, then give me a short manual test checklist for the UI.

### Reviewer checks
- Thumbs up does not change the current screen. Thumbs down shows the next item from the same list, with no new retrieval.
- The boost is capped.
- The highlight box is in the right spot. Save one `page_image` PNG and give its path.
- Uploading a new PDF doesn't require restarting the server.

### Evidence
- Real API call outputs for /session, /query, /feedback, /next
- Path of a highlighted page image
- The TestClient pytest summary
- The manual UI checklist

---

## P8: Final evaluation, error analysis, documentation, demo prep (plan tasks T20–T21)

### Tasks
1. `eval/run_all.py`: one command that rebuilds the qrels if needed, runs every system, the classifier eval and the ANLS eval on the full subset, then regenerates summary.md and all charts.
2. `eval/error_analysis.py`: sample 5 failures of the best system per category (wrong page, right page but wrong component, wrong localization, LLM error). Write `eval/results/error_analysis.md` with the question, gold answer, retrieved items and a one-line cause for each.
3. Final README:
   - overview
   - architecture diagram (mermaid)
   - setup on Windows
   - data download, ingest, build indexes, eval
   - running the API + UI
   - results table
   - team split
4. `docs/demo_script.md`: a 4-minute demo walkthrough with 4 prepared questions (textual, numerical, visual, and one that shows feedback). Include the expected answer for each and what to say at each step.
5. `docs/slides_update.md`: the exact text changes for our slides:
   - dataset -> MP-DocVQA, and why
   - soft routing
   - normalization + RRF comparison
   - metrics, including classifier accuracy and ANLS
   - final results table
6. OPTIONAL, only if I ask: a simple login with an SQLite users table, hashed passwords, a token and a per-user document list. It must not change any other behaviour.

### Acceptance criteria
- A fresh clone + the README steps reproduces the demo. Test this in a temporary clone if feasible.
- summary.md, ablation.png and error_analysis.md are final.

### Reviewer checks
- Everything is reproducible from the README alone.
- The final numbers in summary.md match slides_update.md.
- The error analysis is honest and specific.

### Evidence
- Final summary.md
- error_analysis.md
- demo_script.md
