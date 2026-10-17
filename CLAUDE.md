# DocScout-X: instructions for Claude Code

## How we work (read this first)
We build in 8 phases (P1–P8), defined in `docs/PHASES.md`. The team reviews each phase before the next one starts.

1. Do ONLY the phase I name. Do not start the next phase, or do tasks from a later phase, until I write "P<n> approved".
2. Before writing code, read `docs/PROGRESS.md` (what is already done, if it exists) and the phase's section in `docs/PHASES.md`. Then post a short plan listing the files you will create or change and the commands you will run. Carry on without waiting unless something is ambiguous.
3. Create the files and run the commands yourself, and fix errors yourself. If the same error is still there after 3 attempts, stop and report it. Do not work around the architecture.
4. Verify by actually running things. Never write "should work" and never invent output. Every result in the report must come from a command you ran in this session. If you could not run something, say so.
5. Ask me before you:
   - start any run likely to take more than about 10 minutes (for example the full 150-doc download or the full ingest; run the `--limit` version and give me the full command)
   - delete files
   - change files from an earlier phase beyond small fixes
   - add a library that is not in the list below
   - do anything that needs a secret
6. Don't commit or push during a phase. When I approve a phase, commit with the message I give (or `phase-<n>: <short summary>`) and append the phase summary to `docs/PROGRESS.md`. Never push unless I ask.
7. Never write API keys into any file except `.env`, and never print them. If `.env` or a key is missing, ask me.

## Environment
- Windows. Python 3.10 venv at `.venv` in the repo root. Code must be Python 3.10 compatible (no `tomllib`, `typing.Self` or other 3.11+ features).
- Always call the venv interpreter explicitly so commands work in any shell:
  - `.venv/Scripts/python -m pip ...`
  - `.venv/Scripts/python -m scripts.NAME`
  - `.venv/Scripts/python -m pytest`
- CPU only, no GPU. Install packages in this order:
  1. torch: `.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
  2. `requirements.txt`
  3. `pip install -e .`
- Hugging Face models download into the HF cache on first use, never into the repo.

## End-of-phase report
When a phase is finished, or you are blocked, write `docs/reports/P<n>_REPORT.md` and also print it in full in the chat. Use exactly these headings:

```
## PHASE REPORT: P<n> <phase name>
0. Summary in plain words (5-8 lines: what was built, what it does, does it work)
1. Files created/changed (path - one line each)
2. Design decisions and any deviations from the architecture (and why)
3. How to run (exact commands, in order, as a teammate would on a fresh clone)
4. Actual output (real terminal output from this session, trimmed to the important lines, plus small samples of the files produced)
5. Tests (pytest summary line and what each new test checks)
6. Acceptance criteria and reviewer checks (one line each: PASS / FAIL / NOT RUN + evidence: a number, an output line or a file path)
7. Interfaces for the next phase (function signatures, file formats, config keys)
8. Known limitations / open questions
9. What I need from you before the next phase (files to add, keys, long runs to start, things to look at such as debug images, with their paths)
```

Then stop and wait.

---

## PROJECT CONTEXT

You are helping a 3-member final-year team build an Information Retrieval course project in 4 weeks.

### What the system does
DocScout-X is query-aware, fine-grained evidence retrieval from long documents. It does not return a whole PDF or page. It returns the exact paragraph, table (or table cells) or figure that answers a question, with provenance and a "why retrieved" explanation.

### Libraries (already chosen)
- Parsing: docling, pymupdf, pillow
- Data and graph: numpy, pandas, networkx, datasets, rapidfuzz
- Retrieval: rank-bm25, sentence-transformers, faiss-cpu
- LLM: groq (LLM provider; keep it swappable), python-dotenv
- Backend and UI: fastapi, uvicorn[standard], python-multipart, pydantic, requests, streamlit
- Evaluation and dev: ranx, scikit-learn, matplotlib, tqdm, pyyaml, pytest
- torch is installed as the CPU build.

The repo uses a src layout installed with `pip install -e .`. The package name is `docscout`.

### Final architecture

**Offline (once per document)**
1. Parse with Docling (OCR + layout). Components have type paragraph / heading / table / figure / caption. Each has doc_id, page (1-based), bbox (x0,y0,x1,y1 in page pixels, top-left origin), page width/height and reading order.
2. Text serialization (`text_form`):
   - paragraph: raw text
   - table: title/caption + column headers + row headers + first 3 rows
   - figure: caption + nearest paragraph
3. Document-structure graph (networkx): Document -> Page -> Section (from headings) -> Block. Edges: parent-child, sibling (reading order), caption-of.
4. Three branch indexes: text (paragraph, heading, caption), table, figure.
   - Each has BM25 + dense (bge-base-en-v1.5 primary, all-MiniLM-L6-v2 for comparison) using FAISS.
   - The figure branch also has CLIP (clip-ViT-B-32) image embeddings.

**Online (per query)**
5. Query-type classifier (keyword rules + LLM) labels the query textual / numerical / visual, which sets branch weights. This is SOFT routing: all branches are always searched and the label only sets weights.
6. Per-retriever min-max score normalization, then weighted-sum fusion (main method). Reciprocal Rank Fusion is the comparison method.
7. Cross-encoder rerank of the top 30 (cross-encoder/ms-marco-MiniLM-L-6-v2) on (query, text_form), then session feedback adjustment.
8. Structured value localization (numerical queries only): pandas selects the matching rows/columns/cells of the top table.
9. Graph context expansion: parent section heading, siblings, caption or linked figure.
10. The LLM reasons ONLY over retrieved evidence. It must cite evidence IDs and must say "Not found in the retrieved evidence" if the evidence does not support an answer.
11. Evidence + provenance: page, type, snippet, bbox, scores per stage, graph path and a one-line "why retrieved".
12. Relevance feedback per session:
    - Thumbs up: preference[(query_type, component_type)] += 1; the screen does not change.
    - Thumbs down: -= 1, and the next item from the same reranked list is shown immediately.
    - Preferences give a small capped boost/penalty after reranking for later queries in the session.
    - No login; anonymous session ID.

### Dataset and evaluation
- Data: MP-DocVQA (multi-page DocVQA) validation subset (~150 documents) from Hugging Face (`rubentito/mp-docvqa` or `lmms-lab/MP-DocVQA`), in `data/raw/mpdocvqa/`. Plus 3-5 real long PDFs for the demo in `data/raw/pdfs/` (the team adds these).
- Retrieval setting for evaluation: search WITHIN the question's document, at page level and component level. Page score = max score of its components.
- Metrics:
  - Recall@1/5/10, MRR and nDCG@10 (ranx) at page and component level
  - classifier accuracy
  - ANLS for final answers
- Ablations: BM25, MiniLM, BGE, Hybrid weighted, Hybrid RRF, Hybrid + rerank, hard vs soft routing.

### Folder structure
```
Doc-Scout/
  CLAUDE.md, README.md, pyproject.toml, requirements.txt, requirements.lock.txt, .env.example, .gitignore
  configs/config.yaml
  docs/PHASES.md, docs/PROGRESS.md, docs/reports/
  data/raw/  data/processed/  data/indexes/        (gitignored)
  src/docscout/
    config.py, schema.py, pipeline.py
    ingestion/parser.py, ingestion/serializer.py
    graph/builder.py
    indexing/bm25_index.py, dense_index.py, visual_index.py, build_all.py
    query/classifier.py
    retrieval/retrievers.py, normalize.py, fusion.py
    rerank/cross_encoder.py
    localization/table_locator.py
    context/graph_expander.py
    reasoning/llm_client.py, prompts.py, answerer.py
    evidence/provenance.py
    feedback/session_store.py
  api/main.py
  ui/app.py
  eval/build_qrels.py, metrics.py, run_eval.py, results/
  scripts/download_data.py, ingest.py, (others as needed)
  tests/
```

### Coding rules
- Use type hints and short docstrings.
- Read all settings from `configs/config.yaml` via `docscout.config`, and secrets from `.env`.
- Use pathlib. No absolute paths.
- Scripts:
  - run from the repo root as `python -m scripts.NAME`
  - use argparse, with a `--limit` option where it helps
  - use tqdm for long loops
  - cache results so reruns are fast
- Keep to the scope of the current phase. If you must change a file from an earlier phase, say why in the report.
- Add or extend pytest tests for what you build. Keep tests fast; mock the LLM and avoid downloading big models in unit tests where possible.
