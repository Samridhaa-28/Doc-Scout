## PHASE REPORT: P1 Setup, data and shared schema

### 0. Summary in plain words
- **Built:** the repo skeleton (src layout, package `docscout`, an `__init__.py` in every package of the CLAUDE.md tree), the packaging files, and `configs/config.yaml` with all the keys later phases need.
- **Installed:** everything into `.venv` (torch 2.14.1+cpu, then `requirements.txt`, then `pip install -e .`) and wrote `requirements.lock.txt`. `check_env` imports all 28 libraries and prints "all good".
- **Added:** `docscout.config` (YAML loaded once, paths resolved to the repo root, `.env` loaded) and `docscout.schema` (`Component` / `Evidence` / `QueryResult` with dict and JSONL round trips).
- **MP-DocVQA works differently from the plan's assumption.** Each question carries its own window of 1–20 pages, and rows are not grouped by document. So `download_data` does a cheap metadata pass first, then streams images only up to where the N-th document first appears.
- **Results:** `download_data --limit 5` saved 5 documents (16 pages) and 43 questions. `explore_data` prints sensible stats (mean 3.20 pages per document). `pytest`: 13 passed. The full `--limit 150` run was not started (see section 9).

### 1. Files created/changed
- `pyproject.toml`: setuptools src layout, package `docscout`, `requires-python = ">=3.10"`, pytest config
- `requirements.txt`: the libraries from CLAUDE.md, unpinned, with the note that torch CPU is installed first
- `requirements.lock.txt`: `pip freeze --exclude-editable` (163 packages, UTF-8 without BOM)
- `.gitignore`: .venv, .env, \_\_pycache\_\_, *.pyc, data/raw, data/processed, data/indexes, eval/results/*.json, .ipynb_checkpoints (also *.egg-info, .pytest_cache)
- `.env.example`: LLM_PROVIDER, LLM_API_KEY (empty), LLM_MODEL
- `README.md`: Windows setup, SSL/antivirus note, data commands, tests, layout
- `configs/config.yaml`: paths, dataset, parsing, models, branches, retrieval, fusion, classifier, feedback, llm, eval, api
- `src/docscout/__init__.py`: package docstring and `__version__`
- `src/docscout/config.py`: `get_config()`, `load_config()`, `get_path()`, `get_secret()`, `REPO_ROOT`
- `src/docscout/schema.py`: `Component`, `Evidence`, `QueryResult`, `make_component_id`, JSONL helpers
- `src/docscout/{ingestion,graph,indexing,query,retrieval,rerank,localization,context,reasoning,evidence,feedback}/__init__.py`: empty package homes
- `api/__init__.py`, `ui/__init__.py`, `eval/__init__.py`, `scripts/__init__.py`, `tests/__init__.py`: package markers so `python -m scripts.X`, `eval.X` and `api.main` resolve
- `eval/results/.gitkeep`, `docs/reports/.gitkeep`: keep the empty folders in git
- `docs/PROGRESS.md`: "Starting state" section
- `scripts/check_env.py`: imports every library, prints versions, the torch device and "all good"
- `scripts/download_data.py`: prints the real features, maps fields, runs the two streaming passes, saves images and questions, skips existing pages
- `scripts/explore_data.py`: dataset stats and consistency checks, plus one full example
- `tests/test_schema.py`, `tests/test_config.py`, `tests/test_download_utils.py`: unit tests
- `docs/reports/P1_REPORT.md`: this report
- Local only (gitignored): `data/raw/pdfs/`, `data/raw/mpdocvqa/{images/, questions.jsonl, val_index.jsonl}`, `data/processed/`, `data/indexes/`

### 2. Design decisions and any deviations from the architecture (and why)

1. **Dataset: `lmms-lab/MP-DocVQA`, split `val`.** `rubentito/mp-docvqa` contains only a README (checked through the HF API), so it can't be loaded. Real features: `questionId`, `question`, `doc_id`, `page_ids`, `answers`, `answer_page_idx`, `data_split` (all strings) and `image_1`…`image_20` (Image). There is **no question-type field**.

2. **How MP-DocVQA is structured.** Verified on the cached metadata of all 5,187 val questions:
   - Each row is one question, carrying a contiguous page window of 1–20 pages (mean 5.57). 1,633 questions (31%) have a single page.
   - Rows are **not grouped by document**: 927 documents, and in the first two shards 50 of 77 documents appear in more than one separate run of rows.
   - 147 of 927 documents have questions with **different page windows**.

   Collecting every page of a document would mean streaming nearly the whole split: the last row needed is 5145 of 5187, about 8.5 GB, even for `--limit 5`. So `download_data` works like this:
   - **Pass 1:** stream only the text columns (`load_dataset(..., streaming=True, columns=[...])`, about 2 minutes), cached to `val_index.jsonl`.
   - Select the first N documents in order of first appearance. The selection is deterministic, and `--limit 150` is a superset of `--limit 5`.
   - **Pass 2:** stream full rows only up to the row where the N-th document first appears, saving every page seen for the selected documents. A document = the union of the windows seen in that prefix.
   - Keep every question of a selected document whose **answer page was saved**.

   For `--limit 5` this kept 43 of 112 questions. Projected from the cache for `--limit 150`: 680 rows streamed, 1,060 pages, 947 of 1,178 questions kept (80%), mean 7.07 pages per document.

3. **Page naming.** `page_{i}.png` uses the **original 1-based page number**: `{doc_id}_p{k}` (0-based k) is saved as `page_{k+1}.png`. Each question keeps `answer_page_idx` exactly as in the dataset (0-based index into its own `page_ids`) and gains `answer_page_id` and `answer_page` (1-based). This removes the 0-based/1-based ambiguity before P4, and the page numbers stay stable when a rerun adds pages. Side effect: page numbers within a document can have gaps (for example, `pybv0228` has only `page_81.png`), and `meta.json` lists them. Extra fields `answer_page_id` and `answer_page` were added to `questions.jsonl`; `num_pages` was not added because it differs per question.

4. **Skipping existing data:** skipping is per page (files already on disk aren't re-downloaded), and `questions.jsonl` is rewritten deterministically on every run, so duplicates can't happen.

5. **Config:**
   - `get_config()` returns a deep copy of a cached dict, and paths are `pathlib.Path` objects resolved to the repo root.
   - `LLM_PROVIDER` and `LLM_MODEL` from `.env` override the YAML values.
   - `LLM_API_KEY` is only available through `get_secret()` and is never stored in the config dict.
   - Branch weights sum to 1 for each query type.
   - `parsing.image_dpi = 144` is a starting value for P2.

6. **Schema:**
   - `Component.__post_init__` rejects unknown types and makes `bbox` a tuple of floats.
   - `Evidence` and `QueryResult` have an extra `meta: dict` (for graph_path, query_id, reranked list and similar) so later phases don't need schema changes.
   - `Evidence.localized_cells` is a list of `(row_label, column, value)` tuples, restored as tuples after a JSON round trip.

7. **Environment issue: Avast antivirus HTTPS scanning.** Avast re-signs all TLS connections with "Avast Web/Mail Shield Root", which is trusted in the Windows store but not in `certifi`.
   - The pip that came with the venv (21.2.3) failed. I upgraded it once using `--cert` with a combined certifi + Windows-roots bundle built in my scratchpad (outside the repo). pip 26.2.1 uses the Windows store by default, and every later install used plain pip.
   - Hugging Face (httpx) also uses `certifi`, so the download commands ran with `$env:SSL_CERT_FILE` pointing at that bundle.
   - No repo code was changed for this, and verification was never turned off. The README documents the workaround.

8. **git:** before P1 the repo had no commits, and about 1,100 `.venv` files were **staged in the index**. I ran `git rm -r --cached .venv`, which changes only the index; no files were deleted from disk. Nothing was committed.

### 3. How to run (exact commands, in order, as a teammate would on a fresh clone)
```powershell
py -3.10 -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m scripts.check_env
copy .env.example .env          # fill LLM_API_KEY before P5
# only if behind HTTPS-scanning antivirus: $env:SSL_CERT_FILE = "C:\path\to\ca-bundle.pem"
.venv/Scripts/python -m scripts.download_data --limit 5
.venv/Scripts/python -m scripts.explore_data
.venv/Scripts/python -m pytest
```

### 4. Actual output (real terminal output from this session, trimmed to the important lines)

**check_env**
```
python 3.10.0 (C:\Users\Samridhaa\OneDrive\Desktop\Doc-Scout\.venv\Scripts\python.exe)
  OK   torch                    2.14.1+cpu
  OK   torchvision              0.29.1+cpu
  OK   docling                  2.140.0
  OK   pymupdf                  1.28.2
  OK   pillow                   12.3.0
  OK   numpy                    2.2.6
  OK   pandas                   2.3.3
  OK   networkx                 3.4.2
  OK   datasets                 5.1.0
  OK   rapidfuzz                3.14.5
  OK   rank-bm25                0.2.2
  OK   sentence-transformers    6.1.0
  OK   faiss-cpu                1.15.1
  OK   groq                     1.7.0
  OK   python-dotenv            1.2.4
  OK   fastapi                  0.143.3
  OK   uvicorn                  0.54.0
  OK   python-multipart         0.0.32
  OK   pydantic                 2.14.1
  OK   requests                 2.34.2
  OK   streamlit                1.65.0
  OK   ranx                     0.3.21
  OK   scikit-learn             1.7.2
  OK   matplotlib               3.10.9
  OK   tqdm                     4.70.1
  OK   pyyaml                   6.0.3
  OK   pytest                   9.1.1
  OK   docscout                 0.1.0
torch device: cpu (cuda available: False, threads: 8)
config loaded: repo_root=C:\Users\Samridhaa\OneDrive\Desktop\Doc-Scout, 13 top-level keys
all good
```
(The first run printed `warning: The fitz API is deprecated`. The check now imports `pymupdf`, and the re-run printed `OK pymupdf 1.28.2` ... `all good` with no warning.)

`pip check`: `No broken requirements found.`

**download_data --limit 5** (first run, 176 s in total)
```
Loading lmms-lab/MP-DocVQA split=val (streaming=True)

=== Dataset features (real) ===
  questionId: Value('string')
  question: Value('string')
  doc_id: Value('string')
  page_ids: Value('string')
  answers: Value('string')
  answer_page_idx: Value('string')
  data_split: Value('string')
  image_1: Image(mode=None, decode=True)
  ... (image_2 .. image_19 identical)
  image_20: Image(mode=None, decode=True)

=== Field mapping (target <- source) ===
  question_id      <- questionId
  question         <- question
  doc_id           <- doc_id
  page_ids         <- page_ids
  answer_page_idx  <- answer_page_idx
  answers          <- answers
  question_types   <- (not in dataset)
  page images      <- image_1 .. image_20 (20 columns), aligned with page_ids
metadata pass (no images): 5187row [02:09, 40.19row/s]
Metadata pass: 5187 rows cached to val_index.jsonl

Split has 5187 questions over 927 documents.
Selected first 5 documents; image pass streams rows 0..7.
Resume: 0/5 docs already complete on disk; 16 pages to download.
image pass: 100%|██████████| 8/8 [00:30<00:00,  3.77s/row]

New pages saved: 16. Docs: 5 | pages on disk: 16
Questions written: 43 -> data\raw\mpdocvqa\questions.jsonl
Questions dropped: {'answer page not saved': 69}
Note: dataset has no question-type field; question_types not written.
```
Rerun, to check that saved data is skipped (19 s):
```
Resume: 5/5 docs already complete on disk; 0 pages to download.
New pages saved: 0. Docs: 5 | pages on disk: 16
Questions written: 43 -> data\raw\mpdocvqa\questions.jsonl
```

**explore_data**
```
=== MP-DocVQA subset ===
documents (with images): 5
documents referenced by questions: 5
pages per doc: min=1 mean=3.20 max=12 (total pages: 16)
docs where png count != num_pages: 0
questions: 43 (per doc: 8.60)
questions missing answer_page_idx or answers: 0
questions with answer_page_idx out of range: 0
questions whose answer_page_id != page_ids[answer_page_idx]: 0
questions whose answer page image is missing: 0
question page window (len(page_ids)): min=1 mean=6.12 max=12

answer_page_idx distribution (0-based index into the question's page_ids -> count):
   0:   23 #######################
   1:    2 ##
   2:    2 ##
   3:    5 #####
   7:    4 ####
   9:    3 ###
  10:    4 ####
answer page position within the saved document (1 = first saved page -> count):
   1:   23 ...   2: 2   3: 2   4: 5   8: 4   10: 3   11: 4
questions on single-page docs (page retrieval trivial): 23/43

question types:
  not available in this dataset (no question_types field)

=== One full example ===
{ "question_id": "49153", "question": "What is the ‘actual’ value per 1000, during the year 1975?",
  "doc_id": "pybv0228", "page_ids": ["pybv0228_p80"], "answer_page_idx": 0,
  "answer_page_id": "pybv0228_p80", "answer_page": 81, "answers": ["0.28"] }
doc meta: num_pages=1 page_numbers=[81] page_sizes=[[2257, 1764]]
answer page: page_ids[0] = pybv0228_p80 -> page 81 -> data\raw\mpdocvqa\images\pybv0228\page_81.png exists=True
```

**First 2 lines of questions.jsonl**
```
{"question_id": "49153", "question": "What is the ‘actual’ value per 1000, during the year 1975?", "doc_id": "pybv0228", "page_ids": ["pybv0228_p80"], "answer_page_idx": 0, "answer_page_id": "pybv0228_p80", "answer_page": 81, "answers": ["0.28"]}
{"question_id": "24580", "question": "What is name of university?", "doc_id": "nkbl0226", "page_ids": ["nkbl0226_p0"], "answer_page_idx": 0, "answer_page_id": "nkbl0226_p0", "answer_page": 1, "answers": ["University of California", "university of california", "university of california, san diego"]}
```

**Projection for `--limit 150`**, computed from the cached `val_index.jsonl` (no download):
```
limit150 projection: rows streamed 680 / 5187 | pages 1060 | pages/doc mean 7.07 | single-page docs 41
questions kept 947 | on single-page docs 172 | on docs >=2 pages 775
1/avg_pages over kept questions (random page R@1): 0.357
```

Data on disk: `data/raw/mpdocvqa` = 44 MB (5 documents, plus the 5,187-row metadata cache).

### 5. Tests
`13 passed in 0.59s`
- `test_schema.py::test_component_id_format`: the id is `"{doc_id}_p{page}_{type}_{n}"`
- `test_schema.py::test_unknown_type_rejected`: an invalid component type raises `ValueError`
- `test_schema.py::test_component_dict_round_trip`: `Component.to_dict` and `from_dict` give an equal object, and bbox comes back as a tuple
- `test_schema.py::test_components_jsonl_round_trip`: write, append and read JSONL gives identical Components, with non-ASCII text kept as UTF-8
- `test_schema.py::test_query_result_round_trip`: `QueryResult` → `Evidence` → context and localized cells survive a JSONL round trip
- `test_config.py::test_repo_root_is_project`: `REPO_ROOT` points at the repo
- `test_config.py::test_paths_resolved_relative_to_repo_root`: every path is an absolute `Path` under the repo root
- `test_config.py::test_required_keys_present`: ocr/dpi, the 5 model keys, `rerank_top_n == 30`, fusion method, branch weights for all 3 types (each summing to 1), feedback step/cap, llm keys, eval k and subset
- `test_config.py::test_get_config_returns_copy`: changing the returned dict doesn't change the cached config
- `test_config.py::test_load_config_custom_file`: a custom YAML file and root resolve correctly
- `test_download_utils.py::test_parse_listish_string_and_native`: parses string-encoded lists from the dataset
- `test_download_utils.py::test_page_number_is_one_based`: `_p0` maps to 1 and `_p80` to 81
- `test_download_utils.py::test_resolve_fields_real_names`: maps the real feature names, and a missing required field fails

### 6. Acceptance criteria and reviewer checks
- PASS: `pip install -e .` works. Output: `editable install OK`; check_env shows `OK docscout 0.1.0`.
- PASS: check_env prints "all good". Last line: `all good`; device `cpu`.
- PASS: `download_data --limit 5` saves 5 documents and their questions. Output: `Docs: 5 | pages on disk: 16`, `Questions written: 43`.
- NOT RUN (as instructed): the full download. Command: `.venv/Scripts/python -m scripts.download_data --limit 150` (see section 9).
- PASS: explore_data prints sensible stats. Mean 3.20 pages per document, 43 questions, 0 inconsistencies.
- PASS: pytest passes. `13 passed in 0.59s`.
- PASS: the folder tree matches CLAUDE.md and every package has `__init__.py`. All 11 `src/docscout/*` subpackages plus api, ui, eval, scripts and tests have one (only `src/`, the layout root, and the gitignored `docscout.egg-info` don't). Module files for later phases (parser.py and so on) are created in their own phases.
- PASS: config.yaml has fusion branch weights and feedback keys. It has `fusion.branch_weights.{textual,numerical,visual}` and `feedback.step=0.05, cap=0.2`, checked by `test_required_keys_present`.
- PASS: download_data printed the real dataset features, and the field mapping is correct. See the "Dataset features (real)" and "Field mapping" blocks above; `answer_page_id == page_ids[answer_page_idx]` for 43/43 questions.
- PASS: questions.jsonl has answer_page_idx and answers in every row. Check output: `43 0` (rows, missing).
- PASS: average pages per document > 1. Mean 3.20 for the 5-document sample; 7.07 projected for 150.
- PASS: `.venv` and `data/` are not tracked by git. `git status --short` lists only source files; `git check-ignore` confirms `.gitignore:2:.venv/` and `.gitignore:12:data/raw/`.

### 7. Interfaces for the next phase
- **Config:** `from docscout.config import get_config, get_path, get_secret, REPO_ROOT`
  - `get_config() -> dict` (deep copy). `cfg["paths"][key]` is an absolute `Path`.
  - Keys: `paths.{mpdocvqa_dir, mpdocvqa_images_dir, questions_file, pdfs_dir, processed_dir, components_file, tables_dir, figures_dir, graphs_dir, debug_dir, indexes_dir, results_dir}`, `parsing.{ocr, image_dpi, max_text_form_tokens}`, `models.*`, `branches.*`, `retrieval.{top_k.*, rerank_top_n, final_top_k}`, `fusion.{method, normalization, rrf_k, retriever_weights, branch_weights}`, `feedback.{step, cap}`, `llm.{provider, model, temperature, max_tokens, max_retries, cache_path}`, `eval.{k_values, metrics, subset_size, dev_fraction, anls_threshold, fuzzy_match_threshold, seed}`.
- **Schema:** `from docscout.schema import Component, Evidence, QueryResult, make_component_id, write_components, read_components, write_jsonl, read_jsonl, iter_jsonl, COMPONENT_TYPES, QUERY_TYPES`
  - `Component(id, doc_id, page, type, text, text_form, bbox, page_width, page_height, order, section_id, table_path, image_path, meta)`
  - `make_component_id(doc_id, page, ctype, n) -> "{doc_id}_p{page}_{type}_{n}"`
- **MP-DocVQA on disk (input for P2):**
  - `images/{doc_id}/page_{n}.png`, where `n` is the original 1-based page number. **Not contiguous**: always read `meta.json` and don't assume pages 1..N.
  - `images/{doc_id}/meta.json` = `{doc_id, page_ids, page_numbers, page_sizes:[[w,h],...], num_pages, source}`, sorted by page number.
  - When P2 joins the images into one PDF, PDF page j (0-based) corresponds to `page_numbers[j]`, and `Component.page` must be set to that number so it matches `answer_page`.
- **questions.jsonl:** `{question_id, question, doc_id, page_ids, answer_page_idx (0-based into page_ids), answer_page_id, answer_page (1-based, equal to Component.page), answers: [str]}`.
- **val_index.jsonl:** metadata for all 5,187 val questions (same fields plus `row`), used as a cache.

### 8. Known limitations / open questions
1. **Questions dropped:** 20% of the selected documents' questions (231 of 1,178 projected for 150 documents) have answer pages outside the streamed prefix. Keeping them would require streaming most of the 8.5 GB split.
2. **Single-page documents:** in the 150 subset, 41 documents and 172 of 947 questions (projected) have only one saved page, so page-level retrieval is trivial for them. Options: keep them (component level still matters), report metrics on documents with ≥2 pages separately, or add `--min-pages 2` to the selection. Your call before P4.
3. **Retrieval scope:** "Search within the question's document" here means all saved pages of the document, which can be more than that question's original `page_ids` window. This is harder than the official MP-DocVQA setting; restricting to `page_ids` would match the benchmark. Your call before P4.
4. **No question types:** `lmms-lab/MP-DocVQA` has no question types, but P5 needs them for classifier accuracy. MP-DocVQA was built from SP-DocVQA, whose val set does have `question_types` (in `lmms-lab/DocVQA`). Joining on `questionId` is a likely fix in P5; not verified yet.
5. **Antivirus HTTPS scanning (Avast) on this machine:** HF downloads need `SSL_CERT_FILE` pointing at a bundle that includes the Windows roots. Teammates without HTTPS-scanning antivirus don't need this.
6. **Python 3.10.0** is an early patch release. Everything installed and imported fine, but 3.10.11+ would be safer long term.
7. The Hugging Face cache warns that symlinks aren't supported (Windows without Developer Mode). It still works, just with more disk use; set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to hide the warning.

### 9. What I need from you before the next phase
1. **Start the full download yourself** (my estimate, not measured: 5–15 minutes, about 1.1 GB streamed, about 1,060 pages written; the metadata cache already exists):
   ```powershell
   .venv/Scripts/python -m scripts.download_data --limit 150
   .venv/Scripts/python -m scripts.explore_data
   ```
   On this machine, run `$env:SSL_CERT_FILE = "<path to a certifi + Windows-roots bundle>"` first because of Avast. Alternatively, pause Avast's HTTPS scanning, or ask me to run it.
2. **Decide on open questions 2 and 3** (single-page documents; retrieval scope = saved document vs the question's own window).
3. **Add 3–5 demo PDFs** to `data/raw/pdfs/`. P2 needs at least one, ideally a long one with tables and figures.
4. If you're happy with P1, reply "P1 approved" with a commit message, and I'll commit (nothing has been committed yet) and append the summary to `docs/PROGRESS.md`.
