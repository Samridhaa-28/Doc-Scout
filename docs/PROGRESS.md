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
