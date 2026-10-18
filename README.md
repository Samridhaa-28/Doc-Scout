# DocScout-X

Query-aware, fine-grained evidence retrieval from long documents. Given a question, DocScout-X returns the exact paragraph, table (or table cells) or figure that answers it, with provenance and a "why retrieved" explanation, rather than a whole PDF or page.

Build phases are defined in [docs/PHASES.md](docs/PHASES.md); progress is tracked in [docs/PROGRESS.md](docs/PROGRESS.md).

## Setup (Windows)

Requirements: Python 3.10, about 5 GB of free disk space (torch, docling, Hugging Face model cache). CPU only.

Run all commands from the repo root. They call the venv interpreter directly, so they work in PowerShell, cmd and Git Bash.

```powershell
# 1. Create the virtual environment (skip if .venv already exists)
py -3.10 -m venv .venv

# 2. Upgrade pip (pip >= 24.2 uses the Windows certificate store, see "SSL errors" below)
.venv/Scripts/python -m pip install --upgrade pip

# 3. Install torch CPU build FIRST
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install the project libraries
.venv/Scripts/python -m pip install -r requirements.txt

# 5. Install the docscout package (editable, src layout)
.venv/Scripts/python -m pip install -e .

# 6. Check the environment (must end with "all good")
.venv/Scripts/python -m scripts.check_env

# 7. Secrets: copy the template and fill in your LLM key (needed from Phase 5)
copy .env.example .env
```

`requirements.lock.txt` records the exact versions we tested with (`pip freeze`). To reproduce them exactly, replace steps 3–4 with `.venv/Scripts/python -m pip install -r requirements.lock.txt`. Its first line adds the PyTorch CPU index, so the `torch==…+cpu` pins resolve. Then run step 5.

### SSL errors (antivirus HTTPS scanning)
`CERTIFICATE_VERIFY_FAILED` is usually caused by antivirus HTTPS scanning (for example Avast Web Shield), which re-signs traffic with its own root certificate. That root is in the Windows store but not in Python's `certifi` bundle.
- pip >= 24.2 trusts the Windows store, so pip works once it is upgraded. The pip that ships with Python 3.10 (21.x) does not: for step 2 only, pass `--cert` with a bundle that includes the Windows roots, or briefly pause HTTPS scanning.
- Hugging Face downloads (`datasets`, `huggingface_hub` and `sentence-transformers`, which go through httpx) use `certifi`. Point them at a bundle that includes the Windows roots for the session: `$env:SSL_CERT_FILE = "C:\path\to\ca-bundle.pem"`. The bundle is certifi plus `ssl.enum_certificates("ROOT")` exported as PEM.

Never turn off certificate verification.

## Data

```powershell
# Small test download (5 documents)
.venv/Scripts/python -m scripts.download_data --limit 5

# Full evaluation subset (~150 documents, long run)
.venv/Scripts/python -m scripts.download_data --limit 150

# Stats
.venv/Scripts/python -m scripts.explore_data
```

- Source: `lmms-lab/MP-DocVQA`, `val` split (5,187 questions, 927 documents), read with Hugging Face streaming so the full split is never downloaded.
  - Each question row carries a page window of 1–20 pages.
  - Rows are not grouped by document.
- How the script works:
  1. Caches the text columns of the whole split (`val_index.jsonl`, about 2 minutes).
  2. Takes the first N documents in order of appearance.
  3. Streams full rows only until the N-th document first appears.
  4. Keeps every question whose answer page was saved.
- Page images: `data/raw/mpdocvqa/images/{doc_id}/page_{i}.png`, where `i` is the original 1-based page number (`{doc_id}_p{k}` maps to `page_{k+1}.png`). Page numbers can therefore have gaps. `meta.json` lists the saved pages.
- Questions: `data/raw/mpdocvqa/questions.jsonl`. `answer_page_idx` is kept as in the dataset (0-based index into that question's `page_ids`). `answer_page` is the 1-based page number, which is the number to use for evaluation.
- Reruns skip pages already on disk.
- Demo PDFs go in `data/raw/pdfs/`.

Everything under `data/` is gitignored.

## Ingest (parse, serialize, graph)

```powershell
# a few chosen MP-DocVQA documents
.venv/Scripts/python -m scripts.ingest --source mpdocvqa --docs jjvg0227 yxvw0217 ttwl0228
# all downloaded MP-DocVQA documents (long run: about 18 s per page on CPU with onnxruntime, about 5 h for 1,060 pages; skips documents already done)
.venv/Scripts/python -m scripts.ingest --source mpdocvqa
# demo PDFs in data/raw/pdfs/
.venv/Scripts/python -m scripts.ingest --source pdf
# debug overlays -> data/processed/debug/{doc_id}/page_{n}.png
.venv/Scripts/python -m scripts.visualize_components --doc jjvg0227
```

Outputs:
- `data/processed/components.jsonl`
- `data/processed/tables/*.parquet`
- `data/processed/figures/*.png`
- `data/processed/graphs/{doc_id}.pkl`
- `data/processed/docs/{doc_id}.json` (page manifest)

Re-run with `--force` to re-parse documents, or with `--reserialize` to recompute only `text_form` and the graphs. The first run downloads the Docling layout/table models (HF cache) and the RapidOCR models (into the rapidocr package folder in `.venv`).

## Indexes (BM25, dense BGE / MiniLM, CLIP)

```powershell
.venv/Scripts/python -m scripts.build_indexes            # builds missing indexes, loads up-to-date ones
.venv/Scripts/python -m scripts.build_indexes --rebuild  # required after more documents are ingested
.venv/Scripts/python -m scripts.search_cli "population in poverty" --branch table --retriever bm25 --doc jjvg0227
.venv/Scripts/python -m scripts.search_cli "chart" --branch figure --retriever clip --k 3
```

- Indexes are stored in `data/indexes/{text,table,figure}/{bm25,dense_bge,dense_minilm,clip}/`.
- Each index stores a fingerprint of `components.jsonl`. When the file changes, both scripts warn that the indexes are stale.
- Models load from the HF cache without network access once they have been downloaded.

## Evaluation (within-document retrieval)

```powershell
.venv/Scripts/python -m eval.select_eval_docs --target-pages 250 --max-pages 20   # writes data/raw/mpdocvqa/eval_docs.txt
.venv/Scripts/python -m scripts.ingest --source mpdocvqa --docs-file data/raw/mpdocvqa/eval_docs.txt --max-minutes 12   # repeat until 49/49
.venv/Scripts/python -m scripts.build_indexes --rebuild
.venv/Scripts/python -m eval.build_qrels
.venv/Scripts/python -m eval.run_eval --systems bm25_all dense_minilm_all dense_bge_all
```

Results are written to `eval/results/summary.md`. The per-system JSON files are gitignored.

## Tests

```powershell
.venv/Scripts/python -m pytest
```

## Project layout

```
configs/config.yaml        all settings (paths, models, parsing backends, retrieval, fusion, feedback, llm, eval)
src/docscout/              the package: all reusable logic
  config.py, schema.py, env_check.py
  dataset/                 MP-DocVQA download (mpdocvqa.py) and subset stats (explore.py)
  ingestion/               parser (Docling), serializer, runner (crash-isolated ingest), validate, visualize
  graph/                   document-structure graph
  indexing/                BM25 / dense / CLIP indexes, build_all (build, staleness, report), search
  query/ retrieval/ rerank/ localization/ context/ reasoning/ evidence/ feedback/   (later phases)
scripts/                   thin command-line entry points only (argparse + one call): python -m scripts.NAME
eval/                      evaluation-only code: select_eval_docs, build_qrels, metrics, run_eval, results/
api/, ui/                  FastAPI backend and Streamlit UI (later phases)
tests/                     pytest
```
