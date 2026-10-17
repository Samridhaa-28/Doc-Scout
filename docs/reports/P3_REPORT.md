## PHASE REPORT: P3 Indexes: BM25, dense, CLIP

### 0. Summary in plain words
- **Built:** three branch indexes over `components.jsonl` (text = paragraph/heading/caption, table, figure), each with BM25, dense BGE and dense MiniLM (FAISS). The figure branch also has a CLIP image index over the figure crops.
- **Common interface:** every index has `build / save / load / search(query, top_k, doc_id)`. The doc filter is exact: per-document row lists, BM25 scoring of only those rows, and numpy dot products over only those rows for dense search.
- **Fingerprints:** each index stores a fingerprint of `components.jsonl` (line count + sha256). After more documents are ingested, `build_indexes` and `search_cli` warn that the indexes are stale and ask for `--rebuild`. This was tested for real by ingesting a 4th document.
- **Demos:**
  - a BM25 table query finds the right table;
  - a paraphrase that shares no words with its target is found 1st by BGE and not retrieved by BM25;
  - "chart" ranks a real chart first with CLIP.
- **Results:** indexes reload in milliseconds, 36 tests pass, and all indexes take 1.1 MB on disk.
- **Demo PDF:** skipped, because `data/raw/pdfs/` is still empty, so the CLIP check used MP-DocVQA figures as instructed.
- **Before P3:** P2 was committed as `e0b2676` and pushed.

### 1. Files created/changed
**Step 1** (committed in `e0b2676 phase-2: parsing, serialization, graph`, pushed):
- `docs/PROGRESS.md`: "P2 (approved for MP-DocVQA)" section

**P3** (not committed):
- `src/docscout/indexing/base.py`: `BaseIndex` interface, per-document rows, deterministic top-k, `file_fingerprint()`, `Timer`
- `src/docscout/indexing/bm25_index.py`: tokenizer + `BM25Index` (rank_bm25 BM25Okapi)
- `src/docscout/indexing/dense_index.py`: `DenseIndex` (sentence-transformers, normalized, FAISS IndexFlatIP, BGE instruction on queries only), cache-first model loading
- `src/docscout/indexing/visual_index.py`: `ClipIndex` (clip-ViT-B-32 image embeddings of figure crops; text queries through the same model)
- `src/docscout/indexing/build_all.py`: branch/retriever layout, build/load, fingerprint, stale detection
- `scripts/build_indexes.py`: builds missing indexes, loads up-to-date ones, `--rebuild`; prints counts, times and sizes; exits 2 on stale indexes
- `scripts/search_cli.py`: `"query" --branch --retriever --doc --k --target --show-encoded`
- `tests/test_indexing.py`: 8 tests
- `README.md` (P1 file, small addition): "Indexes" section
- `docs/reports/P3_REPORT.md`: this report

`base.py` is a new module that isn't in the CLAUDE.md tree. It holds the shared interface used by the three index modules.

### 2. Design decisions and any deviations from the architecture (and why)
1. **Interface:**
   - `build(components)`, `save(dir)`, `load(dir)` (classmethod), `search(query, top_k, doc_id=None) -> [(component_id, score)]`.
   - Each index keeps `ids`, `doc_ids` and `doc_rows[doc_id] -> np.array(rows)`.
   - Doc-filtered search touches only that document's rows: `BM25Okapi.get_batch_scores(q, rows)` for BM25, `emb[rows] @ q` for dense and CLIP.
   - Unfiltered dense search uses FAISS `IndexFlatIP`.
   - Ties are broken by row order, so results are deterministic.
2. **BM25 tokenizer:**
   - lowercase;
   - tokens are numbers with internal separators (`26.9`, `1,234.5`; commas removed, so `10,700` → `10700`) or alphanumeric runs (`2023`, `co2`);
   - split on all other punctuation;
   - a small English stopword list.

   BM25Okapi uses k1 = 1.5, b = 0.75. Rows that match no query term (score 0) are not returned. IDF is computed over the whole branch (all documents), and the doc filter only restricts which rows are scored.
3. **Dense:**
   - Embeddings are normalized (cosine = dot product) and stored both as `embeddings.npy` (for doc-filtered numpy search) and in `faiss.index`.
   - The BGE instruction (`models.bge_query_instruction`) is applied only when the model is the configured BGE model, and only in `query_text()`.
   - `doc_text()` returns the plain `text_form`.
   - The instruction actually used is saved in each index's meta.json (MiniLM: `''`).
4. **CLIP:** `clip-ViT-B-32` through sentence-transformers. Documents are figure crops (`Component.image_path`); queries are text encoded by CLIP's text tower. Figures without a crop are skipped and counted (`skipped_no_image`; 0 here).
5. **Branches and layout:**
   - `data/indexes/{text,table,figure}/{bm25,dense_bge,dense_minilm}` plus `figure/clip`, which makes 10 indexes;
   - branch membership comes from config `branches`;
   - a branch with no components is skipped and reported (none were empty here).
6. **Fingerprint and staleness:**
   - Every index's meta.json stores `fingerprint = {lines, sha256}` of `components.jsonl`, plus `build_secs` and `built_at`.
   - `build_indexes` without `--rebuild`: missing indexes are built; up-to-date ones are loaded, not rebuilt; if any is stale it prints the warning with the exact `--rebuild` command and exits with code 2 without touching anything.
   - `search_cli` prints the same warning and still searches.
7. **Models load from the cache first** (`local_files_only=True`, falling back to a download). Found during the demos (fix 1): after the first download, search and build need no network and no SSL workaround.
8. **Demo PDF:** `data/raw/pdfs/` is empty, so it was skipped as instructed and the CLIP checks use `ttwl0228` (and `pybv0228`) figures.
9. **Extra document ingested:** `pybv0228` (1 page, 3 components) was ingested to test staleness for real. The index set therefore covers **4 documents / 111 components**.

### 3. How to run (exact commands, in order, as a teammate would on a fresh clone)
```powershell
# after P1/P2 setup and ingest:
.venv/Scripts/python -m scripts.ingest --source mpdocvqa --docs jjvg0227 yxvw0217 ttwl0228 pybv0228
# first run downloads BGE + CLIP (behind Avast: set $env:SSL_CERT_FILE first, only for this download)
.venv/Scripts/python -m scripts.build_indexes
.venv/Scripts/python -m scripts.build_indexes            # second run: loads, no rebuild
.venv/Scripts/python -m scripts.search_cli "population in poverty Butler county" --branch table --retriever bm25 --doc jjvg0227 --k 3
.venv/Scripts/python -m scripts.search_cli "trapping CO2 underground so power plants pollute less" --branch text --retriever all --doc ttwl0228 --target ttwl0228_p6_paragraph_10 --show-encoded
.venv/Scripts/python -m scripts.search_cli "chart" --branch figure --retriever clip --doc ttwl0228 --k 3
.venv/Scripts/python -m pytest
# after ingesting more documents:
.venv/Scripts/python -m scripts.build_indexes --rebuild
```

### 4. Actual output (real terminal output from this session, trimmed)

**Step 1, commit and push:**
```
e0b2676 phase-2: parsing, serialization, graph
3fe7553 phase-1: setup, data, schema
## main...origin/main
To https://github.com/Samridhaa-28/Doc-Scout.git
   3fe7553..e0b2676  main -> main
```
**Demo PDF:** `ls -la data/raw/pdfs` → only `.` and `..` (empty), so the PDF part was skipped.

**First `build_indexes`** (3 documents; downloads BGE + CLIP; MiniLM was already cached):
```
components.jsonl: 108 components, 3 documents ['jjvg0227', 'ttwl0228', 'yxvw0217']
by type: {'paragraph': 85, 'heading': 10, 'table': 6, 'figure': 6, 'caption': 1}
fingerprint: lines=108 sha256=13b4f6728314725f...
branch  retriever         n  action                     secs        size
text    bm25             96  built                     0.004     25.5 KB
text    dense_bge        96  built                    61.440    580.2 KB
text    dense_minilm     96  built                    22.928    292.1 KB
table   bm25              6  built                     0.002     10.8 KB
table   dense_bge         6  built                     2.566     36.7 KB
table   dense_minilm      6  built                     0.177     18.7 KB
figure  bm25              6  built                     0.000      3.3 KB
figure  dense_bge         6  built                     0.394     36.7 KB
figure  dense_minilm      6  built                     0.069     18.7 KB
figure  clip              6  built                    53.649     24.7 KB
total: 141.2s | indexes in data\indexes
```
**Second run** (no `--rebuild`), reload only:
```
text    bm25             96  up to date (loaded)       0.003     25.5 KB
text    dense_bge        96  up to date (loaded)       0.008    580.2 KB
...
figure  clip              6  up to date (loaded)       0.003     24.7 KB
total: 0.0s | indexes in data\indexes          (whole command: 1 s)
```
**`--rebuild` with models cached** (3 documents). The time includes loading each model:
```
text    dense_bge        96  built                    22.442    580.2 KB
text    dense_minilm     96  built                     4.923    292.1 KB
table   dense_bge         6  built                     2.487     36.7 KB
figure  clip              6  built                     6.798     24.7 KB
(bm25 0.000–0.001 s; other dense < 0.5 s)   total: 37.3s
```
**Model download sizes** (HF cache folders):
```
models--BAAI--bge-base-en-v1.5                        418.7 MB   (downloaded in this phase)
models--sentence-transformers--clip-ViT-B-32          578.5 MB   (downloaded in this phase)
models--sentence-transformers--all-MiniLM-L6-v2       262.0 MB   (already cached before P3; the folder holds extra formats)
```

**Staleness, tested for real:** I ingested one more document (`pybv0228`, 1 page: `docs: 1 | pages: 1 | components: 3`, 68.1 s including model loading), then:
```
--- build_indexes (no --rebuild)
components.jsonl: 111 components, 4 documents ['jjvg0227', 'pybv0228', 'ttwl0228', 'yxvw0217']
fingerprint: lines=111 sha256=b373f1188102b47d...
WARNING: 10 index(es) are STALE (components.jsonl changed since they were built: text/bm25, text/dense_bge, text/dense_minilm, table/bm25, table/dense_bge, table/dense_minilm, figure/bm25, figure/dense_bge, figure/dense_minilm, figure/clip). Run: .venv/Scripts/python -m scripts.build_indexes --rebuild
exit code: 2
--- search_cli on stale index
WARNING: 10 index(es) are STALE (...). Run: .venv/Scripts/python -m scripts.build_indexes --rebuild
  1    1.0406  jjvg0227_p4_table_0          table     p4    Table: Missouri Food Donation Program ...
--- build_indexes --rebuild
text    bm25             98  built                     0.001     25.8 KB
text    dense_bge        98  built                    15.891    592.2 KB
text    dense_minilm     98  built                     0.740    298.2 KB
table   bm25              6  built                     0.001     10.8 KB
table   dense_bge         6  built                     2.776     36.7 KB
table   dense_minilm      6  built                     0.184     18.7 KB
figure  bm25              7  built                     0.000      3.9 KB
figure  dense_bge         7  built                     0.496     42.7 KB
figure  dense_minilm      7  built                     0.077     21.7 KB
figure  clip              7  built                     1.486     28.7 KB
total: 21.7s
exit code: 0
--- build_indexes again: 10 indexes "up to date"
```
Final index size on disk: **1,079.4 KB** in total. Example: `text/dense_bge` = `embeddings.npy` 301,184 B + `faiss.index` 301,101 B + `meta.json` 4,167 B.

**search_cli 1: BM25 table query with the doc filter:**
```
=== table/bm25  query='population in poverty Butler county'  doc=jjvg0227 ===
  1    2.7197  jjvg0227_p5_table_0          table     p5    Table: Missouri Food Donation Program By MASW Divisions and Counties Columns: COUNTY. | Population (1).No. | Families In
  2    1.2230  jjvg0227_p8_table_0          table     p8    Table: Missouri Food Donation Program By MASW Divisions and Counties Columns: COUNTY | Population (1) No. | Families In
  3    1.1394  jjvg0227_p4_table_0          table     p4    Table: Missouri Food Donation Program By MASW Divisions and Counties Columns: COUNTY. | Population (1).No. | Families In
  doc filter: 4 results, all from jjvg0227: True
```
The Butler county row is on page 5, and `p5_table_0` is ranked first.

**search_cli 2: paraphrase, BM25 vs BGE (and MiniLM).**
Target `ttwl0228_p6_paragraph_10`: "Other research seeks ways to capture and store carbon dioxide emitted during the use of fossil fuels. Even lessconventional options, such as marine fertilization to absorb carbon dioxide, should be examined."
```
query tokens: ['trapping', 'co2', 'underground', 'power', 'plants', 'pollute', 'less']
shared tokens: []
=== text/bm25  query='trapping CO2 underground so power plants pollute less'  doc=ttwl0228 ===
  1    5.4899  ttwl0228_p5_paragraph_4      paragraph p5    CO2 Emissions Relative
  doc filter: 5 results, all from ttwl0228: True
  rank of ttwl0228_p6_paragraph_10: not retrieved (of 5 results)
=== text/dense_bge  ... ===
encoded query string: 'Represent this sentence for searching relevant passages: trapping CO2 underground so power plants pollute less'
  1    0.5657  ttwl0228_p6_paragraph_10     paragraph p6    Other research seeks ways to capture and store carbon dioxide ...
  doc filter: 61 results, all from ttwl0228: True
  rank of ttwl0228_p6_paragraph_10: 1 (of 61 results)
=== text/dense_minilm  ... ===
encoded query string: 'trapping CO2 underground so power plants pollute less'
  rank of ttwl0228_p6_paragraph_10: 2 (of 61 results)
```
(The same ranks were measured before and after the 4-document rebuild: BM25 not retrieved, BGE 1, MiniLM 2.)

**BGE instruction: queries only, never documents.** The exact strings encoded, plus a check against the stored vector:
```
QUERY string encoded   : 'Represent this sentence for searching relevant passages: trapping CO2 underground so power plants pollute less'
DOCUMENT string encoded: 'Other research seeks ways to capture and store carbon dioxide emitted during the use of fossil fuels. Even lessconventional options, such as marine fertilization to absorb carbon dioxide, should be examined.'
cosine(stored doc vector, encode(text_form))            = 1.000000
cosine(stored doc vector, encode(instruction+text_form)) = 0.957952
query_instruction stored in index meta: 'Represent this sentence for searching relevant passages: '
minilm query_instruction: ''
```

**search_cli 3: CLIP figure queries.** The 6 `ttwl0228` crops were inspected visually: `p4_figure_0` is a line chart of temperatures; `p5_figure_0` and `p5_figure_1` are small grainy fragments of a bar chart (Docling split it); `p5_figure_2` is the ExxonMobil logo; `p7_figure_0` and `p8_figure_0` are whole, nearly black cover pages.
```
=== figure/clip  query='chart'  doc=ttwl0228 ===
  1    0.2439  ttwl0228_p4_figure_0         figure    p4    Source: Science (1996) 78 77 Medieval warm périod 76 Little icè age ...
  2    0.2338  ttwl0228_p8_figure_0         figure    p8    EXonMobil 5959 Las Colinas Boulevard ...
  3    0.2324  ttwl0228_p7_figure_0         figure    p7    G These Op-Eds appeared in several U.S. publications ...
  doc filter: 6 results, all from ttwl0228: True
=== figure/clip  query='a line graph of temperature over time'  doc=ttwl0228 ===
  1    0.3109  ttwl0228_p4_figure_0   (line chart; next best 0.2221)
=== figure/clip  query='company logo'  doc=ttwl0228 ===
  1    0.2622  ttwl0228_p5_figure_2   (ExxonMobil logo)
=== figure/clip  query='bar graph'  doc=ttwl0228 ===
  1    0.2374  ttwl0228_p7_figure_0   (black cover page)  <- wrong
  2    0.2364  ttwl0228_p5_figure_0   (bar fragment)
  5    0.2145  ttwl0228_p5_figure_1   (bar fragment)
=== figure/clip  query='chart'  (all documents, after the 4-document rebuild) ===
  1    0.2548  pybv0228_p81_figure_0        figure    p81   FIGURE C.2 AGE ADJUSTED MOTOR VEHICLE ACCIDENT MORTALITY RATE CANADA ...
  2    0.2439  ttwl0228_p4_figure_0         figure    p4    Source: Science (1996) ... (line chart)
```

**Component checks** (4 documents): `lines: 111 | unique ids: 111 | duplicates: 0` … `ALL CHECKS PASSED`.

### 5. Tests
`36 passed in 12.78s` (13 P1 + 15 P2 + 8 new in `tests/test_indexing.py`; none skipped):
- `test_tokenizer_numbers_punctuation_stopwords`: `"The Revenue (2023) was 1,234.5 & 26.9%!"` → `['revenue', '2023', '1234.5', '26.9']`; splits on hyphens and dots
- `test_bm25_keyword_match_first`: synthetic corpus; the keyword match "weather rainy" is ranked first
- `test_bm25_doc_filter_exact`: `doc_id="b"` returns only doc b; an unknown doc returns `[]`
- `test_bm25_save_load_roundtrip`: identical results after save/load
- `test_bge_instruction_only_on_queries`: a recording encoder sees plain `text_form` for every document and `instruction + query` for the query
- `test_dense_doc_filter_matches_brute_force`: doc-filtered ranking and scores equal a brute-force numpy computation, the unfiltered FAISS top hit agrees, and results survive save/load
- `test_dense_ranks_paraphrase_first_bm25_does_not`: real MiniLM (from the local cache, offline). "sales went up this quarter" has no shared token with the revenue sentence; dense ranks it 1st and BM25 doesn't retrieve it
- `test_stale_detection`: fingerprint has line count + sha256; status goes missing → ok → stale after a line is appended

### 6. Acceptance criteria and reviewer checks
- PASS: `build_indexes` finishes on all processed documents. 10 indexes over 4 documents / 111 components (`total: 21.7s`, exit 0).
- PASS: indexes load again without rebuilding. Second run: all 10 `up to date (loaded)`, 0.001–0.008 s each, 1 s for the whole command.
- PASS: search_cli shows a table query finding a table via BM25. `jjvg0227_p5_table_0` is rank 1 for "population in poverty Butler county" (the Butler row is on that table).
- PASS: search_cli shows a paraphrased query found by BGE but not by BM25. Target rank: BGE 1, BM25 not retrieved (0 shared tokens).
- PASS (on MP-DocVQA; the demo PDF was unavailable): a "chart" query returns a figure via CLIP. "chart" → `ttwl0228_p4_figure_0` (a line chart) at rank 1. Across all documents, "chart" → the two real charts at ranks 1–2.
- FAIL (reported, not tuned): "bar graph" → rank 1 is a black cover page, with the bar fragments at ranks 2 and 5. The bar chart exists only as tiny, noisy fragments (a Docling split), and the scores are within 0.003 of each other. See section 8.
- PASS: index sizes and build time reported. Per-index KB and seconds are in section 4; 1,079.4 KB in total; first build 141.2 s (with downloads), rebuild 21.7–37.3 s.
- PASS: model download sizes reported. BGE 418.7 MB, CLIP 578.5 MB (MiniLM already cached, 262.0 MB folder).
- PASS: the BGE query prefix is used only for queries. Exact strings shown; stored document vector = `encode(text_form)` (cosine 1.000000), not `encode(instruction + text_form)` (0.958); unit test `test_bge_instruction_only_on_queries`.
- PASS: the doc filter works and all results come from the requested document. Every search_cli run printed `all from <doc>: True` (BM25, BGE, MiniLM, CLIP); unit tests check BM25 and dense against brute force.
- PASS: the paraphrase example shows dense beating BM25. BGE 1 vs BM25 not retrieved (MiniLM 2).
- PASS: indexes reload from disk with no rebuild on every run (see above), and search_cli only loads.
- PASS: stale index detection. Real test with a 4th document: warning plus the `--rebuild` command in both scripts, `build_indexes` exit code 2; after `--rebuild` all 10 are up to date.
- SKIPPED (as instructed, folder empty): demo-PDF ingest, visualize and check, and its overlay descriptions and s/page.
- PASS: `pytest`: `36 passed`.

### 7. Interfaces for the next phase
- **Load and search:**
  - `from docscout.indexing.build_all import load_index, retrievers_for, stale_indexes`
  - `load_index(branch, retriever) -> BaseIndex | None`, with branch ∈ {text, table, figure} and retriever ∈ {bm25, dense_bge, dense_minilm, clip (figure only)}
  - `idx.search(query, top_k=50, doc_id=doc) -> [(component_id, score)]`, best first. BM25 returns only rows with a positive score; dense and CLIP return every row of the document (cosine scores).
  - `DenseIndex.encode_query(q)` and `search_vector(vec, top_k, doc_id)` let P5 encode a query once and reuse the vector across branches.
- **Score scales** (for P5 normalization): BM25 is unbounded and ≥ 0; BGE/MiniLM cosine is about 0.2–0.9; CLIP text-image cosine is about 0.2–0.35, a narrow range.
- **On disk:** `data/indexes/{branch}/{retriever}/meta.json` = `{kind, n, ids, doc_ids, model, query_instruction, dim, fingerprint{lines, sha256}, build_secs, built_at}`, plus `bm25.pkl` or `embeddings.npy` + `faiss.index`.
- **Config used:** `branches.*`, `models.{bge, minilm, clip, bge_query_instruction, device, batch_size}`, `paths.indexes_dir`, `paths.components_file`. `retrieval.top_k.*` is ready for P5.
- **Scripts:** `scripts.build_indexes [--rebuild]` (exit 2 = stale); `scripts.search_cli "q" --branch --retriever --doc --k --target --show-encoded`.

### 8. Known limitations / open questions
1. **CLIP on scanned figures:** CLIP works on clear images ("line graph of temperature" margin 0.311 vs 0.222; "company logo" correct). On this data, though:
   - Docling sometimes splits a chart into tiny noisy fragments, and whole dark cover pages get detected as "figures";
   - CLIP scores cluster within about 0.02, so "bar graph" ranks a black page first.

   Fusion (P5) uses the figure branch's text retrievers too (caption + nearest paragraph). A real demo PDF with clean charts is still needed for the CLIP demo.
2. **Scale:** indexes currently cover only 4 MP-DocVQA documents (111 components). After the full ingest (~5 h), run `build_indexes --rebuild`. Projected cost: dense build over about 10k components ≈ a few minutes for BGE on CPU, plus a few MB of disk (estimate, not measured).
3. Embeddings are stored twice (`embeddings.npy` + FAISS flat index), which doubles the dense index size. That's small now and simple to change later.
4. BM25 IDF is computed per branch across all documents, not per document. For P4/P5 within-document evaluation this is a reasonable choice, but it means a document's scores shift slightly when other documents are added (seen: 5.4688 → 5.4899 for the same hit).
5. `pybv0228` (ingested only to test staleness) has a table-like page that Docling parsed as 2 paragraphs + 1 figure (P2 parser behaviour, noted for the full ingest).

### 9. What I need from you before the next phase
1. **Demo PDF:** `data/raw/pdfs/` is still empty. Please copy the PDF in (OneDrive "Always keep on this device" if it shows as cloud-only). Then I'll run `ingest --source pdf`, `visualize_components`, `check_components`, describe a table page and a chart page, report s/page, rebuild the indexes, and repeat the CLIP "chart"/"bar graph" check on a clean figure.
2. **Full ingest** (~5 h, resumable) whenever convenient, followed by `build_indexes --rebuild`:
   ```powershell
   .venv/Scripts/python -m scripts.ingest --source mpdocvqa
   .venv/Scripts/python -m scripts.build_indexes --rebuild
   ```
   P4 (qrels and baselines) needs most of the 150 documents ingested to give meaningful numbers.
3. P3 is **not committed**. Approve it and give a commit message when you're ready.

### 10. Fixes made during this phase
| # | Problem found (how) | Change (file: one line) | Rerun result |
|---|---|---|---|
| 1 | Every model load sent Hub HEAD requests even though the models were cached. Without `SSL_CERT_FILE` they failed and retried for about 30 s per model (seen in the first paraphrase `search_cli` run, which printed many `CERTIFICATE_VERIFY_FAILED ... Retrying` lines) | `src/docscout/indexing/dense_index.py`: `load_st_model()` tries `SentenceTransformer(..., local_files_only=True)` first and falls back to a download | CLIP `search_cli` run with `SSL_CERT_FILE` unset: no Hub requests, finished in 14 s; all later runs quiet |
| 2 | `ClipIndex` inherited `DenseIndex.load()`, which calls the constructor with a `query_instruction` argument that `ClipIndex.__init__` doesn't accept (caught while reviewing the code, before the first run) | `src/docscout/indexing/visual_index.py`: added `ClipIndex.load()` | Second `build_indexes` run: `figure clip ... up to date (loaded)`; CLIP searches work after reload |
| 3 | `build_all` deleted the files in an index folder before saving, which was unnecessary because every file is overwritten (caught in review; avoids unrequested deletes) | `src/docscout/indexing/build_all.py`: removed the delete loop; `save()` overwrites | `--rebuild` runs replace every index (sizes and fingerprint updated: `lines=111`) |
| — | "bar graph" CLIP query ranks a black cover page first | Not changed: the cause is the data (fragmented chart crops, dark full-page "figures"), and changing it would mean tuning on test queries | Reported as FAIL in section 6 and limitation 1 in section 8 |
