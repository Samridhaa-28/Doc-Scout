## PHASE REPORT: P2 Parsing, text serialization, document graph

### Changes after review
| # | Issue | What I changed (file: one line) | Rerun result |
|---|---|---|---|
| R1 | `ingest --force` left old files behind and removed old lines for all documents up front | `scripts/ingest.py`: new `clean_doc_outputs()` and a per-document `drop_docs()` run just before each document is parsed. Removed: its `tables/{doc}_*`, `figures/{doc}_*`, `pages/{doc}/`, `debug/{doc}/`, `graphs/{doc}.pkl`, `docs/{doc}.json` and its `components.jsonl` lines | `yxvw0217: removed 18 old components.jsonl lines, old files {'tables': 2, ..., 'other': 4}`; `jjvg0227: removed 23 …`; `ttwl0228: removed 67 …`. Afterwards `data/processed/tables` + `figures` hold exactly 12 files (6 + 6); `test_force_cleanup_removes_old_lines_and_files` passes |
| R2 | The original label was lost when a caption was relabelled `page_marker` | `src/docscout/ingestion/parser.py`: store `meta["docling_label_original"]` when the label changes | 4 page markers: `('jjvg0227_p5_paragraph_1', 'page_marker', 'caption')` and the same for pages 4, 6 and 8 |
| R3 | No check that component ids are unique | New `scripts/check_components.py` (all validity checks; prints line count vs unique-id count). `tests/test_ingestion.py`: `test_duplicate_ids_detected`, and the data test now runs the full validator | `lines: 108 \| unique ids: 108 \| duplicates: 0` … `ALL CHECKS PASSED` |
| R4 | OCR speed (torch backend) | `requirements.txt`: `+onnxruntime`. `requirements.lock.txt`: regenerated (line 1 is still the extra-index-url; `onnxruntime==1.23.2`). `configs/config.yaml`: `parsing.ocr_backend: onnxruntime`. `parser.py`: `RapidOcrOptions(backend=cfg, lang=[])`, the same options Docling's auto picker used | Warm run on `yxvw0217`, back-to-back: **torch 27.4 s/page → onnxruntime 20.0 s/page (−27%)**; OCR stage 51.7 s → 35.3 s. Output identical: 108/108 texts, same ids and counts, max bbox difference 0.00 px. **onnxruntime kept**; all 3 documents re-ingested with it |
| R5 | Demo-PDF part (you added a PDF) | — | **NOT RUN: `data/raw/pdfs/` is still empty in this session.** `ls -laR data/raw/pdfs` shows only `.` and `..`, a search of the repo found no `.pdf`, and a recheck after the re-ingest found nothing either. Possibly OneDrive hasn't synced it yet, or it went to another folder. See section 9 |
| R6 | Recheck everything | `scripts/check_components.py` + `pytest` on all ingested documents | `28 passed`; all checks pass on 3 documents / 108 components (section 6) |

### 0. Summary in plain words
- **Built:** the offline pipeline for one document: Docling parsing (OCR + layout + table structure) → Components → `text_form` → a networkx structure graph.
  - MP-DocVQA documents are joined from their page PNGs into one PDF and go through the same code path as real PDFs.
  - Components keep the original page numbers from `meta.json`, so `Component.page` equals `answer_page`.
  - Bboxes are top-left pixel coordinates of the exact image we display.
- **Results on 3 chosen MP-DocVQA documents** (11 pages, including a page-numbering gap and 6 tables):
  - Parsing worked: 108 components with unique ids, every page processed, 0 errors.
  - The debug overlays line up with the page content, and the tables are real multi-column parquet files.
- **OCR backend changed** from torch to onnxruntime after review: same output, about 27% faster on a warm converter. Full ingest estimated at about **5 h** (was about 8 h); not run.
- **Still not run:** the demo-PDF part, because `data/raw/pdfs/` is empty here (see section 9).

### 1. Files created/changed
**Before P2** (steps 1–3 of the P1-approval message, committed in `3fe7553 phase-1: setup, data, schema`):
- `requirements.lock.txt`: first line `--extra-index-url https://download.pytorch.org/whl/cpu`
- `README.md`: one line on installing from the lock file
- `docs/PROGRESS.md`: "P1 (approved)" section

**P2** (not committed):
- `src/docscout/ingestion/parser.py`: Docling converter (RapidOCR, backend from config), PNG→PDF join, bbox conversion, label mapping, tables→parquet, figure crops, caption linking (`docling_label_original` kept), section ids, page manifest
- `src/docscout/ingestion/serializer.py`: `text_form` for every type, table title/caption, ~512-token truncation
- `src/docscout/graph/builder.py`: DiGraph Document→Page→Section→Block, sibling and caption_of edges, pickle, helpers
- `scripts/ingest.py`: `--source mpdocvqa|pdf --limit N --docs ... --force --reserialize`; removes old outputs per document before parsing; per-document and total counts and timing
- `scripts/visualize_components.py`: `--doc DOC_ID` → bboxes and continuation regions coloured by type, with ids, drawn on the displayed page images
- `scripts/check_components.py` (new after review): every validity check, including unique ids
- `tests/test_ingestion.py`: 15 tests
- `configs/config.yaml` (P1 file, small additions): `paths.docs_dir`, `paths.pages_dir`, `paths.combined_pdfs_dir`, `parsing.mpdocvqa_dpi`, `parsing.ocr_backend`, `parsing.caption_max_gap`, `parsing.num_threads`
- `requirements.txt` / `requirements.lock.txt` (P1 files): `+onnxruntime` (you approved it); lock regenerated with the extra-index-url line kept (169 lines)
- `README.md` (P1 file, small addition): "Ingest" section
- `docs/reports/P2_REPORT.md`: this report

### 2. Design decisions and any deviations from the architecture (and why)
1. **One code path.**
   - MP-DocVQA: the PNGs listed in `meta.json` are joined in page order into `data/processed/mpdocvqa_pdfs/{doc_id}.pdf`. Each PDF page measures `png_px × 72 / mpdocvqa_dpi` points, with `mpdocvqa_dpi = 200`. This keeps Docling's OCR render (scale 3, about 216 dpi) close to the PNG's native resolution.
   - PDF page j (1-based in Docling) maps to `PageInfo(page=page_numbers[j-1], width, height, image_path)`. `Component.page` and the ids use that number (for example `jjvg0227_p8_table_0` on a document with pages 4, 5, 6 and 8).
2. **Coordinates.** `docling_bbox_to_pixels()` turns Docling's bbox (PDF points, BOTTOMLEFT or TOPLEFT origin) into top-left pixels of the displayed image:
   - scale by `img_w / page_w_pt` and `img_h / page_h_pt`;
   - flip y when the origin is bottom-left;
   - sort the corners and clamp to the image.

   The displayed image is the original PNG for MP-DocVQA and the page rendered at `image_dpi` (144) for PDFs (cached in `data/processed/pages/{doc_id}/`). `page_width`/`page_height` are that image's size, and `visualize_components` asserts this before drawing. A per-document manifest `data/processed/docs/{doc_id}.json` records the pages, sizes, image paths and parse stats.
3. **Label mapping:**
   - heading = `section_header`, `title`
   - caption = `caption`
   - table = `table`, `document_index`
   - figure = `picture`, `chart`
   - paragraph = every other text item

   **Page headers and footers are kept** (DocVQA answers are often in them). `meta.docling_label` holds the label used; `meta.docling_label_original` is set when we relabelled it.
4. **Multi-region items.** Docling merges a paragraph that continues in another column or page into one item with several provenance boxes. `bbox` is the first box, and all boxes are stored in `meta["regions"]` and drawn with a `+` label.
5. **Captions.**
   - Docling's caption references are used first. The fallback is the nearest unlinked caption above or below on the same page, with horizontal overlap and a gap of at most 15% of the page height.
   - Page markers that Docling labels "caption" (`^page \d+`) are relabelled `page_marker` and treated as paragraphs.
   - Links are stored both ways (`caption_id` / `caption_of`, plus `caption_link`).
6. **Table `text_form`:**
   - `Table: <title>`, where the title is the nearest block directly above the table, excluding page markers and footers;
   - `Caption: <caption>`;
   - `Columns:`, `Rows:` (first column, up to 30), then the first 3 rows.

   Figures: caption, the nearest paragraph, and text found inside the picture. A component never gets an empty `text_form`. Truncation keeps 512 word/punctuation tokens.
7. **Graph.**
   - A Section node is (heading, page), so `path_to_root(block)` always passes through the block's own page. Blocks before the first heading hang under their Page.
   - Edge `rel` values: `parent_child`, `sibling` (reading order), `caption_of`.
8. **Robustness:**
   - `convert(raises_on_error=False)`; each item is processed in its own try/except.
   - Missing or empty pages are counted in the manifest.
   - **Before a document is parsed, all of its old outputs and its old lines are removed** (R1). This also cleans up after a crashed run.
9. **OCR engine (changed after review):** RapidOCR, backend from `parsing.ocr_backend`, set to **onnxruntime**.
   - Same options Docling's auto picker used (`lang=[]`, so the default PP-OCRv6 det/rec small + v2 cls models); only the backend changes.
   - In the first review round the auto picker chose torch, because onnxruntime wasn't installed.
   - Evidence that the switch is safe: identical texts and bboxes on all 108 components (R4).
10. **Model storage note:** RapidOCR keeps its models in `.venv/Lib/site-packages/rapidocr/models/` (`.pth` for torch; `.onnx` files were already present for onnxruntime). That is inside the gitignored venv, not the repo. Docling's layout and table models go to the HF cache.
11. **`ingest --reserialize`** recomputes `text_form` and the graphs without re-parsing.

**Documents chosen for the 3-document ingest:**

| doc | pages | why |
|---|---|---|
| `jjvg0227` | 4 (numbered 4, 5, 6, 8) | multi-page with a gap in page numbers; 4 large statistical tables |
| `yxvw0217` | 2 (3, 4) | questions say "first column of the table"; 1 table per page under a section heading |
| `ttwl0228` | 5 (4–8) | questions mention "the chart"; figures and captions, two-column text |

### 3. How to run (exact commands, in order, as a teammate would on a fresh clone)
```powershell
# after the P1 setup (README; requirements now include onnxruntime) and the data download:
.venv/Scripts/python -m scripts.download_data --limit 150
# P2:
.venv/Scripts/python -m scripts.ingest --source mpdocvqa --docs jjvg0227 yxvw0217 ttwl0228
.venv/Scripts/python -m scripts.visualize_components --doc jjvg0227
.venv/Scripts/python -m scripts.visualize_components --doc ttwl0228
.venv/Scripts/python -m scripts.check_components
.venv/Scripts/python -m pytest
# demo PDFs (once data/raw/pdfs/ has a PDF):
.venv/Scripts/python -m scripts.ingest --source pdf
.venv/Scripts/python -m scripts.visualize_components --doc <pdf-doc-id>
# full MP-DocVQA ingest (LONG, ~5 h with onnxruntime; not run):
.venv/Scripts/python -m scripts.ingest --source mpdocvqa
```
On this machine (Avast HTTPS scanning), set `$env:SSL_CERT_FILE` to the certifi + Windows-roots bundle first, for model downloads.

### 4. Actual output (real terminal output from this session, trimmed)

**Steps 3–4 of the P1-approval message:**
```
git log --oneline -3          ->  3fe7553 phase-1: setup, data, schema
git status -sb                ->  ## main...origin/main
push                          ->  * [new branch]      main -> main ; branch 'main' set up to track 'origin/main'.
```
**`download_data --limit 150`** (445 s):
```
Selected first 150 documents; image pass streams rows 0..679.
Resume: 4/150 docs already complete on disk; 1044 pages to download.
image pass: 100%|██████████| 680/680 [07:04<00:00,  1.60row/s]
New pages saved: 1044. Docs: 150 | pages on disk: 1060
Questions written: 947 -> data\raw\mpdocvqa\questions.jsonl
Questions dropped: {'answer page not saved': 231}
exit: 0  elapsed: 445s
```
**`explore_data`** (150 documents):
```
documents (with images): 150
pages per doc: min=1 mean=7.07 max=61 (total pages: 1060)
docs where png count != num_pages: 0
questions: 947 (per doc: 6.31)
questions missing answer_page_idx or answers: 0
questions with answer_page_idx out of range: 0
questions whose answer_page_id != page_ids[answer_page_idx]: 0
questions whose answer page image is missing: 0
questions on single-page docs (page retrieval trivial): 172/947
question types: not available in this dataset (no question_types field)
```

**OCR backend comparison** (after review):

onnxruntime install: `Successfully installed ... onnxruntime-1.23.2`; providers `['AzureExecutionProvider', 'CPUExecutionProvider']`; `pip check` → `No broken requirements found.`

Re-ingest of `yxvw0217` on onnxruntime (`--force`):
```
Converter ready in 21.8s (models load lazily on first document); OCR: RapidOCR, backend=onnxruntime
yxvw0217: removed 18 old components.jsonl lines, old files {'tables': 2, 'figures': 0, 'pages': 0, 'other': 4}
[RapidOCR] base.py:23: Using engine_name: onnxruntime
yxvw0217                   2    14     2     2     0     0    59.8   29.92  success
```
This cold number includes OCR model loading. Torch's cold run on the same document was 59.8 s (29.9 s/page). The warm runs below are the fair comparison: same script and PDF, back-to-back, converter warmed by run 1.
```
[torch] run 1: 66.0s for 2 pages (33.0 s/page)
[torch] run 2: 54.8s for 2 pages (27.4 s/page)
  pipeline_total 54.7s | layout 2.8s | ocr 51.7s | table_structure 4.6s
[onnxruntime] run 1: 50.9s for 2 pages (25.5 s/page)
[onnxruntime] run 2: 40.1s for 2 pages (20.0 s/page)
  pipeline_total 39.8s | layout 9.9s | ocr 35.3s | table_structure 21.9s
```
(The stages run in parallel threads, so they add up to more than the total. With onnxruntime, layout and table stages take longer because the threads compete for CPU, but the total still drops.)

Quality, `yxvw0217`, torch vs onnxruntime:
```
counts torch: {'paragraph': 14, 'heading': 2, 'table': 2} | onnxruntime: {'paragraph': 14, 'heading': 2, 'table': 2}
same ids: True
text similarity (rapidfuzz ratio) over 18 components: mean 100.0, identical 18, min 100.0
total chars torch 5907 | onnxruntime 5907
--- yxvw0217_p3_heading_0
  torch: 'II. Manuscripts - Totelle Cycle 1 mg'
  onnx : 'II. Manuscripts - Totelle Cycle 1 mg'
--- yxvw0217_p3_paragraph_2
  torch: 'DesignW rite, 1 Inc.'
  onnx : 'DesignW rite, 1 Inc.'
--- yxvw0217_p3_table_0
  torch: 'Authors | Priority | Article title | Stage | Journal | Timing | Status\nGenazzani AR, Gambaccia ni M, Spielmann D, and the Trimegest one 301 Study Group | A | Paper PS(1) A study of the control of climacteric symptoms in postmenopa'
  onnx : (identical)
```
Re-ingest of the other 2 documents on onnxruntime:
```
jjvg0227: removed 23 old components.jsonl lines, old files {'tables': 4, 'figures': 0, 'pages': 0, 'other': 6}
ttwl0228: removed 67 old components.jsonl lines, old files {'tables': 0, 'figures': 6, 'pages': 0, 'other': 7}
doc_id                 pages  para  head table   fig   cap    secs  s/page  status
jjvg0227                   4    18     1     4     0     0    90.6   22.64  success
ttwl0228                   5    53     7     0     6     1    45.9    9.17  success
time: 136.4s total, 15.16 s/page (includes first-document model loading)
```
(Same documents on torch: `jjvg0227` 113.5 s (28.38 s/page), `ttwl0228` 121.7 s (24.33 s/page).)

All 3 documents, torch vs onnxruntime:
```
jjvg0227: torch 23 / onnx 23 components | same ids True | identical texts 23/23 | mean ratio 100.0
yxvw0217: torch 18 / onnx 18 components | same ids True | identical texts 18/18 | mean ratio 100.0
ttwl0228: torch 67 / onnx 67 components | same ids True | identical texts 67/67 | mean ratio 100.0
max bbox coordinate difference torch vs onnxruntime over 108 components: 0.00 px
```

**Measured speed** with onnxruntime on all 11 pages: (59.8 + 136.4) s / 11 = **17.8 s/page**, including model loading at the start of each run. Torch, same 11 pages: 295.0 s / 11 = 26.8 s/page.

**Component checks** (`scripts.check_components`, all ingested documents):
```
lines: 108 | unique ids: 108 | duplicates: 0
  jjvg0227: {'source': 'mpdocvqa', 'components': 23, 'pages': [4, 5, 6, 8], 'meta_page_numbers': [4, 5, 6, 8], 'answer_pages': [4, 5, 6, 8]}
  ttwl0228: {'source': 'mpdocvqa', 'components': 67, 'pages': [4, 5, 6, 7, 8], 'meta_page_numbers': [4, 5, 6, 7, 8], 'answer_pages': [4]}
  yxvw0217: {'source': 'mpdocvqa', 'components': 18, 'pages': [3, 4], 'meta_page_numbers': [3, 4], 'answer_pages': [3]}
ok   duplicate_ids: 0
ok   bad_id_format: 0
ok   empty_text_form: 0
ok   bad_page: 0
ok   bbox_outside: 0
ok   missing_files: 0
ok   size_mismatch: 0
ok   page_mismatch: 0
ok   answer_page_missing: 0
ALL CHECKS PASSED
```
**Every page processed** (manifests, onnxruntime run):
```
jjvg0227 success docling pages 4 | with components 4 | without [] | errors []
yxvw0217 success docling pages 2 | with components 2 | without [] | errors []
ttwl0228 success docling pages 5 | with components 5 | without [] | errors []
```
**Relabelled page markers** (R2):
```
[('jjvg0227_p4_paragraph_1', 'page_marker', 'caption'), ('jjvg0227_p5_paragraph_1', 'page_marker', 'caption'),
 ('jjvg0227_p6_paragraph_1', 'page_marker', 'caption'), ('jjvg0227_p8_paragraph_1', 'page_marker', 'caption')]
```

**Debug overlays.** Both were opened and inspected in the first round. They were regenerated after the onnxruntime re-ingest, and the bboxes are identical (0.00 px difference).
- `data/processed/debug/jjvg0227/page_5.png` (MP-DocVQA page with a table): the green table box covers exactly the 9-column county table, from the "COUNTY / Population…" header to "Ste. Genevieve". The title "Missouri Food Donation Program By MASW Divisions and Counties" (top left) and "Page 5 January, 1970" (top right) are boxed separately, and so is the "Source: …" footer. No flip or offset.
- `data/processed/debug/ttwl0228/page_5.png` (two-column article with a chart): every paragraph box sits on its text block in both columns, including the continuation regions `+p5_paragraph_7` and `+p5_paragraph_10`. The chart is split by Docling into two magenta bar figures plus axis-label paragraphs, with an orange caption; the ExxonMobil logo is a figure inside a heading box.

**Table parquet** (`yxvw0217_p3_table_0`, 3×7):
```
                       Authors Priority                Article title      Stage     Journal                       Timing                       Status
0  Genazzani AR, Gambaccia ...        A  Paper PS(1) A study of t...  Phase III   Maturitas  July/August 2003 journal...  First draft submitted fo...
1  Koninckx P, Spielmann D,...        A  Paper PS(2) A comparativ...  Phase III  Menopaus e  July 2003 journal submis...  First draft submitted fo...
2                          TBD        C  Paper PS(3) Metabolic an...  Phase III         TBD   Q4 2003 journal submission  First draft submitted fo...
```
`jjvg0227_p5_table_0`: (17, 9). Its `text_form` starts:
```
Table: Missouri Food Donation Program By MASW Divisions and Counties
Columns: COUNTY. | Population (1).No. | Families In Poverty (2). | Estimated Population In Poverty (3).No. | ...
Rows: Osage; Phelps; Pulaski; Washington; SOUTHEAST:; Bollinger; Butler; Cape Girardeau; ...
Osage | 10,700 | 26.9 | 2,878 | 677 | 23.5 | 439 | 209 | 47.6
```
**components.jsonl lines.** Table (trimmed):
```
{"id": "yxvw0217_p3_table_0", "doc_id": "yxvw0217", "page": 3, "type": "table", "text": "Authors | Priority | Article title | Stage | Journal | Timing | Status\n...", "text_form": "Table: II. Manuscripts - Totelle Cycle 1 mg\nColumns: Authors | Priority | Article title | Stage | Journal | Timing | Status\nRows: ...", "page_width": 2200, "page_height": 1786, "section_id": "yxvw0217_p3_heading_0", "table_path": "data/processed/tables/yxvw0217_p3_table_0.parquet", "meta": {"docling_label": "table", "n_rows": 3, "n_cols": 7, ...}}
```
Paragraph:
```
{"id": "yxvw0217_p3_paragraph_2", "doc_id": "yxvw0217", "page": 3, "type": "paragraph", "text": "DesignW rite, 1 Inc.", "text_form": "DesignW rite, 1 Inc.", "bbox": [1647.22, 1552.78, 1917.59, 1608.33], "page_width": 2200, "page_height": 1786, "order": 4, "section_id": "yxvw0217_p3_heading_0", "table_path": null, "image_path": null, "meta": {"docling_label": "text", "docling_ref": "#/texts/3"}}
```
**Graph printout for one table** (same after the re-ingest, since the components are identical):
```
===== graph yxvw0217: 24 nodes, 40 edges, rels=['parent_child', 'sibling']
pages: ['page::yxvw0217::3', 'page::yxvw0217::4']
path_to_root: yxvw0217_p3_table_0 -> sec::yxvw0217_p3_heading_0::p3 -> page::yxvw0217::3 -> doc::yxvw0217
section heading: yxvw0217_p3_heading_0 [heading] 'II. Manuscripts - Totelle Cycle 1 mg'
caption: None
neighbours (k=2): 7/17/2003 | II. Manuscripts - Totelle Cycle 1 mg | DesignW rite, 1 Inc. | Page 3 of 13
===== figure ttwl0228_p5_figure_1 [figure] 'Future Technology'
caption_of: ttwl0228_p5_caption_0 | linked_component(caption): ttwl0228_p5_figure_1
```
**Demo PDF:** not run. `data/raw/pdfs/` is empty (`ls -laR data/raw/pdfs` lists only `.` and `..`; no `*.pdf` in the repo outside `.venv`/`data/processed`).

### 5. Tests
`28 passed` (13 from P1 and 15 in `tests/test_ingestion.py`):
- `test_bbox_bottom_left_points_to_rendered_pixels`: known Docling bbox (PDF points, bottom-left) → `(144, 144, 1080, 288)` at 144 dpi
- `test_bbox_top_left_origin_and_mpdocvqa_png_space`: MP-DocVQA PNG space; TOPLEFT and BOTTOMLEFT forms give the same pixels
- `test_bbox_is_clamped_inside_page`: out-of-range boxes are clamped
- `test_images_to_pdf_page_size`: PNG→PDF join, one page per image, px×72/dpi
- `test_page_marker_regex`: page marker vs a real "Table …" caption
- `test_safe_doc_id`: file stem → id without underscores
- `test_table_text_form_hand_made_table`: title, columns, all row headers, first 3 rows only
- `test_serialize_components_all_types`: `text_form` for every type, including the non-empty fallback
- `test_table_title_skips_page_marker`: table title = the real title, not the page marker
- `test_truncate_tokens`: 512-token cut
- `test_nearest_caption_fallback`: nearest caption below is linked; a far one isn't
- `test_graph_structure_and_rels`: levels, `rel` on every edge, path_to_root, parent_section, siblings, caption_of, linked_component, pickle round trip
- **new** `test_duplicate_ids_detected`: the duplicate-id detector
- **new** `test_force_cleanup_removes_old_lines_and_files`: cleanup deletes only this document's tables, figures, pages and graph, and `drop_docs` removes only its lines
- **changed** `test_processed_components_valid`: runs `scripts.check_components.validate()` on the real `components.jsonl` (unique ids, id format, non-empty `text_form`, page, bbox and regions inside the page, files exist, size = manifest, pages = `meta.json`, answer pages present)

### 6. Acceptance criteria and reviewer checks
- PASS: `ingest --source mpdocvqa --limit 3` finishes. Ran as `--docs jjvg0227 yxvw0217 ttwl0228` (same code path), now on onnxruntime: 3 × `success`, exit 0.
- NOT RUN: `ingest --source pdf` on one demo PDF, because `data/raw/pdfs/` is empty in this session (R5).
- PASS: every component has a non-empty `text_form`, a valid page and a bbox inside the page. `check_components`: `empty_text_form: 0`, `bad_page: 0`, `bbox_outside: 0` over 108 components.
- PASS (new check): component ids are unique. `lines: 108 | unique ids: 108 | duplicates: 0`.
- PASS: page numbers match `meta.json` and `answer_page`: `page_mismatch: 0`, `answer_page_missing: 0` (for example `jjvg0227` pages `[4, 5, 6, 8]` = `meta_page_numbers`).
- PASS (MP-DocVQA) / NOT RUN (demo PDF): the debug overlays line up with the real paragraphs and tables. Overlays inspected: `jjvg0227/page_5.png` (table) and `ttwl0228/page_5.png` (chart). Bboxes unchanged after the re-ingest (0.00 px). The demo-PDF table and chart pages couldn't be opened because there is no PDF.
- PASS: for one table, print its section heading, caption and neighbouring paragraphs from the graph (`yxvw0217_p3_table_0`, section 4).
- PASS: average seconds per page reported and full run estimated. onnxruntime: **17.8 s/page** measured over 11 pages (warm 9–20 s/page depending on text density). Full run of the remaining 1,049 pages ≈ 18,700 s ≈ **5.2 h** (range about 3–6 h). Torch was 26.8 s/page, about 7.9 h.
- NOT RUN (as instructed): the full ingest. Command: `.venv/Scripts/python -m scripts.ingest --source mpdocvqa`.
- PASS: overlays align (no bottom-left/top-left mix-up). Visual check plus `test_bbox_bottom_left_points_to_rendered_pixels`.
- PASS (MP-DocVQA) / NOT RUN (demo PDF): tables exist as parquet files and look like real tables. 6/6 parquet files (`missing_files: 0`); 17×9 and 3×7 with proper column names.
- PASS with caveat: captions are linked to the right table/figure. The false "Page N" links are fixed (4 page markers, original label kept); the remaining link is `ttwl0228_p5_caption_0` → `ttwl0228_p5_figure_1`. That "caption" is the chart's x-axis text as labelled by Docling: attached to the right figure, but not a real caption. The demo PDF (likely to have real captions) is not tested yet.
- PASS: the graph has a Page level and `rel` attributes on its edges (`rels=['parent_child', 'sibling']`, plus `caption_of` in `ttwl0228`; unit test checks every edge).
- PASS (improved): seconds per page reasonable. 17.8 s/page on CPU with onnxruntime: fine for the demo PDFs and incremental runs, about 5 h for the full subset as one resumable overnight run.
- PASS: OCR engine reported and every page processed. RapidOCR, backend onnxruntime (`Using engine_name: onnxruntime`); 11/11 pages have components, 0 errors.
- PASS: bbox space = displayed image (`size_mismatch: 0`; visualize asserts image size = manifest size).
- PASS: `ingest --force` cleans old outputs (R1 output lines; 12 files in tables/figures = 6 + 6; unit test).
- PASS: `pytest`: `28 passed`.

### 7. Interfaces for the next phase
- **`components.jsonl`:** one `Component` per line, ids unique. Extra `meta` keys:
  - `docling_label`, `docling_label_original` (only if relabelled), `docling_ref`
  - `regions` (list of `{page, bbox}`, only for multi-region items)
  - tables: `n_rows`, `n_cols`, `caption_id`, `caption_link`
  - figures: `inner_text`, `caption_id`, `caption_link`
  - captions: `caption_of`, `caption_link`
  - `table_path` / `image_path` are repo-relative posix paths
- **Branches** (config `branches`): text = paragraph, heading, caption; table = table; figure = figure (CLIP uses `image_path`). Index `text_form`; it's never empty.
- **`docscout.ingestion.parser`:**
  - `build_converter(cfg=None)` (RapidOCR, backend `parsing.ocr_backend`)
  - `parse_mpdocvqa(doc_id, converter=None, cfg=None)` / `parse_pdf(pdf_path, doc_id=None, converter=None, cfg=None)` → `(list[Component], stats)`
  - `docling_bbox_to_pixels(...)`, `load_manifest(doc_id)`, `abs_path(rel)`, `safe_doc_id(stem)`
- **`docscout.ingestion.serializer`:** `serialize_components(components, tables=None, max_tokens=None)`, `table_text_form(df, title, caption)`, `truncate_tokens(text, n)`.
- **`docscout.graph.builder`:** `build_graph`, `save_graph`, `load_graph`, `parent_section`, `siblings(G, n, k=1)`, `caption_of`, `linked_component`, `path_to_root`. Node attribute `kind` ∈ {document, page, section, block}.
- **`scripts.check_components.validate(path) -> dict`:** reusable after later ingests.
- **Files:**
  - `data/processed/graphs/{doc_id}.pkl`
  - `data/processed/docs/{doc_id}.json`
  - `data/processed/tables/{component_id}.parquet`
  - `data/processed/figures/{component_id}.png`
  - `data/processed/mpdocvqa_pdfs/{doc_id}.pdf`
  - `data/processed/pages/{doc_id}/page_{n}.png` (PDF source)
- **Config keys added in P2:** `paths.docs_dir`, `paths.pages_dir`, `paths.combined_pdfs_dir`, `parsing.mpdocvqa_dpi` (200), `parsing.ocr_backend` (onnxruntime), `parsing.caption_max_gap` (0.15), `parsing.num_threads` (8).

### 8. Known limitations / open questions
1. **Speed:** 17.8 s/page with onnxruntime. OCR is still the main cost. The full subset takes about 5 h (resumable).
2. **Docling layout errors on scans** (not caused by our code):
   - charts get split into bar fragments plus axis-label paragraphs, and axis text can be labelled "caption";
   - a chart label can be merged with a body-text column (`ttwl0228_p5_paragraph_10`; both regions recorded);
   - top-of-page titles are often labelled `page_header`.
3. **OCR noise** (for example "Gambaccia ni", "DesignW rite") may hurt BM25 exact matching.
4. **The demo-PDF path is untested on a real file** (no PDF found). Rendering at 144 dpi and the coordinate conversion are unit-tested only.
5. Section detection depends on Docling headings; blocks before the first heading attach to their Page.

### 9. What I need from you before the next phase
1. **The demo PDF:** please check where it went. `data/raw/pdfs/` in `C:\Users\Samridhaa\OneDrive\Desktop\Doc-Scout` is empty in this session. If OneDrive shows it as "cloud only", right-click → "Always keep on this device", or copy it in again. Once it's there, I'll run (short):
   ```powershell
   .venv/Scripts/python -m scripts.ingest --source pdf
   .venv/Scripts/python -m scripts.visualize_components --doc <pdf-doc-id>
   ```
   Then I'll open one table page and one chart page, print a table's parquet head and `text_form` and a figure's caption link, report s/page, and update sections 4 and 6.
2. **The full ingest** (not started), now about **5.2 h** with onnxruntime. It's resumable, so it can run overnight or in parts:
   ```powershell
   .venv/Scripts/python -m scripts.ingest --source mpdocvqa
   ```
3. **Optional, look at the overlays:** `data/processed/debug/jjvg0227/page_5.png` and `data/processed/debug/ttwl0228/page_5.png`.
4. P2 is **not committed**. When you approve, I'll commit with your message and update `docs/PROGRESS.md`.

### 10. Fixes made during this phase (first review round)
| # | Problem found (how) | Change (file: one line) | Rerun result |
|---|---|---|---|
| 1 | `ttwl0228` p5 overlay: two right-column text blocks had no box. Docling merged them into `paragraph_7`/`paragraph_9` as a second provenance region, and only `prov[0]` was used | `parser.py`: store every provenance box in `meta["regions"]`; `visualize_components.py`: draw extra regions with a `+` label; data check validates regions | 4 multi-region items; the overlay shows `+p5_paragraph_7` and `+p5_paragraph_10`; all regions inside the page |
| 2 | All 4 `jjvg0227` tables had Docling caption links to the running header "Page N January, 1970" | `parser.py`: relabel captions matching `^page \d+` to `page_marker`; `test_page_marker_regex` | captions_docling 4 → 0 on `jjvg0227`; no false table captions |
| 3 | After fix 2, `jjvg0227` tables had no title in `text_form` | `serializer.py`: `title_above()` (skips page markers and footers; page headers allowed); `text_form` = `Table: <title>` + `Caption: <caption>` | `Table: Missouri Food Donation Program By MASW Divisions and Counties` on all 4 tables |
| 4 | Re-serializing needed a full re-parse | `ingest.py`: `--reserialize` | `Re-serialized 108 components and rebuilt 3 graphs.` |
| 5 | `pytest` failed to start once (the shell had moved into `site-packages`) | Environment only: ran from the repo root | tests pass |
