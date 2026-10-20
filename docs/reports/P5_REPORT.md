## PHASE REPORT: P5 Query classifier, normalization, fusion, reranking, full ablation

### 0. Summary in plain words
- **Built:** a cached, retrying Groq LLM client; a query-type classifier (regex rules, with the LLM deciding when the rules are unsure); hybrid retrieval that scores every component of the document with BM25 + BGE (+ CLIP for figures); per-list min-max normalization; weighted-sum and RRF fusion with soft or hard routing; and a cross-encoder rerank of the top 30.
- **Evaluation setup:** a document-level dev/test split (10 / 39 documents, 76 / 217 questions). Weights were tuned on dev only, and every main table is reported on TEST.
- **On TEST, hybrid_weighted_soft beats every single retriever** on R@1 and MRR at page and component level. Component MRR rises from 0.544 (BGE) to 0.599, which is significant (p = 0.003). The page MRR gain (0.756 → 0.788) is not significant (p = 0.077).
- **Rerank adds the largest gain:** page MRR 0.828 and component MRR 0.653, both significant against BGE.
- **Other ablations:** soft routing beats hard (hard leaves 34 test questions with nothing to rank), and weighted-sum ≥ RRF.
- **Classifier:** combined accuracy 0.751 and macro-F1 0.652 against DocVQA question_types (100% join coverage). A cached rerun makes 0 LLM calls.
- **One result against expectations:** on DEV (and therefore on "all 293"), page-level hybrid is below BGE. The cause is the min-max "lone table gets 1.0" effect (section 8.1, examples in 4.8).
- **LLM model change:** the requested `llama-3.1-8b-instant` does not exist on this Groq key. With your approval it was replaced by `openai/gpt-oss-20b`.

### 1. Files created/changed
New:
- `src/docscout/reasoning/llm_client.py`: `LLMProvider` interface plus `GroqProvider`; `LLMClient` with a JSON helper, sha256 disk cache, backoff on 429/5xx, minimum call gap and counters.
- `src/docscout/query/classifier.py`: `rule_matches`, `rule_classify`, `llm_classify`, `classify`, `weights_for`.
- `src/docscout/retrieval/retrievers.py`: `Candidate` (every stage score) and `HybridRetriever.retrieve(query, doc_id, branches)`.
- `src/docscout/retrieval/normalize.py`: `minmax`, `zscore`, `normalize_candidates` (per (branch, retriever) list).
- `src/docscout/retrieval/fusion.py`: `weighted_sum`, `rrf`, `hard_route`, `rank_candidates`.
- `src/docscout/rerank/cross_encoder.py`: `CrossEncoderReranker` (model loaded once, batched) and `rerank_candidates` (top-N cut-off).
- `eval/split.py`: document-level dev/test split → `eval/results/split.json`.
- `eval/eval_classifier.py`: DocVQA question_types join, label mapping, metrics for 3 classifier modes, latency.
- `eval/tune_fusion.py`: 81-point grid on DEV only → `eval/results/tuning.json`.
- `eval/plot_results.py`: `eval/results/ablation.png` (TEST).
- `tests/test_retrieval.py`, `tests/test_classifier.py`, `tests/test_llm_client.py`, `tests/test_split.py`: 39 tests.

Changed:
- `eval/run_eval.py` (P4 file, extended as requested):
  - adds the hybrid systems (`HYBRID_SYSTEMS`), per-split metrics (test / dev / all), p90 latency, full runs saved, the `--significance` ranx t-test and `--summary-only`
  - `summary.md` is now TEST-first, with all-293 as a secondary table
- `configs/config.yaml`:
  - `llm`: model `openai/gpt-oss-20b`, `max_retries 5`, `backoff_base_s`, `backoff_max_s`, `min_interval_s 2.1`, `timeout_s`, `reasoning_effort low`
  - `classifier.unsure_fallback`, `eval.split_file`
  - tuned `fusion.retriever_weights` / `branch_weights`
- `.env` (local, gitignored) and `.env.example`: `LLM_MODEL=openai/gpt-oss-20b`. Only that line changed; the key was not touched or printed.
- Regenerated: `eval/results/summary.md`. The `eval/results/*.json` files are gitignored.

### 2. Design decisions and any deviations from the architecture (and why)
1. **LLM model:** Groq returned 404 `model_not_found` for `llama-3.1-8b-instant`. The key lists no Llama chat models (available: gpt-oss-20b/120b, qwen3.8-27b, allam-2-7b). I asked, and you chose `openai/gpt-oss-20b`. Because it is a reasoning model, `reasoning_effort: low` is sent (only to gpt-oss models) to save hidden tokens.
2. **Account limits:** the Groq headers show 1,000 requests/day and 8,000 tokens/min. At about 285 tokens per classifier call, `min_interval_s = 2.1` keeps under the TPM limit. 0 retries were needed.
3. **Cache key:** sha256 of `[provider, model, messages (system + user), temperature]`, stored as JSONL in `data/processed/llm_cache.jsonl` (append-only; a torn last line is skipped).
4. **Retries:** `RetryableError` covers 429, 5xx and connection/timeout errors. The wait is `min(60, 2·2^attempt)`, or the server's Retry-After if larger. Other errors (e.g. 401/404) fail immediately with `LLMError`. `classify()` lets `LLMError` propagate, so a dead LLM stops the run instead of silently falling back.
5. **Rules:** `\d` counts as numerical, as specified, so "What was the value in 1975?" is numerical. Rules decide alone only when exactly one group matches (91/293); the other 202 go to the LLM.
6. **BM25 zeros:** every component of the document is scored, and a missing BM25 match is 0. In RRF a BM25 score of 0 is **not ranked** (it means "no term matched"). Otherwise the 19 zero-score paragraphs in a document would get arbitrary ranks from id order.
7. **CLIP weight:** CLIP is only in the figure branch, so figure retriever weights are renormalized over BM25 + BGE + CLIP (0.7 / 0.3 / 0.3 → 0.538 / 0.231 / 0.231). Text and table renormalize over BM25 + BGE.
8. **Hard routing is strict:** only the predicted branch's candidates are ranked. When a document has no component in that branch (for example, numerical but no table), the run is empty: 34 of 217 test questions. That is the real cost of hard routing, so I did not add a fallback.
9. **Rerank tail scores:** the head gets its CE score; tail item *i* gets `min(head CE) − 1 − i`. Final scores therefore stay strictly decreasing, and page = max(component) stays consistent with the component order.
10. **Classifier excluded from timing:** query types are computed once before the timed loop (cache hits) and looked up by `(question, doc_id)`.
11. **Saved runs:** result files now store **full** runs (not top-50), so the t-test sees the exact rankings. The top-level `metrics` key is now the TEST split; `splits` has test / dev / all.
12. **Plot:** shows page R@1, page MRR, component R@1 and component MRR instead of the R@5 in PHASES.md. Page R@5 is near ceiling (random = 0.77–0.80) and hides the differences.
13. **Tuning grid (as specified):**
    - BM25 ∈ {0.3, 0.5, 0.7} (BGE = 1 − BM25, CLIP 0.3).
    - The own-branch weight per type ∈ {0.4, 0.6, 0.8}. The remainder is split in fixed ratios: textual table:figure = 5:3, numerical text:figure = 3:1, visual text:table = 5:3.
    - 81 settings in total.
    - The best point is at the grid edge for BM25 (0.7) and textual-text (0.4). I did not extend the grid (see 8.4).

### 3. How to run (exact commands, in order, as a teammate would on a fresh clone)
Prerequisites: P1–P4 done (qrels and indexes built), and `.env` with `LLM_PROVIDER=groq`, `LLM_API_KEY=...`, `LLM_MODEL=openai/gpt-oss-20b`.
```
# only behind HTTPS-scanning antivirus (Avast on this machine): $env:SSL_CERT_FILE = "C:\path\to\ca-bundle.pem"
.venv/Scripts/python -m eval.split
.venv/Scripts/python -m eval.eval_classifier          # first run ~10 min (about 290 Groq calls), then cached
.venv/Scripts/python -m eval.eval_classifier          # rerun: 0 LLM calls
.venv/Scripts/python -m eval.tune_fusion              # DEV only; copy the CHOSEN values into configs/config.yaml
.venv/Scripts/python -m eval.run_eval --systems bm25_all dense_minilm_all dense_bge_all
.venv/Scripts/python -m eval.run_eval --systems hybrid_weighted_soft hybrid_rrf_soft hybrid_weighted_hard hybrid_weighted_soft_rerank hybrid_weighted_soft_goldtype --significance
.venv/Scripts/python -m eval.plot_results
.venv/Scripts/python -m pytest
```
The cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) downloads into the HF cache on first use. It took 36 s here, including the first load.

### 4. Actual output (from this session, trimmed)
**4.1 Groq smoke test + rate limits**
```
provider groq model llama-3.1-8b-instant
groq.NotFoundError: Error code: 404 - {'error': {'message': 'The model `llama-3.1-8b-instant` does not exist or you do not have access to it.' ...
(after switching)
provider groq model openai/gpt-oss-20b
{'ok': True}
LLM calls=1 cache_hits=0 retries=0 tokens=153 (prompt 115, completion 38) 0.81s
{'x-ratelimit-limit-requests': '1000', 'x-ratelimit-limit-tokens': '8000'}
```
**4.2 Split**
```
dev  : 10 documents,  76 questions,  70 from documents with >= 2 pages
test : 39 documents, 217 questions, 176 from documents with >= 2 pages
total: 293 questions | document overlap: 0 | seed 42, dev fraction 0.2
```
`split.json` sample: `{"seed": 42, "dev_fraction": 0.2, "dev": {"doc_ids": ["fqvw0217", "gyfb0228", "hnhd0227", ...], "question_ids": ["49168", "49172", "49177", ...]}, "test": {...}}`

**4.3 Classifier (first, uncached run)**
```
gold join: 293/293 questions found in lmms-lab/DocVQA/DocVQA/validation -> coverage 100.0% (102.2s)
mapping table (DocVQA question_types, gold answer kind -> our label): count
  table/list                               numeric      -> numerical  49
  layout                                   non-numeric  -> textual    47
  table/list                               non-numeric  -> numerical  42
  free_text                                non-numeric  -> textual    42
  form                                     non-numeric  -> textual    24
  layout                                   numeric      -> numerical  20
  form                                     numeric      -> numerical  15
  Image/Photo                              non-numeric  -> visual     7
  free_text                                numeric      -> numerical  7
  form + layout                            non-numeric  -> textual    6
  figure/diagram                           non-numeric  -> visual     6
  figure/diagram                           numeric      -> visual     4
  ... (14 more rare combinations, 1-4 questions each)
label counts (DocVQA question_types): textual=129, numerical=144, visual=20
labels: DocVQA question_types | n = 293 | combined sources: {'rules': 91, 'llm': 202}

rules-only (unsure -> textual): accuracy=0.614 macro-F1=0.521
  gold/pred      textual numerical    visual
  textual            112        12         5
  numerical           80        63         1
  visual              10         5         5
llm-only: accuracy=0.747 macro-F1=0.644
  textual            115        10         4
  numerical           46        98         0
  visual              10         4         6
combined (rules, LLM when unsure): accuracy=0.751 macro-F1=0.652
  textual            107        16         6
  numerical           37       106         1
  visual               6         7         7

LLM calls=275 cache_hits=220 retries=0 tokens=79427 (prompt 66488, completion 12939) | wall time 578.7s | model groq/openai/gpt-oss-20b
classifier latency (ms/query, median): combined_median_ms=1887.93, rules_decided_median_ms=0.43, llm_cached_median_ms=0.44, llm_uncached_median_ms=2090.36
```
Rerun:
```
LLM calls=0 cache_hits=495 retries=0 tokens=0 (prompt 0, completion 0) | wall time 0.2s | model groq/openai/gpt-oss-20b
classifier latency (ms/query, median): combined_median_ms=0.32, rules_decided_median_ms=0.30, llm_cached_median_ms=0.32, llm_uncached_median_ms=2090.36
real	0m3.811s
```
(Before the full run, a `--limit 10` check made 10 calls and 2,847 tokens.)

**4.4 Tuning (DEV only)**
```
DEV only: 76 questions, 10 documents | page>=2: 70 | component: 69
query types: {'numerical': 34, 'textual': 38, 'visual': 4} | LLM calls=0 cache_hits=52
current config (before tuning): objective=0.4753 page_mrr=0.5623 comp_mrr=0.3884
rank  bm25 txt.text num.table vis.fig objective  pageMRR  compMRR
   1   0.7      0.4       0.6     0.6    0.5133   0.5860   0.4406
   2   0.7      0.4       0.6     0.8    0.5133   0.5860   0.4406
   3   0.7      0.4       0.8     0.6    0.5031   0.5691   0.4371
   5   0.7      0.4       0.6     0.4    0.5013   0.5746   0.4279
   6   0.5      0.4       0.6     0.6    0.5004   0.5796   0.4213
   8   0.7      0.6       0.6     0.6    0.4962   0.5827   0.4096
  10   0.7      0.8       0.6     0.6    0.4930   0.5897   0.3963
best objective by bm25            : 0.3->0.4888  0.5->0.5004  0.7->0.5133
best objective by textual_text    : 0.4->0.5133  0.6->0.4962  0.8->0.4930
best objective by numerical_table : 0.4->0.4888  0.6->0.5133  0.8->0.5031
best objective by visual_figure   : 0.4->0.5013  0.6->0.5133  0.8->0.5133
CHOSEN: retriever_weights={'bm25': 0.7, 'dense': 0.3, 'clip': 0.3}
        textual   {'text': 0.4, 'table': 0.375, 'figure': 0.225}
        numerical {'text': 0.3, 'table': 0.6, 'figure': 0.1}
        visual    {'text': 0.25, 'table': 0.15, 'figure': 0.6}
```
Visual 0.6 vs 0.8 tie: only 4 dev questions are visual. The tie goes to the earlier grid point, 0.6.

**4.5 Systems and significance (TEST)**
```
[hybrid_weighted_soft_goldtype] median 57.2 ms, p90 66.4 ms | empty runs 0 | TEST page>=2: R@1=0.670 MRR=0.795 | TEST comp: R@1=0.449 MRR=0.607
t-test hybrid_weighted_soft         page MRR (>=2 pages)   0.788 vs 0.756 p=0.0765 n.s.
t-test hybrid_weighted_soft_rerank  page MRR (>=2 pages)   0.828 vs 0.756 p=0.0002 SIGNIFICANT
t-test hybrid_weighted_soft         component MRR          0.599 vs 0.544 p=0.0032 SIGNIFICANT
t-test hybrid_weighted_soft_rerank  component MRR          0.653 vs 0.544 p=0.0000 SIGNIFICANT
```
**4.6 `eval/results/summary.md` (main tables, TEST)**

Page level, documents with ≥ 2 pages (176 questions):

| system | R@1 | MRR | R@5 | R@10 | nDCG@10 | median ms | p90 ms |
|---|---|---|---|---|---|---|---|
| random (expected) | 0.270 | 0.501 | 0.801 | 0.966 | - | - | - |
| bm25_all | 0.636 | 0.718 | 0.841 | 0.852 | 0.751 | 0.2 | 0.5 |
| dense_minilm_all | 0.568 | 0.725 | 0.932 | 0.983 | 0.788 | 8.9 | 11.8 |
| dense_bge_all | 0.631 | 0.756 | 0.926 | 0.983 | 0.810 | 35.6 | 40.4 |
| hybrid_weighted_soft | 0.659 | 0.788 | 0.966 | 0.994 | 0.839 | 54.3 | 60.2 |
| hybrid_rrf_soft | 0.659 | 0.786 | 0.955 | 0.994 | 0.837 | 54.8 | 60.5 |
| hybrid_weighted_hard | 0.602 | 0.696 | 0.835 | 0.875 | 0.740 | 36.1 | 43.3 |
| hybrid_weighted_soft_rerank | **0.733** | **0.828** | 0.949 | 0.994 | 0.869 | 762.7 | 1850.9 |
| hybrid_weighted_soft_goldtype (analysis) | 0.670 | 0.795 | 0.966 | 0.994 | 0.844 | 57.2 | 66.4 |

Component level (209 questions):

| system | R@1 | MRR | R@5 | R@10 | nDCG@10 |
|---|---|---|---|---|---|
| bm25_all | 0.417 | 0.552 | 0.612 | 0.648 | 0.556 |
| dense_minilm_all | 0.259 | 0.465 | 0.621 | 0.764 | 0.524 |
| dense_bge_all | 0.362 | 0.544 | 0.670 | 0.755 | 0.582 |
| hybrid_weighted_soft | 0.432 | 0.599 | 0.701 | 0.763 | 0.624 |
| hybrid_rrf_soft | 0.398 | 0.581 | 0.698 | 0.793 | 0.612 |
| hybrid_weighted_hard | 0.360 | 0.473 | 0.491 | 0.519 | 0.458 |
| hybrid_weighted_soft_rerank | **0.488** | **0.653** | 0.716 | 0.805 | 0.677 |
| hybrid_weighted_soft_goldtype (analysis) | 0.449 | 0.607 | 0.694 | 0.757 | 0.628 |

Secondary, all 293 questions (dev was used for tuning):

| system | page≥2 R@1 | page≥2 MRR | comp R@1 | comp MRR |
|---|---|---|---|---|
| bm25_all | 0.593 | 0.679 | 0.380 | 0.522 |
| dense_minilm_all | 0.537 | 0.696 | 0.244 | 0.448 |
| dense_bge_all | 0.606 | 0.738 | 0.331 | 0.514 |
| hybrid_weighted_soft | 0.585 | 0.730 | 0.382 | 0.560 |
| hybrid_rrf_soft | 0.573 | 0.716 | 0.350 | 0.540 |
| hybrid_weighted_hard | 0.524 | 0.630 | 0.313 | 0.428 |
| hybrid_weighted_soft_rerank | 0.683 | 0.796 | 0.437 | 0.610 |
| hybrid_weighted_soft_goldtype | 0.638 | 0.765 | 0.432 | 0.597 |

The full file, which also has R@5, R@10, nDCG@10, the page-all table and the t-tests, is at `eval/results/summary.md`.

**4.7 Latency** (ms/query, models loaded, classifier excluded; from the run above)
- bm25 0.2 / 0.5; MiniLM 8.9 / 11.8; BGE 35.6 / 40.4 (median / p90).
- Hybrid soft 54.3 / 60.2 (BGE + CLIP query encoding plus fusion over the whole document); hard 36.1 / 43.3 (no CLIP unless visual).
- **Rerank 762.7 / 1850.9.**
- Classifier, measured separately:
  - rules-decided: 0.30 ms
  - cached LLM: 0.32 ms
  - uncached LLM: median 2,090 ms in the batch run, which includes the up-to-2.1 s rate-limit gap; a single call took 0.81 s.

**4.8 Result against expectations: hybrid < BGE at page level on DEV / all-293**
```
DEV (tuned on) page>=2 MRR / R@1 | comp MRR / R@1
  dense_bge_all                    0.693 / 0.543 | 0.422 / 0.237
  hybrid_weighted_soft             0.586 / 0.400 | 0.441 / 0.229
  hybrid_weighted_soft_rerank      0.715 / 0.557 | 0.480 / 0.280
  hybrid_weighted_soft_goldtype    0.691 / 0.557 | 0.566 / 0.382
DEV page R@1: BGE right & hybrid wrong 14 | hybrid right & BGE wrong 4
  hybrid top-1 type on the lost ones: Counter({'paragraph': 9, 'table': 5}) pred types: Counter({'numerical': 7, 'textual': 7})
dev questions per doc: [('fqvw0217', 16), ('rnbx0223', 13), ('txpp0227', 11), ('mybv0228', 11)]
dev:  numerical questions with a table at rank 1 scored exactly 0.6 (norm=1 from both retrievers): 11, page wrong: 6, of which BGE had page right: 3
test: numerical questions with a table at rank 1 scored exactly 0.6 (norm=1 from both retrievers): 46, page wrong: 13, of which BGE had page right: 4
```
Example 1: question 49172 (dev, predicted and gold type numerical, document txpp0227, 3 pages, answer on p10). Q: "Who is 'presiding' TRRF GENERAL SESSION (PART 1)?" Gold: "lee a. waller".

| rank | BGE (before) | hybrid_weighted_soft (after) |
|---|---|---|
| 1 | 0.795 `p10_paragraph_4` "TRRF GENERAL SESSION (PART I) Presiding: Lee A. Waller ..." ✓ | 0.600 `p9_table_0` "Table: MONDAY, MAY 15 Rows: 8:15 to 8:56 a.m. ..." ✗ |
| 2 | 0.580 `p10_paragraph_8` | 0.300 `p10_paragraph_4` ✓ |
| 3 | 0.559 `p10_paragraph_5` | 0.102 `p10_paragraph_18` |

Rerank puts `p10_paragraph_4` back at rank 1 (CE = 8.512, next −1.795).

Example 2: question 57428 (dev, gold visual, predicted textual: a classifier error). Q: "Which branch of Scissors has been launched on Kerala and Tamil Nadu?" Gold: "Menthol Fresh". BGE top-3: `p208_figure_5` "4 2 SCISSORS" ✓, `p208_figure_3`, `p208_paragraph_4`. Hybrid top-3: `p206_paragraph_3` "After the success of Mint and Orange variants ..." ✗, `p208_figure_5` (0.225), `p208_paragraph_4`. Under the textual weights (figure 0.225), the strongest text match on the wrong page wins.

### 5. Tests
`87 passed in 22.48s` (48 from earlier phases + 39 new; no network and no model downloads; the LLM is mocked).
- `test_retrieval.py`:
  - min-max basic, all-equal (→ 0), single item (→ 1), empty; z-score
  - per-(branch, retriever) normalization of hand-made candidates
  - retriever-weight renormalization (figure /1.3)
  - weighted_sum textual order worked by hand (t1, tb1, t2, t3, f2, f1, with the t2/t3 tie broken by id)
  - numerical lone-table effect (tb1 0.6 > t1 0.3)
  - RRF order worked by hand (t1, t3, t2, tb1, f2, f1), with BM25 zeros unranked
  - RRF invariant to raw score scale and never normalizes
  - tie-break by id
  - hard vs soft routing
  - **rerank keeps items 31–35 in fused order** and final scores strictly decreasing
  - fake-index retriever: only the document's components; BM25 = 0 kept; query encoded once
- `test_classifier.py`:
  - 8 rule cases
  - 5 unsure cases: neither, both, and the word-boundary traps photosynthesis / totally / figured
  - the LLM is called only when the rules are unsure (mock)
  - fallback when the LLM is off; an invalid label raises; confidence is clamped
  - `weights_for`
  - gold mapping priority and the numeric/date answer parser
- `test_llm_client.py`:
  - cache miss → hit, and persisted across client instances
  - the key depends on model, prompt and temperature
  - backoff on a fake 429: sleeps [2, 4], then success
  - gives up after max retries on a 503 (sleeps [2, 4, 8])
  - a non-retryable error is not retried
  - minimum gap between real calls (fake clock; a cache hit doesn't wait)
  - the JSON helper rejects non-JSON
- `test_split.py`: the synthetic split has no document overlap, assigns every question once and is deterministic; the saved `split.json` has no doc or question overlap.

### 6. Acceptance criteria and reviewer checks
- PASS: classifier accuracy table printed. 3 modes × accuracy / macro-F1 / confusion matrix (4.3).
- PASS: summary.md shows all 7 systems, plus the goldtype analysis row (4.6).
- PASS (TEST): hybrid beats single retrievers. Page R@1 0.659 vs best single 0.636 (BM25); page MRR 0.788 vs 0.756 (BGE); component R@1 0.432 vs 0.417; component MRR 0.599 vs 0.552. The page MRR gain over BGE is not significant (p = 0.077), and on DEV / all-293 page level hybrid is below BGE (explained in 4.8 and 8.1).
- PASS: rerank improves component-level metrics. R@1 0.432 → 0.488, MRR 0.599 → 0.653; page level improves too (MRR 0.788 → 0.828).
- PASS: soft ≥ hard. Page MRR 0.788 vs 0.696; component MRR 0.599 vs 0.473.
- PASS: `eval/results/ablation.png` saved and inspected (labels readable, one colour per metric, value labels).
- PASS: results against expectations are explained without tuning on test (4.8). The tuning script asserts that no non-dev document is loaded (`tune_fusion.py`: "a non-dev document leaked into tuning").
- PASS: normalization comes before the weighted sum; RRF uses ranks only (`rank_candidates` normalizes only for weighted_sum; `test_rrf_uses_ranks_only`).
- PASS: hard and soft routing both reported (two rows in every table).
- PASS: weights tuned only on dev (`tuning.json` `"split": "dev"`, 76 questions; config comment).
- PASS (with caveat): rerank latency on CPU for the top 30 is median 763 ms and p90 1,851 ms per query. That is fine for an interactive demo, but p90 is close to 2 s (8.5).
- PASS: LLM calls are cached. The rerun made 0 calls with 495 cache hits; classification took 0.2 s and the whole command 3.8 s.
- PASS: dev/test split by document with 0 overlap (4.2, `test_split.py`).
- PASS: significance tests run (4.5).
- PASS: latency median and p90 per system, classifier reported separately (4.7).

### 7. Interfaces for the next phase
- `docscout.reasoning.llm_client`:
  - `get_client(cfg=None) -> LLMClient` raises `MissingKeyError` / `LLMError`
  - `client.complete(prompt, system=None, json_mode=False) -> str`; `client.complete_json(prompt, system=None) -> dict`
  - `client.stats` (`calls`, `cache_hits`, `retries`, `prompt_tokens`, `completion_tokens`)
  - to add a provider: subclass `LLMProvider.complete(messages, model, temperature, max_tokens, json_mode) -> (text, usage)` and register it in `PROVIDERS`
- `docscout.query.classifier`:
  - `classify(query, client=None, use_llm=None) -> {"type", "confidence", "source", ...}`
  - `weights_for(type) -> {text, table, figure}`; `LABEL_TO_BRANCH`
- `docscout.retrieval.retrievers`:
  - `HybridRetriever(load=None, comp_meta=None, cfg=None).retrieve(query, doc_id, branches=None) -> List[Candidate]`
  - `Candidate(id, doc_id, page, type, branch, raw, norm, fused, rrf, rerank, final)` with `.stage_scores()` for P6 provenance
- `docscout.retrieval.fusion`:
  - `rank_candidates(cands, qtype, method, routing, retriever_weights, branch_weights, rrf_k=60, normalization="minmax") -> List[Candidate]`
  - also `weighted_sum`, `rrf`, `hard_route`
- `docscout.rerank.cross_encoder`: `CrossEncoderReranker(cfg=cfg).rerank(query, ranked, texts) -> List[Candidate]` (top `retrieval.rerank_top_n`). The P6 feedback boost should be added to `c.final` after this, then re-sorted.
- `eval.split.load_split()` → `{"dev": {...}, "test": {...}}`. Result JSON: `{system, splits: {test, dev, all: {page_all, page_multi, component}}, latency_ms_median, latency_ms_p90, empty_runs, runs (full)}`.
- Config keys:
  - `llm.{provider, model, temperature, max_tokens, max_retries, backoff_base_s, backoff_max_s, min_interval_s, timeout_s, reasoning_effort, cache_path}`
  - `classifier.{use_llm, rule_confidence, unsure_fallback}`
  - `fusion.{method, normalization, rrf_k, retriever_weights, branch_weights}`
  - `retrieval.rerank_top_n`, `eval.split_file`

### 8. Known limitations / open questions
1. **Min-max "lone table" effect: seen, and it hurts on DEV.** Min-max gives the best table 1.0 from both retrievers regardless of how good it is, so a numerical query always gets the top table at exactly `w_table = 0.6`. The best paragraph can reach at most 0.3. In example 49172 the paragraph has the highest BM25 in the whole document (33.5 vs the table's 12.0) but still loses 0.300 to 0.600.
   - DEV: 11 numerical questions had a table at rank 1 at exactly 0.6; 6 had the wrong page, 3 of which BGE got right.
   - TEST: 46 such questions, 13 wrong, 4 that BGE got right. On test, a table at rank 1 is mostly correct (page right for 47 of 71 top-1 tables), so the net effect there is small.
   - Dev is 10 documents dominated by 4 (fqvw0217 16 questions, rnbx0223 13, txpp0227 11, mybv0228 11). txpp0227 has schedule tables on p8/p9 while the answers are paragraphs on p10.
   - Rerank fixes most of it (dev page MRR 0.715 > BGE 0.693).
   - Possible P6+ fixes, not applied (that would be tuning after seeing results): z-score normalization (already an option), or scaling a branch's normalized scores by its raw best score.
2. **The classifier misses visual and under-calls numerical.** Visual recall is 7/20. 37 numerical questions are predicted textual, mostly because gold "numerical" includes `table/list` questions with text answers (e.g. "What is the name of the Dealer?" → "A. C. Monk"), which neither the rules nor the LLM can see from the question.
   - The gold-type row shows what perfect types would add: TEST page MRR +0.007 and component MRR +0.008 (small); DEV +0.105 / +0.125 (large).
3. **Hybrid vs BGE page MRR on TEST is not significant** (p = 0.077). Hybrid + rerank is significant at both levels.
4. **Tuning:**
   - the best point is on the grid edge (BM25 0.7, textual text weight 0.4), so a wider grid might do better on dev
   - the dev set is small (76 questions, 4 visual), and the visual weight is effectively untuned (0.6 / 0.8 tie)
   - RRF uses the same tuned branch weights and was not tuned separately
5. **Rerank latency:** median 0.76 s and p90 1.85 s on CPU, driven by long `text_form`s (tables up to 512 tokens) × 30 pairs. A shorter CE `max_length` (e.g. 256) or a smaller top-N would speed it up; not tried, to avoid changing results.
6. **Hard routing** returns nothing for 34/217 test questions (numerical, with no table in the document). This is by design for the ablation.
7. **Groq free tier:** 1,000 requests/day and 8,000 tokens/min. The classifier is cached, but P6 answer generation will share this budget.
8. **The Avast TLS workaround** (`SSL_CERT_FILE`) was again needed on this machine for Groq and HF; the bundle is in my session scratchpad, not the repo.
9. `eval/results/split.json` (and every `eval/results/*.json`) is gitignored. It is reproducible with `python -m eval.split` (seed 42), but you may want it committed.

### 9. What I need from you before the next phase
1. **Review** `eval/results/summary.md` and `eval/results/ablation.png`, especially limitation 8.1 (min-max lone table) and whether P6 should use the rerank system as the default pipeline (my recommendation).
2. **Decide on `split.json` in git:** add `!eval/results/split.json` to `.gitignore` so teammates share the exact split? (Not done; it changes an earlier phase's file.)
3. **Confirm the LLM** `openai/gpt-oss-20b` for P6 grounded answers too, or choose `openai/gpt-oss-120b` for answering (better quality, more of the daily token budget).
4. **Demo PDF:** `data/raw/pdfs/` is still empty (needed for P6/P7).
5. P5 is **not committed.** Approve it and give a commit message when ready.

### 10. Fixes made during this phase
| # | Issue (how found) | Fix | Rerun result |
|---|---|---|---|
| 1 | First Groq call: `APIConnectionError`; httpx showed `CERTIFICATE_VERIFY_FAILED` (Avast TLS interception, as in P1) | Built a certifi + Windows-roots bundle in the session scratchpad; set `SSL_CERT_FILE` per command only (nothing in the repo) | HTTPS OK; the next error was a 404 (fix 2) |
| 2 | `404 model_not_found` for `llama-3.1-8b-instant`; the model list has no Llama chat models | Asked you → `openai/gpt-oss-20b` in `.env` (LLM_MODEL line only), `.env.example` and config; added `llm.reasoning_effort: low` | `{'ok': True}`, 153 tokens, 0.81 s |
| 3 | `eval_classifier.py` would not compile on Python 3.10 (backslash inside an f-string expression) | Moved the header text into a variable | `--limit 10` run OK, then the full run |
| 4 | 2 new tests failed: `(0.6−0.4)/(0.8−0.4) = 0.4999…` broke the hand-worked tie (test data, not code) | Test BGE values changed to 0.75 / 0.5 / 0.25 (exact in binary floats) | 39/39 new tests pass |
| 5 | summary.md t-test table had two "system" column headers (read the generated file) | Columns renamed to "system score" / "baseline score" | `--summary-only` regenerated |
| 6 | ablation.png title clipped at the right edge (opened the PNG) | Two-line, shorter title (two attempts: the first heredoc turned `\n` into a real newline and broke the string) | Re-rendered and inspected: fully visible |

### 11. Logic explained
**Classifier decision flow.**
1. Run the word-boundary regexes. NUMERICAL: `\d`, how much/many, total(s), percent/percentage/%, amount(s), number of, date(s), year(s), cost(s), price(s). VISUAL: chart, graph, figure, diagram, image, logo, photo(graph), picture (each with an optional plural).
2. Exactly one group matches → that label, confidence 0.9 (`classifier.rule_confidence`), source "rules".
3. Both or neither match → rules are "unsure". The LLM gets one-line definitions of the 3 labels and returns JSON `{"type", "confidence"}`; the label is validated and the confidence clamped to [0, 1]; source "llm".
4. If the LLM is disabled → `classifier.unsure_fallback` (textual). If it fails after retries → `LLMError` (stop).
5. `weights_for(type)` → `fusion.branch_weights[type]`.

**Normalization.** For each document and each (branch, retriever) list L, for example (text, bm25) over all text components of the document:
- min-max: `norm(c) = (s(c) − min L) / (max L − min L)`; if all scores are equal, 0 for every item, or 1 if |L| = 1
- z-score (option): `(s − mean) / std`

**Weighted sum.** `fused(c) = w_branch[type][branch(c)] · Σ_r w'_r · norm_r(c)`, where `w'_r = w_ret[r] / Σ_{r' ∈ retrievers of branch(c)} w_ret[r']`.

**RRF.** `rrf(c) = Σ_{(branch, r) lists containing c} w_branch[type][branch] / (60 + rank_{branch,r}(c))`. The rank is 1-based by raw score, ties by id, and BM25 = 0 is unranked. No normalization.

**Worked example on a real question:** 49172 "Who is 'presiding' TRRF GENERAL SESSION (PART 1)?" in document txpp0227 (26 components: 24 text, 2 tables). Predicted numerical → branch weights text 0.3, table 0.6; retriever weights BM25 0.7, BGE 0.3 (text and table renormalize to 0.7 / 0.3).
- Text lists: BM25 min 0 (19 of 24 are 0), max 33.538; BGE min 0.2545, max 0.7954. Table lists: BM25 {0.2643, 11.954}; BGE {0.4571, 0.5202}.
- `p10_paragraph_5`: raw BM25 6.735 → norm 6.735/33.538 = 0.201; raw BGE 0.559 → (0.559 − 0.2545)/(0.7954 − 0.2545) = 0.563. Fused = 0.3 · (0.7·0.201 + 0.3·0.563) = 0.3 · 0.310 = **0.0929**.
- `p10_paragraph_4` (the answer): norm 1.0 / 1.0 → 0.3 · 1.0 = **0.300**.
- `p9_table_0`: the larger of the 2 tables on both retrievers → norm 1.0 / 1.0 → 0.6 · 1.0 = **0.600**, so it ranks first.
- RRF: the table is rank 1 in table/bm25 and table/bge → 0.6/61 + 0.6/61 = 0.01967. The paragraph is rank 1 in text/bm25 and text/bge → 0.3/61 + 0.3/61 = 0.00984. RRF gives the same order (the second table, p8, also beats the paragraph at 0.01935).
- Rerank: CE(`p10_paragraph_4`) = 8.512 vs −1.795 for the next → the correct paragraph returns to rank 1.

**Hard vs soft routing.** Soft: every component of every branch is scored and fused; the label only sets `w_branch`. Hard: only the components of `LABEL_TO_BRANCH[type]` are ranked (textual → text, numerical → table, visual → figure). Normalization is per branch, so a branch's scores are identical in both modes. A document without that branch gives an empty ranking.

**Rerank cut-off.** Take the fused ranking and send the first `rerank_top_n = 30` (query, text_form) pairs to the CE in batches of 32. Re-sort those 30 by CE score (ties by id) and store `c.rerank`. Items 31+ keep their fused order after them, with `final = min(head CE) − 1 − i`, so the scores are strictly decreasing.

**Tuning objective.** `objective = ½ · (dev page MRR on documents with ≥ 2 pages + dev component MRR)`, using classifier-predicted types (as the system does).
- Retrieval and normalization are computed once per dev question; only fusion is re-run for each of the 81 settings.
- The best is chosen by objective, ties going to the earlier grid point.
- Test questions are never loaded (`tune_fusion` reads only `split["dev"]` and asserts that every question's document is a dev document).

**Split.**
1. Take the 49 evaluation documents (`eval_docs.txt` order), shuffle them with `random.Random(42)`, and put the first `round(0.2 · 49) = 10` in dev and the other 39 in test.
2. Every question follows its document: dev has 76 questions (70 from documents with ≥ 2 pages), test has 217 (176).

**Page scores from component scores.** `page_score(p) = max_{c on p} final(c)` over the system's component ranking (`eval.metrics.pages_from_components`); pages are sorted by that score, ties by page id. With rerank, the strictly decreasing `final` means a page's rank equals the rank of its best component.
