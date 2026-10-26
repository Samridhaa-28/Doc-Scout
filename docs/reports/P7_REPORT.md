## PHASE REPORT: P7 Relevance feedback (Rocchio), cross-session memory, API, React UI
Evidence: `docs/reports/p7_evidence/` · results: `eval/results/summary.md` (feedback sections), `eval/results/feedback_{dev,test,memory_dev}.json` · pre-registration: `docs/reports/P7_PREREGISTRATION.md`

### 0. Summary in plain words
- **Feedback:** you can mark any evidence item relevant or not relevant. Votes are stored in SQLite.
- **Session Rocchio:** a vote moves the BGE query vector toward the up-voted items and away from the down-voted ones. The document is then re-scored, re-fused and re-ranked: up-votes are pinned, down-votes are removed, and the answer is not regenerated.
- **Cross-session memory** (off by default) adds a small capped prior from votes on similar past questions.
- **Simulated feedback** (retrieval only):
  - DEV: residual MRR +0.091 [+0.011, +0.180], on only 34 questions.
  - TEST, run once with the pre-registered choice: +0.023 [−0.048, +0.091]. **The gain is not significant on TEST.**
- **Memory test (DEV):** related questions improve (22 pairs fired: 5 better, 0 worse). Unrelated questions are flat on average, but 5 of 34 pairs moved down slightly below rank 1.
- **P5/P6 numbers are unchanged:** feedback is off by default, and round 1 reproduces the P5 rankings.
- **API:** FastAPI with 5 endpoints. The React UI shows the answer and citations, re-ranks visibly after a vote, and marks the cited table cells on the real page.
- **Demo:** two votes plus "Re-answer" turn the numerical demo question from "Not found" into **62,050**. 160 tests pass.

### 1. Files created/changed
New
- `src/docscout/feedback/session_store.py` - SQLite `FeedbackStore`, one `votes` table (evidence + answer votes, question vector)
- `src/docscout/feedback/rocchio.py` - `rocchio_query`, `FeedbackHook` (P6 `feedback_adjust` hook), `CachedScorer`, `make_feedback_hook`
- `src/docscout/feedback/memory.py` - `memory_prior` (the one memory function) + `question_vector` helper
- `src/docscout/evidence/cell_boxes.py` - page-pixel boxes of localized cells via PyMuPDF (digital PDFs only)
- `api/main.py` - FastAPI app factory, `AppState`, `/ask /feedback /next /documents /page-image`
- `eval/eval_feedback.py` - simulated feedback on the residual collection, pre-registered rule, memory test, summary section
- `ui/` - Vite + React + TS app: `package.json`, `package-lock.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, `src/{main.tsx, App.tsx, api.ts, styles.css}`, `src/components/{AnswerPanel, EvidenceCard, PageViewer, ScoreBars}.tsx`
- `tests/test_rocchio.py`, `tests/test_memory.py`, `tests/test_api.py`, `tests/test_cell_boxes.py` - 27 new tests
- `docs/reports/P7_PREREGISTRATION.md` - selection rule, written before any DEV run
- `docs/reports/p7_evidence/` - demo slot runs, eval logs, API session, memory check, cell-box crop, 3 UI screenshots
- `.claude/launch.json` - dev-server entry for the UI preview (tooling only)

Changed (small, earlier phases)
- `src/docscout/pipeline.py` (P6) - uses `hook.deltas` / `hook.tags` when the hook has them; identity hook unchanged
- `src/docscout/evidence/provenance.py` (P6) - why-line `fb ±x | <tag>`; `scores.feedback` present whenever tagged
- `scripts/ask.py` (P6) - `--slots off|conditional|both` (needed for the step-0 comparison)
- `eval/run_eval.py` (P5) - appends the feedback section to summary.md (like the P6 ANLS section)
- `configs/config.yaml` - `feedback:` block replaced (the old `step`/`cap` were unused by code; kept and reused by memory), `api.cors_origins`, `api.default_store`, `api.demo_questions`
- `README.md` - API and UI section, feedback eval commands; `.gitignore` - `ui/node_modules/`, `ui/dist/`
- `eval/results/summary.md` - regenerated; only the two feedback sections were added (diff in 4.4)

### 2. Design decisions and any deviations from the architecture (and why)
1. **This spec replaces PHASES.md P7.** React instead of Streamlit, Rocchio instead of preference counts, your endpoint list. There is no `POST /ingest` (PDF upload) and no `POST /session`: `/ask` creates the session id when it's missing.
2. **`GET /documents` instead of `GET /docs`.** FastAPI serves Swagger at `/docs`.
3. **The Rocchio term (your change).** Over the cross-encoder top 30, `final = (1−λ)·minmax(CE) + λ·minmax(cos(q′, d))`. Before that, dense raw scores of every component of the document are re-scored with q′, then min-max normalized and weighted-sum fused, and the CE re-ranks the new top 30 with the original query text. λ = 0 is the literal pipeline. Items below the top 30 get `final = −1 − i`.
4. **Identity by default.** With no votes and no memory prior, the hook returns the list untouched, so `final` is the raw CE as in P6 (`test_no_votes_no_memory_is_identity_and_final_is_raw_ce`, 6/6 P6 pipeline tests pass).
5. **Session Rocchio scope** = votes in this session, on this document, for this question (normalized text). Carry-over to other questions is the memory's job.
6. **Pin and exclude.** Up-voted items get `final += 10` (above every other value) and keep their order among themselves. Down-voted items are removed from the ranking, so they are also gone from `/next`. `fb` in the why-line is the change on the normalized scale (`final − minmax(CE)`), without the pin offset.
7. **Memory only (no session votes):** λ = 0, so `final = minmax(CE) + prior`. The ±0.15 cap then lives on a [0, 1] scale. On raw CE logits (range about −11…10) it would be meaningless.
8. **Memory similarity uses plain question embeddings** (BGE without the retrieval instruction). The shared instruction prefix would raise every question–question cosine. Rocchio keeps the instruction-prefixed query vector, which is the right one for scoring documents. The vector is stored with each vote.
9. **Memory excludes the current session** (that is session Rocchio's job). It counts only evidence votes; answer votes are logged with `target='answer'` and never read for ranking.
10. **A vote never calls the LLM.** `/feedback` re-runs retrieval and re-ranking with `generate=False` and returns `answer_stale: true`. "Re-answer with current evidence" is a new `/ask` in the same session, so the votes shape the new evidence.
11. **`/next` has no new retrieval.** It pages through the stored post-feedback ranking and builds the card from the stored Candidate (provenance, graph path, locator for tables).
12. **The API serves both stores in one process.** Each pipeline gets index and graph loaders bound to its own config, because the default loaders read the global `get_config()` paths. The cross-encoder and its (query, text) score cache are shared.
13. **Cell boxes only on digital PDFs.**
    - P2 never kept cell boxes. Each localized cell value is searched with PyMuPDF inside the table bbox.
    - A value found once is used. A repeated value must line up with its row band (from single-hit values or the row label) and, when known, its column; otherwise it gets **no box**.
    - On image-only MP-DocVQA pages, the UI marks the table box and lists the cells. It never guesses positions.
14. **Highlights are drawn by the UI** (SVG over the PNG), not by the server. `/page-image` returns the raw page.
15. **Pre-registration.** The rule was written to `P7_PREREGISTRATION.md` at 2026-10-23 18:40 +0530, before any DEV feedback run, and is copied here unchanged:
    - Grid: α = 1; β ∈ {0.5, 0.75}; γ ∈ {0, 0.15, 0.25}; λ ∈ {0.3, 0.5}. Default β 0.75, γ 0.15, λ 0.3.
    - Primary metric: DEV residual component MRR.
    - Keep the default unless another point has MRR ≥ default + 0.01 AND R@1 ≥ default R@1. Several qualify → highest MRR; exact ties → smaller β, γ, λ.
    - References (never chosen): no feedback; λ = 0 at the default β, γ.
    - TEST once with the choice. Memory parameters are not tuned.
16. **Simulated-feedback protocol.**
    - Round 1 = P6 pipeline ranking (predicted query type, feedback off). The judged set = round-1 top 5, gold = up, others = down.
    - Round 2 = Rocchio with those 5 votes.
    - Residual collection: judged items are removed from both rankings and from the qrels. Questions with no gold left are dropped.
    - Metrics: R@1, MRR, nDCG@5, binary relevance. 95% paired bootstrap CI of (system − no feedback), 2,000 resamples, seed 42.
17. **Memory test protocol.** For every ordered pair (A, B) on the same DEV document, A's top-5 votes are recorded by **two** simulated users, because min_votes = 2 and one simulated user can give only one vote per component. B is then ranked with the memory prior vs without, on the full ranking. Related = A and B share a gold component; unrelated = they share none. No pairs were invented.
18. **New npm dependencies** (approved in the plan): react 18.3.1, react-dom 18.3.1, vite 5.4.10, @vitejs/plugin-react 4.3.3, typescript 5.6.3, @types/react 18.3.12, @types/react-dom 18.3.1. No component library and no `@types/node` (`vite.config.ts` is left out of `tsc`).
19. **UI animations have a safety net.** FLIP re-rank animations are cancelled before new ones start and are forced to their end state after 600 ms by a timer. They are skipped with `prefers-reduced-motion`, when the page is hidden, and with `?still=1` (screenshots). See fix 6.

### 3. How to run (exact commands, in order, as a teammate would on a fresh clone)
Prerequisites from P1–P6: venv and packages, data, ingest (eval + demo store), indexes, qrels, `.env` with `LLM_API_KEY`.
```powershell
# behind HTTPS-scanning antivirus only (bundle NOT in the repo): $env:SSL_CERT_FILE = "C:\path\to\ca-bundle.pem"
.venv/Scripts/python -m pytest                                                   # 160 tests

# step 0: demo questions at k = 5, slots off vs conditional (6 Groq calls)
.venv/Scripts/python -m scripts.ask "How many people had Medicaid coverage in 2022, in thousands?" --doc p60-281 --k 5 --slots off
.venv/Scripts/python -m scripts.ask "How many people had Medicaid coverage in 2022, in thousands?" --doc p60-281 --k 5 --slots conditional
#   same for "Which subtype of health insurance was the most common?" and
#   "According to Figure 2, which age group had the highest uninsured rate in 2022?"

# feedback evaluation (retrieval only; classifier labels come from the LLM cache: 0 calls)
.venv/Scripts/python -m eval.eval_feedback --split dev --grid                    # ~1.5 min
.venv/Scripts/python -m eval.eval_feedback --split test --beta 0.75 --gamma 0.15 --lam 0.3   # ~3 min, TEST once
.venv/Scripts/python -m eval.eval_feedback --memory-test                         # ~1.5 min
.venv/Scripts/python -m eval.run_eval --summary-only                             # regenerate summary.md

# API + UI (two terminals)
.venv/Scripts/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000       # Swagger: http://127.0.0.1:8000/docs
npm --prefix ui install
npm --prefix ui run dev                                                          # http://localhost:5173
```

### 4. Actual output (real terminal output from this session, trimmed)
**4.1 Step 0: demo questions at k = 5, slots off vs conditional** (`p7_evidence/demo_slots_k5.md`, 6 output files)

The saved P6 files are not a k = 5 baseline. `ask_demo_numerical_k12.txt` is k = 12, and the textual and visual P6 files list only E1–E3 (prompt est. 772 / 1,125 tokens). So both sides were run fresh in this session, with the same wording.

| question | slots | answer | citations | E5 | LLM |
|---|---|---|---|---|---|
| numerical (gold 62,050, table p8) | off | Not found | - | paragraph p22 | 1 call |
| | conditional | Not found | - | **table p6** = list of tables (front matter), "slot: best fused table" | 1 call |
| textual (gold employment-based) | off | employment-based insurance | E1 | heading p16 | 1 call |
| | conditional | employment-based insurance | E1 | same (textual: no reservation) | cache hit |
| visual (Figure 2, p9) | off | 14.0 percent | E2 | paragraph p14 | 1 call |
| | conditional | 14.0 percent | E2 | **figure p16** (Figure 6), "slot: best fused figure" | 1 call |

Conditional slots change E5, but they reserve the wrong table and the wrong figure, so no answer changes. The default is unchanged.

**4.2 DEV grid** (`p7_evidence/feedback_dev_grid.txt`; header from `feedback_dev_brief.json`)
```
DEV: 69 questions with component qrels | classifier LLM calls 0
residual questions 34 / 69 (dropped 35), with >= 1 up-vote 5 | no-fb full-ranking r@1 0.2802, mrr 0.4794, ndcg@5 0.4644
ranx check (no fb, residual): recall@1 0.0588, mrr 0.2022, ndcg@5 0.2260
| system | R@1 | MRR | nDCG@5 | round-2 ms |
| no feedback | 0.059 | 0.202 | 0.226 | - |
| lambda0_literal | 0.074 (+0.015 [+0.000, +0.044]) | 0.229 (+0.027 [-0.002, +0.083]) | 0.244 (+0.018 [+0.000, +0.054]) | 3 |
| b0.5_g0_l0.3 | 0.147 (+0.088 [+0.000, +0.176]) | 0.253 (+0.050 [-0.002, +0.111]) | 0.262 (+0.036 [-0.004, +0.082]) | 1 |
| b0.5_g0.15_l0.3 | 0.191 (+0.132 [+0.029, +0.250]) | 0.293 (+0.091 [+0.011, +0.180]) | 0.270 (+0.044 [-0.027, +0.115]) | 1 |
| b0.5_g0.25_l0.3 | 0.191 (+0.132 [+0.029, +0.250]) | 0.287 (+0.085 [+0.003, +0.174]) | 0.267 (+0.040 [-0.030, +0.114]) | 3 |
| b0.75_g0_l0.5 | 0.118 (+0.059 [-0.059, +0.176]) | 0.242 (+0.040 [-0.023, +0.105]) | 0.243 (+0.017 [-0.039, +0.071]) | 1 |
| b0.75_g0.15_l0.3 | 0.191 (+0.132 [+0.029, +0.250]) | 0.293 (+0.091 [+0.011, +0.180]) | 0.270 (+0.044 [-0.027, +0.115]) | 1 |
| b0.75_g0.15_l0.5 | 0.162 (+0.103 [+0.015, +0.206]) | 0.282 (+0.080 [+0.005, +0.168]) | 0.265 (+0.039 [-0.040, +0.115]) | 1 |
| b0.75_g0.25_l0.3 | 0.191 (+0.132 [+0.029, +0.250]) | 0.287 (+0.085 [+0.003, +0.174]) | 0.267 (+0.040 [-0.030, +0.114]) | 1 |
  ... (all 12 points in summary.md)
RULE: default kept: no grid point has MRR >= default + 0.01 with R@1 >= default
```
- β barely matters: only 5 of 34 residual questions had an up-vote in the top 5. γ > 0 helps (down-votes on 5 judged items).
- λ = 0.3 ≥ λ = 0.5 everywhere.
- b0.5_g0.15_l0.3 ties the default to 3 decimals, but the rule needs +0.01.

**4.3 TEST, once** (`p7_evidence/feedback_test.txt`)
```
TEST: 209 questions with component qrels | query types {'numerical': 95, 'textual': 104, 'visual': 10} | classifier LLM calls 0 | round 1 178s
residual questions 68 / 209 (dropped 141), with >= 1 up-vote 16 | no-fb full-ranking r@1 0.4884, mrr 0.6526, ndcg@5 0.6435
ranx check (no fb, residual): recall@1 0.1201, mrr 0.2892, ndcg@5 0.2706
| no feedback | 0.120 | 0.289 | 0.271 | - |
| lambda0_literal | 0.110 (-0.010 [-0.044, +0.015]) | 0.294 (+0.004 [-0.020, +0.027]) | 0.271 (+0.000 [-0.018, +0.018]) | 10 |
| b0.75_g0.15_l0.3 | 0.140 (+0.020 [-0.069, +0.101]) | 0.312 (+0.023 [-0.048, +0.091]) | 0.295 (+0.024 [-0.040, +0.089]) | 1 |
```
- On TEST every CI includes 0. The DEV effect (+0.091) shrank to +0.023, so I can't claim a gain.
- The literal pipeline (λ = 0) does nothing, as expected: CE decides the order of the top 30.

**4.4 P5/P6 numbers unchanged**
- Round 1 reproduces P5 `hybrid_weighted_soft_rerank`:
  - DEV component R@1 0.28019: identical.
  - DEV MRR 0.47943 vs P5's 0.47955. P5's saved DEV runs scored with the P7 metric code give exactly 0.47943 / 0.28019, so the 0.0001 gap is in how MRR is computed (ranx vs `query_metrics`), not in the ranking.
  - TEST R@1 0.488 and MRR 0.653 match P5.
- `run_eval --summary-only`, diff against summary.md before: `100a101,131`, so only lines added after the ANLS section.

**4.5 Memory test, DEV** (`p7_evidence/feedback_memory_dev.txt`)
```
related          pairs  172 fired   22 | MRR +0.0070 [0.0003, 0.0161] | R@1 +0.0116 | nDCG@5 +0.0080 | better 5 worse 0
related_fired    pairs   22 fired   22 | MRR +0.0551 [0.0021, 0.1220] | R@1 +0.0909 | nDCG@5 +0.0623 | better 5 worse 0
unrelated        pairs  426 fired   34 | MRR +0.0003 [-0.0008, 0.0015] | R@1 +0.0000 | nDCG@5 +0.0021 | better 4 worse 5
unrelated_fired  pairs   34 fired   34 | MRR +0.0033 [-0.0094, 0.0185] | R@1 +0.0000 | nDCG@5 +0.0264 | better 4 worse 5
pairs 598 on 10 docs | related passing sim+qtype gate 22 | related with the shared gold up-voted by A 109 | same-doc cosine q10/q50/q90 ['0.39', '0.51', '0.76']
```
- (i) **Related:** improves where the gates fire (22 of 172 related pairs), 5 better, 0 worse. Most related pairs don't pass the 0.75 similarity gate: the same-doc median cosine is 0.51.
- (ii) **Unrelated:** no mean change and R@1 never changes. But 5 of the 34 fired pairs got worse individually, so "must not get worse" holds on the mean, not for every pair. All 9 changed unrelated pairs move the gold below rank 1; the worst is rank 3 → 4 (MRR 0.333 → 0.250).

**4.6 Real API session** (uvicorn on 127.0.0.1:8000; `p7_evidence/api_session.md`)
```
GET /health  {"ok":true,"stores":{"eval":49,"demo":1},"votes":0}
POST /ask (2348 ms)  query_type numerical | answer "Not found in the retrieved evidence" | confidence: 0 cited, slot_used true
   E1 paragraph p8 | E2 paragraph p9 | E3 paragraph p12 | E4 paragraph p18 | E5 table p6 (slot)
POST /feedback down-vote p60-281_p6_table_0 -> recorded, answer_stale true, the p6 table gone, items tagged rocchio
GET /next x8 -> positions 5..12 (paragraphs/headings), remaining 337..330; p8 table not reached within 8
POST /feedback up-vote p60-281_p8_table_0 (rerank_ms 194.5)
   E1 table p8 tag=pinned fb +0.10 | why: numerical (table 0.6) | bm25 2.01->0.41, bge 0.85->1.00, fused 0.352, CE 0.43, fb +0.10 | pinned | terms: medicaid, coverage, 2022, thousands | Page 8 -> Section 'Number ...
   E2 paragraph p8 tag=rocchio fb -0.09 | ...
   pinned_cells: Medicaid 2 | 2022.Number | 62,050 (+ 9 more); cell_boxes: 6 located (62,050 at [721,1129,774,1149])
POST /feedback answer vote -> {"recorded": true, "target": "answer", ...}
GET /page-image/p60-281/8 -> 200 image/png 361,072 bytes (saved: p7_evidence/page_image_p60-281_p8.png)
GET /next wrong session -> 404 {"detail": "unknown query_id for this session"}
```
- Cell boxes are shown in `p7_evidence/cell_boxes_p8_crop.png`: the 2022 Number / MoE / Percent cells of Marketplace and Medicaid.
- The four repeated values ("0.1", "0.3") get no box (fix 5).

**4.7 Memory over real HTTP** (`p7_evidence/api_memory_check.txt`; the store held 23 votes from earlier sessions on this question)
```
memory=True: session 472f1314 | memory {'on': True, 'prior_items': 2} | feedback_active True
   E5 p60-281_p6_table_0           tag=memory   fb=-0.150
memory=False: session 87061042 | memory {'on': False, 'prior_items': 0} | feedback_active False
memory=True: p8 table at displayed position 8 (tag=memory, fb=+0.150)
memory=False: p8 table at displayed position 12
```
- The down-voted p6 table is still E5 even with −0.15. Slot reservation picks the best *fused* table and ignores `final` (limitation 3).

**4.8 UI** (React, real API, headless Edge 1440×1000; `p7_evidence/ui_*.png`)
- `ui_1_feedback_rerank.png`: Medicaid question after down-voting the p6 table and up-voting the p8 table.
  - Left: answer still "Not found" with the "re-ranked by your votes" banner; the pinned table at #1 (▲4, kept) with localized cells and score bars; the paragraphs show ▼1 and `rocchio`.
  - Right: page 8, the table box in vermilion and the six located cells highlighted.
- `ui_2_reanswer.png`: after "Re-answer", **62,050**, chip E1 · p.8, strip "1 cited source | answered | no slot".
- `ui_3_textual.png`: "employment-based insurance", E1 · p.9, the cited paragraph box on page 9.
- Checked by hand in the browser pane: J ×4 then ↓ on card 5 down-voted the p6 table, the list re-ranked, and the "Re-answer" button appeared.
- `npm run build`: `tsc -b && vite build` ✓ built in 813 ms (JS 158.6 kB, gzip 51.6 kB).

### 5. Tests
`160 passed, 1 warning in 65.24s` (133 from P6 + 27 new). The warning is Starlette's httpx deprecation notice from `fastapi.testclient`.
- `test_rocchio.py` (9):
  - Hand-computed Rocchio vector.
  - Empty groups and γ = 0.
  - Identity hook (final = raw CE 8.5).
  - Up-vote pinned, down-vote excluded.
  - Dense re-scored with q′ and the λ blend formula per item.
  - λ = 0 keeps CE order.
  - Why-line shows `fb` and `| rocchio`.
  - CE cache scores only new texts.
  - Memory-only = minmax(CE) + prior.
- `test_memory.py` (9):
  - Session votes: the latest vote wins, same question only.
  - Bad votes rejected.
  - SQLite file persistence.
  - The gates: min votes, cap ±0.15, similarity, doc/qtype match, 30-day half-life, current session and answer votes excluded.
- `test_api.py` (7, TestClient + fake pipeline):
  - `/documents` and `/page-image` (200/404).
  - `/ask` answer, cards, cells and session.
  - `/feedback` re-ranks with no LLM call; down-vote excluded, up-vote pinned.
  - Answer vote logged only.
  - Validation (422/404).
  - `/next` pages through the whole ranking with no new retrieval.
  - Memory across sessions on/off.
- `test_cell_boxes.py` (2): synthetic PDF.
  - Unique values, row band from a single-hit value, row band from the label, no guessed box.
  - Clipped to the table bbox.
- The P6 pipeline tests (6) still pass unchanged.

### 6. Acceptance criteria and reviewer checks
- Step 0 demo comparison, k = 5 vs k = 5, same wording: PASS. 4.1: no answer changes; the slots pick the wrong table/figure.
- Votes stored in SQLite with session_id, doc_id, query, qtype, component_id, vote, timestamp: PASS. `session_store.py` schema; 23 rows after the session.
- Answer thumbs logged only: PASS. `test_answer_vote_is_logged_only`; 4.6.
- Rocchio through the P6 hook; 👍 pinned, 👎 excluded: PASS. `test_upvote_pinned_downvote_excluded`; 4.6 E1 `pinned`, p6 table gone.
- Why-line "fb ±x | rocchio": PASS. 4.6: `..., CE 0.43, fb +0.10 | pinned | ...`, `fb -0.09 | rocchio`.
- Pre-registered rule before DEV; 12-point grid; λ = 0 and no-feedback references with paired CIs: PASS. `P7_PREREGISTRATION.md` (18:40, before the first DEV run); 4.2.
- TEST once with the chosen setting: PASS (run once). Gain not significant: +0.023 [−0.048, +0.091].
- Feedback off ⇒ final = raw CE, and the P6 tests pass: PASS. `test_no_votes_no_memory_is_identity_and_final_is_raw_ce`; 6/6 P6 tests; round 1 = P5 (4.4).
- P5/P6 numbers unchanged: PASS. summary.md diff only adds lines; round 1 reproduces P5 R@1 / MRR.
- Memory (cap, min votes, similarity, doc/qtype, half-life; off by default): PASS. `test_memory.py`; config `memory: off`.
- Memory test (i) related improves: PASS. +0.055 MRR on 22 fired pairs, 5 better, 0 worse.
- Memory test (ii) unrelated not worse: PARTIAL. Mean +0.003, R@1 unchanged, but 5 / 34 fired pairs individually worse (rank 3 → 4 at worst).
- API endpoints with TestClient: PASS. 7 tests.
- API endpoints over real HTTP: PASS. 4.6, 4.7.
- UI split view, real page, cited bbox, localized cells in place: PASS. `ui_2_reanswer.png` (cells on the Medicaid row).
- UI cards with graph-path breadcrumb and score bars (bm25 / bge / CE / fb): PASS. `ui_1_feedback_rerank.png`.
- Voting by click and keyboard; the list visibly re-ranks; "Show next"; memory toggle: PASS for keyboard, re-rank and next (4.8 browser check). The memory toggle is checked through the API only (4.7); I didn't screenshot it.
- Confidence strip (cited count, not found, slot used): PASS. `ui_1` ("0 cited sources | not found | modality slot used") vs `ui_2` ("1 cited source | answered | no slot").
- README section: PASS. `README.md` "API and UI (P7)".
- Full suite passes: PASS. 160 passed.

### 7. Interfaces for the next phase
- `FeedbackStore(path=":memory:")`:
  - `.record(session_id, doc_id, query, qtype, component_id, vote, target="evidence"|"answer", query_vec=None, timestamp=None)`
  - `.session_votes(session_id, doc_id, query) -> {cid: ±1}`
  - `.doc_votes(doc_id, qtype, exclude_session=None)`
  - `.count(target=None)`
- `rocchio_query(q, pos, neg, alpha, beta, gamma) -> unit vector`.
- `make_feedback_hook(pipe, query, votes, memory=None, params=None, scorer=None, q_vec=None) -> FeedbackHook`:
  - Pass it as `pipe.answer(..., feedback_adjust=hook)`.
  - After the call: `hook.ranked` (post-feedback Candidates), `hook.deltas`, `hook.tags`, `hook.q_prime`.
- `memory_prior(store, doc_id, qtype, q_vec, cfg, now=None, exclude_session=None) -> {cid: prior}`; `question_vector(retriever, text)`.
- `cell_boxes(pdf_path, page_no, table_bbox_px, page_size_px, cells) -> [{row, column, value, bbox}]`.
- API (JSON):
  - `POST /ask {doc_id, query, session_id?, top_k=5, memory?, generate=true}` → `{query_id, session_id, query_type, classification, answer{text, not_found, citations, confidence{cited_sources, not_found, slot_used, ...}}, evidence[card], ranking_size, votes, memory{on, prior_items}, feedback_active}`
  - `POST /feedback {session_id, query_id, vote: ±1, target, component_id?}` → same shape + `answer_stale`, `rerank_ms`, or `{recorded, target: "answer"}`
  - `GET /next?session_id&query_id` → `{item: card|null, position, remaining}`
  - `GET /documents`, `GET /page-image/{doc}/{page}`
  - card = `{id, eid, rank, type, page, branch, bbox, page_size, text, graph_path, why, scores, feedback_tag, matched_terms, localized_cells, localize_flag, cell_boxes, table_preview, reserved_slot, context}`.
- Config:
  - `feedback.{db_path, rocchio.{alpha, beta, gamma, lambda}, memory, step, cap, min_votes, min_similarity, half_life_days}`
  - `api.{cors_origins, default_store, demo_questions}`
- `eval.eval_feedback`: `query_metrics(ranking, gold)`, `choose_setting(grid)`, `summary_section(results_dir)`; results `eval/results/feedback_{dev,test}.json` (+ `_brief`), `feedback_memory_dev.json`.

### 8. Known limitations / open questions
1. **The Rocchio gain is not shown on TEST.** +0.023 MRR [−0.048, +0.091] on 68 residual questions. Only 16 of them had an up-vote in the top 5, so most of the effect comes from down-votes (γ). DEV (34 questions, 5 with an up-vote) was too small for a reliable choice.
2. **Memory on unrelated questions** moves 9 of 34 fired pairs a little; 5 get worse (never at rank 1). The 0.75 gate passes only 22 of 172 related pairs on DEV (median same-doc cosine 0.51).
3. **Slot reservation ignores feedback.** The P6 "best fused table/figure" slot uses `fused`, not `final`, so a memory penalty can't remove a reserved item (4.7). Session Rocchio is not affected, because down-votes are excluded before the slot. Making slots feedback-aware would change P6 behaviour, so I left it.
4. **Memory's effect is small by design.** ±0.15 on [0, 1] moved the gold table 12 → 8, not into the top 5.
5. **Cell boxes** exist only for digital PDFs. Repeated values without a unique row/column match get no box (4 of 10 localized cells on the demo table). The locator's extra Marketplace row ("coverage" matches both rows) is the known P6 locator limitation.
6. **`/next` is memory-only state.** Query states live in the API process (lost on restart); votes persist in SQLite.
7. **The visual demo question still fails** (Figure 2 not retrieved at k = 5, with or without slots). P7 feedback can't fix it unless the user finds Figure 2 with "Show next" (it ranks around 25) and up-votes it.
8. The 0.0001 DEV MRR gap vs the P5 JSON is a ranx-vs-local metric difference (4.4), not a ranking change.
9. Windows: the API needs the `SSL_CERT_FILE` workaround on this machine for Groq; the bundle stays outside the repo.
10. Limitation 3 left as is by decision (memory off by default; session Rocchio unaffected).

### 9. What I need from you before the next phase
- Look at the three UI screenshots: `docs/reports/p7_evidence/ui_1_feedback_rerank.png`, `ui_2_reanswer.png`, `ui_3_textual.png`, and the cell-box crop `cell_boxes_p8_crop.png`.
- Try the UI yourself (README "API and UI"). Manual checklist:
  1. Ask the Medicaid question.
  2. Press J four times, then ↓.
  3. Press N until the p8 table shows, then ↑.
  4. Re-answer.
  5. Toggle memory and ask again in a new session.
- Decide on limitation 3 (feedback-aware slot reservation) for P8, or leave it.
- Approve P7 for the commit (`phase-7: rocchio feedback, memory, api, react ui` or your message). `ui/node_modules` and `ui/dist` are gitignored; `ui/package-lock.json` will be committed. `data/processed/feedback.sqlite` (23 demo votes) is under the ignored `data/processed/`.

### 10. Fixes made during this phase
1. Step 0 first run: every Groq call failed with `APIConnectionError` (Avast TLS). I rebuilt the certifi + Windows-roots bundle (84 roots) in the session scratchpad and set `SSL_CERT_FILE` per command; the 6 runs were redone.
2. The saved P6 textual/visual outputs were not k = 5 (E1–E3 only), so the slots-off side was re-run at k = 5 rather than compared with those files.
3. `scores.feedback` was missing when the feedback delta was exactly 0 (the top item on both CE and cosine). Tagged items now always carry it (`test_why_line_shows_fb_and_rocchio` failed first).
4. **API bug, found by `test_memory_flag_carries_votes_across_sessions`:** the memory lookup ran before classification, with qtype "", so it never matched a vote. `_run` now classifies first (same LLM-failure fallback as the pipeline) and passes the type into the pipeline.
5. Cell boxes, first version: repeated values ("0.1", "0.3") were boxed on the wrong row, because the row label (with a footnote digit) and the flattened header "2022.Number" aren't found as text. Now single-hit values define the row/column bands, and an ambiguous value without a unique aligned hit gets no box. `fitz` was replaced by `import pymupdf` (deprecation).
6. **UI, found in the first screenshot:**
   - (a) The cell highlights were one row too high: the SVG overlay took the height of the scrolling container, not of the image. Image and overlay now sit in a wrapper sized by the image.
   - (b) After re-answer, cards stayed at opacity 0 or displaced: re-rank animations stacked, and in a page that isn't painting frames (the preview pane, headless Edge: `requestAnimationFrame` never fired and `document.timeline` didn't advance in 2 s) they never finished. Running animations are now cancelled first, entries start at opacity 0.35, a 600 ms timer forces every animation to its end, and animations are skipped for reduced motion, hidden pages and `?still=1`. Verified: all 5 cards in order, opacity 1, 0 animations.
7. The UI rank-change arrows used a stale `items` closure in scripted runs; it now reads a ref.
8. `tsc` failed on `vite.config.ts` (`process` needs Node types); the config is left out of the type-check instead of adding `@types/node`.
9. `tests/test_cell_boxes.py` assumed the text top was below 140 px at baseline 80 pt; it is 138.5 (top ≈ 69 pt). The test range was fixed; the code was right.

### 11. Logic explained
**Rocchio re-ranking for one (session, document, question)**
1. Read the session's votes for this question (latest per component). No votes and no memory prior → return the ranking unchanged.
2. Build the query vector:
   - q = BGE(instruction + question).
   - q′ = normalize(α·q + β·mean(BGE of up-votes) − γ·mean(BGE of down-votes)).
   - BGE component vectors come from the stored index embeddings.
3. Re-score every component of the document: dense raw = cos(q′, d); BM25 and CLIP raw are unchanged. Min-max per (branch, retriever), weighted-sum fusion with the query type's branch weights (same code as steps 3–4).
4. The cross-encoder scores the new fused top 30 with the original question. Scores are cached by (question, text), so only items new to the window are scored (round-2 median 1–10 ms in the evals, 194 ms over HTTP including the pipeline run).
5. For the top 30: final = (1−λ)·minmax(CE) + λ·minmax(cos(q′, d)). Below: −1 − i.
6. Add the memory prior if on. Up-votes: final += 10 (pinned). Down-votes: removed. `fb` = final − minmax(CE) (pin excluded).
7. The pipeline continues: slots, localization, graph context, provenance (why-line with `fb ±x | tag`). `/feedback` stops before the LLM.

**Memory prior**
- Take votes on the same document and query type from other sessions.
- Keep those whose question has cosine ≥ 0.75 to this question (plain BGE question vectors).
- Per component: needs ≥ 2 votes. prior = clip(0.05 · Σ vote·0.5^(age/30 days), −0.15, +0.15).

**Simulated feedback metric** (per question, binary)
- Residual gold G = gold − round-1 top 5. Both rankings drop the top 5.
- R@1 = |G ∩ top 1| / |G|; MRR = 1 / rank of the first gold; nDCG@5 = DCG@5 / IDCG@5 with ideal min(|G|, 5).
- CI = percentile bootstrap of the per-question differences. ranx agrees on the no-feedback residual.

### 12. Checklist (P7 prompt items 0–8)
| item | done? | evidence |
|---|---|---|
| 0 Demo questions k = 5 with slots conditional vs off; default unchanged | yes | 4.1; `p7_evidence/demo_slots_k5.md` |
| 1 Thumbs per evidence item in SQLite (session, doc, query, qtype, component, vote, timestamp) | yes | `session_store.py`; `test_memory.py` |
| 1 Optional answer thumbs, logged only | yes | `test_answer_vote_is_logged_only`; UI 👍👎 |
| 2 Rocchio q′ via the P6 hook; re-score dense, re-fuse, re-rank | yes | `rocchio.py`; `test_rocchio_rescoring_and_lambda_blend` |
| 2 👍 pinned, 👎 excluded | yes | `test_upvote_pinned_downvote_excluded`; 4.6 |
| 2 Defaults α 1, β 0.75, γ 0.15 (+ λ 0.3, your change); pre-registered rule before DEV | yes | config; `P7_PREREGISTRATION.md` |
| 2 Why-line "fb ±x \| rocchio" | yes | 4.6 |
| 3 Memory: one function + one table; cap ±0.15; doc + qtype + cosine ≥ 0.75; ≥ 2 votes; half-life 30 d; off by default | yes | `memory.py`; `test_memory.py`; 4.7 |
| 4 Simulated feedback on DEV, residual R@1 / MRR / nDCG@5 with paired CIs | yes | 4.2 |
| 4 Memory test (related improves, unrelated not worse); pair counts reported | yes (ii partial) | 4.5 |
| 4 Rule applied, TEST once | yes | 4.2 RULE line; 4.3 |
| (your change) λ grid, λ = 0 and no-feedback references with paired CIs | yes | 4.2, 4.3 |
| (your change) Feedback off ⇒ final = raw CE; P6 tests pass | yes | 5; 4.4 |
| 5 API: /ask, /feedback, /next, /documents (for /docs), /page-image; models loaded once; TestClient tests | yes | `api/main.py`; `test_api.py`; 4.6 |
| 6 React (Vite + TS) split view, page image with bbox and cells in place | yes | `ui_1`, `ui_2` |
| 6 Cards: breadcrumb + score bars; click/keyboard voting; visible re-rank; Show next; memory toggle | yes | 4.8; `EvidenceCard.tsx`, `App.tsx` |
| 6 Confidence strip; paper-and-ink identity with one accent per modality | yes | `ui_1`/`ui_2`/`ui_3`; `styles.css` |
| 6 README section | yes | `README.md` |
| 7 Tests: Rocchio math, pin/exclude, memory gating, API; full suite passes | yes | 160 passed |
| 8 Report with 2–3 UI screenshots; no commit | yes | this file; 3 screenshots; nothing committed |
