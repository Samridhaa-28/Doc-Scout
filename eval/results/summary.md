# Retrieval results (within-document search)

Dev/test split by document (`eval/results/split.json`). Weights were tuned on dev only. **Main tables: TEST** (39 documents, 217 questions; 176 from documents with >= 2 pages; 209 with >= 1 relevant component). Headline metrics: R@1 and MRR (bold). R@5 / R@10 / nDCG@10 are secondary: documents are short, so even random pages reach R@5 = 0.77.

Latency: per query, models loaded, query classifier excluded (see classifier.json for its latency).

## TEST: page level, documents with >= 2 pages (main)

| system | R@1 | MRR | R@5 | R@10 | nDCG@10 | median ms | p90 ms |
|---|---|---|---|---|---|---|---|
| random (expected) | 0.270 | 0.501 | 0.801 | 0.966 | - | - | - |
| bm25_all | **0.636** | **0.718** | 0.841 | 0.852 | 0.751 | 0.2 | 0.5 |
| dense_minilm_all | **0.568** | **0.725** | 0.932 | 0.983 | 0.788 | 8.9 | 11.8 |
| dense_bge_all | **0.631** | **0.756** | 0.926 | 0.983 | 0.810 | 35.6 | 40.4 |
| hybrid_weighted_soft | **0.659** | **0.788** | 0.966 | 0.994 | 0.839 | 54.3 | 60.2 |
| hybrid_rrf_soft | **0.659** | **0.786** | 0.955 | 0.994 | 0.837 | 54.8 | 60.5 |
| hybrid_weighted_hard | **0.602** | **0.696** | 0.835 | 0.875 | 0.740 | 36.1 | 43.3 |
| hybrid_weighted_soft_rerank | **0.733** | **0.828** | 0.949 | 0.994 | 0.869 | 762.7 | 1850.9 |
| hybrid_weighted_soft_goldtype (analysis: gold types) | **0.670** | **0.795** | 0.966 | 0.994 | 0.844 | 57.2 | 66.4 |

## TEST: component level (questions with >= 1 relevant component)

| system | R@1 | MRR | R@5 | R@10 | nDCG@10 |
|---|---|---|---|---|---|
| bm25_all | **0.417** | **0.552** | 0.612 | 0.648 | 0.556 |
| dense_minilm_all | **0.259** | **0.465** | 0.621 | 0.764 | 0.524 |
| dense_bge_all | **0.362** | **0.544** | 0.670 | 0.755 | 0.582 |
| hybrid_weighted_soft | **0.432** | **0.599** | 0.701 | 0.763 | 0.624 |
| hybrid_rrf_soft | **0.398** | **0.581** | 0.698 | 0.793 | 0.612 |
| hybrid_weighted_hard | **0.360** | **0.473** | 0.491 | 0.519 | 0.458 |
| hybrid_weighted_soft_rerank | **0.488** | **0.653** | 0.716 | 0.805 | 0.677 |
| hybrid_weighted_soft_goldtype (analysis: gold types) | **0.449** | **0.607** | 0.694 | 0.757 | 0.628 |

## TEST: page level, all questions

| system | R@1 | MRR | R@5 | R@10 | nDCG@10 |
|---|---|---|---|---|---|
| random (expected) | 0.408 | 0.596 | 0.839 | 0.972 | - |
| bm25_all | **0.664** | **0.730** | 0.829 | 0.839 | 0.757 |
| dense_minilm_all | **0.650** | **0.777** | 0.945 | 0.986 | 0.828 |
| dense_bge_all | **0.700** | **0.802** | 0.940 | 0.986 | 0.846 |
| hybrid_weighted_soft | **0.724** | **0.828** | 0.972 | 0.995 | 0.869 |
| hybrid_rrf_soft | **0.724** | **0.826** | 0.963 | 0.995 | 0.868 |
| hybrid_weighted_hard | **0.585** | **0.661** | 0.774 | 0.806 | 0.697 |
| hybrid_weighted_soft_rerank | **0.783** | **0.861** | 0.959 | 0.995 | 0.893 |
| hybrid_weighted_soft_goldtype (analysis: gold types) | **0.733** | **0.834** | 0.972 | 0.995 | 0.874 |

## TEST: paired t-test vs dense_bge_all (ranx, p < 0.05)

| system | metric | system score | baseline score | p-value | significant |
|---|---|---|---|---|---|
| hybrid_weighted_soft | page MRR (>=2 pages) | 0.788 | 0.756 | 0.0765 | no |
| hybrid_weighted_soft_rerank | page MRR (>=2 pages) | 0.828 | 0.756 | 0.0002 | yes |
| hybrid_weighted_soft | component MRR | 0.599 | 0.544 | 0.0032 | yes |
| hybrid_weighted_soft_rerank | component MRR | 0.653 | 0.544 | 0.0000 | yes |

## Secondary: all 293 questions (dev + test; dev was used for tuning)

Page level, documents with >= 2 pages (246 questions):

| system | R@1 | MRR | R@5 | R@10 | nDCG@10 |
|---|---|---|---|---|---|
| random (expected) | 0.246 | 0.473 | 0.770 | 0.938 | - |
| bm25_all | **0.593** | **0.679** | 0.793 | 0.833 | 0.716 |
| dense_minilm_all | **0.537** | **0.696** | 0.902 | 0.963 | 0.760 |
| dense_bge_all | **0.606** | **0.738** | 0.907 | 0.976 | 0.795 |
| hybrid_weighted_soft | **0.585** | **0.730** | 0.919 | 0.984 | 0.791 |
| hybrid_rrf_soft | **0.573** | **0.716** | 0.894 | 0.976 | 0.778 |
| hybrid_weighted_hard | **0.524** | **0.630** | 0.772 | 0.829 | 0.677 |
| hybrid_weighted_soft_rerank | **0.683** | **0.796** | 0.923 | 0.980 | 0.840 |
| hybrid_weighted_soft_goldtype (analysis: gold types) | **0.638** | **0.765** | 0.931 | 0.980 | 0.817 |

Component level (278 questions):

| system | R@1 | MRR | R@5 | R@10 | nDCG@10 |
|---|---|---|---|---|---|
| bm25_all | **0.380** | **0.522** | 0.588 | 0.633 | 0.530 |
| dense_minilm_all | **0.244** | **0.448** | 0.593 | 0.729 | 0.502 |
| dense_bge_all | **0.331** | **0.514** | 0.633 | 0.726 | 0.551 |
| hybrid_weighted_soft | **0.382** | **0.560** | 0.653 | 0.735 | 0.583 |
| hybrid_rrf_soft | **0.350** | **0.540** | 0.641 | 0.767 | 0.570 |
| hybrid_weighted_hard | **0.313** | **0.428** | 0.432 | 0.470 | 0.408 |
| hybrid_weighted_soft_rerank | **0.437** | **0.610** | 0.673 | 0.780 | 0.637 |
| hybrid_weighted_soft_goldtype (analysis: gold types) | **0.432** | **0.597** | 0.666 | 0.747 | 0.613 |

## TEST: answer quality (ANLS, P6)

System: hybrid_weighted_soft_rerank + grounded answering with `openai/gpt-oss-20b` (temperature 0, JSON output, citations validated). "Not found" and answers without a valid citation score 0. Gold types: DocVQA question_types mapping (P5).

ANLS with 95% percentile bootstrap CI over questions (2000 resamples, seed 42); per-type n in brackets.

| setting | n | ANLS [95% CI] | numerical | textual | visual | not found | no valid citation | invalid citation ids |
|---|---|---|---|---|---|---|---|---|
| (a) top-1 evidence only | 217 | **0.486** [0.424, 0.546] | 0.515 [0.419, 0.609] (106) | 0.466 [0.375, 0.561] (103) | 0.344 [0.094, 0.688] (8) | 38.2% (83) | 0 | 0 in 0 q |
| (b) top-3 + graph context + localized cells | 217 | **0.646** [0.585, 0.707] | 0.674 [0.585, 0.764] (106) | 0.632 [0.543, 0.725] (103) | 0.457 [0.125, 0.787] (8) | 15.7% (34) | 0 | 0 in 0 q |
| (c) oracle: gold answer page as evidence | 40 | **0.791** [0.666, 0.910] | 0.778 [0.604, 0.913] (23) | 0.798 [0.596, 0.985] (16) | 1.000 [1.000, 1.000] (1) | 15.0% (6) | 0 | 0 in 0 q |

Paired (b) - (a) on the same 217 questions: +0.161 [+0.103, +0.218]; numerical +0.159 [+0.073, +0.247] (n=106); textual +0.166 [+0.086, +0.250] (n=103); visual +0.114 [+0.000, +0.341] (n=8). (b) better on 47, (a) better on 11.

Same 40 oracle-sample questions: (a) 0.545, (b) 0.740, oracle 0.791.

## TEST: relevance feedback (P7, simulated, residual collection, component level)

68 questions with residual gold (of 209); top-5 judged from qrels. Cells: score (diff vs no feedback [95% paired bootstrap CI]).

| system | R@1 | MRR | nDCG@5 |
|---|---|---|---|
| no feedback | 0.120 (+0.000 [+0.000, +0.000]) | 0.289 (+0.000 [+0.000, +0.000]) | 0.271 (+0.000 [+0.000, +0.000]) |
| lambda0_literal | 0.110 (-0.010 [-0.044, +0.015]) | 0.294 (+0.004 [-0.020, +0.027]) | 0.271 (+0.000 [-0.018, +0.018]) |
| b0.75_g0.15_l0.3 (chosen) | 0.140 (+0.020 [-0.069, +0.101]) | 0.312 (+0.023 [-0.048, +0.091]) | 0.295 (+0.024 [-0.040, +0.089]) |

## DEV: relevance feedback (P7, simulated, residual collection, component level)

34 questions with residual gold (of 69); top-5 judged from qrels. Cells: score (diff vs no feedback [95% paired bootstrap CI]).

| system | R@1 | MRR | nDCG@5 |
|---|---|---|---|
| no feedback | 0.059 (+0.000 [+0.000, +0.000]) | 0.202 (+0.000 [+0.000, +0.000]) | 0.226 (+0.000 [+0.000, +0.000]) |
| lambda0_literal | 0.074 (+0.015 [+0.000, +0.044]) | 0.229 (+0.027 [-0.002, +0.083]) | 0.244 (+0.018 [+0.000, +0.054]) |
| b0.5_g0_l0.3 | 0.147 (+0.088 [+0.000, +0.176]) | 0.253 (+0.050 [-0.002, +0.111]) | 0.262 (+0.036 [-0.004, +0.082]) |
| b0.5_g0_l0.5 | 0.118 (+0.059 [-0.059, +0.176]) | 0.242 (+0.040 [-0.023, +0.105]) | 0.243 (+0.017 [-0.039, +0.071]) |
| b0.5_g0.15_l0.3 | 0.191 (+0.132 [+0.029, +0.250]) | 0.293 (+0.091 [+0.011, +0.180]) | 0.270 (+0.044 [-0.027, +0.115]) |
| b0.5_g0.15_l0.5 | 0.162 (+0.103 [+0.015, +0.206]) | 0.282 (+0.080 [+0.005, +0.168]) | 0.265 (+0.039 [-0.040, +0.115]) |
| b0.5_g0.25_l0.3 | 0.191 (+0.132 [+0.029, +0.250]) | 0.287 (+0.085 [+0.003, +0.174]) | 0.267 (+0.040 [-0.030, +0.114]) |
| b0.5_g0.25_l0.5 | 0.191 (+0.132 [+0.029, +0.250]) | 0.285 (+0.082 [-0.002, +0.178]) | 0.258 (+0.032 [-0.055, +0.118]) |
| b0.75_g0_l0.3 | 0.147 (+0.088 [+0.000, +0.176]) | 0.252 (+0.050 [-0.002, +0.111]) | 0.262 (+0.036 [-0.004, +0.082]) |
| b0.75_g0_l0.5 | 0.118 (+0.059 [-0.059, +0.176]) | 0.242 (+0.040 [-0.023, +0.105]) | 0.243 (+0.017 [-0.039, +0.071]) |
| b0.75_g0.15_l0.3 (chosen) | 0.191 (+0.132 [+0.029, +0.250]) | 0.293 (+0.091 [+0.011, +0.180]) | 0.270 (+0.044 [-0.027, +0.115]) |
| b0.75_g0.15_l0.5 | 0.162 (+0.103 [+0.015, +0.206]) | 0.282 (+0.080 [+0.005, +0.168]) | 0.265 (+0.039 [-0.040, +0.115]) |
| b0.75_g0.25_l0.3 | 0.191 (+0.132 [+0.029, +0.250]) | 0.287 (+0.085 [+0.003, +0.174]) | 0.267 (+0.040 [-0.030, +0.114]) |
| b0.75_g0.25_l0.5 | 0.191 (+0.132 [+0.029, +0.250]) | 0.284 (+0.082 [-0.002, +0.178]) | 0.258 (+0.032 [-0.055, +0.118]) |
