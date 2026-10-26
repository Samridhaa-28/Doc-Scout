# P7 pre-registered rule for the Rocchio setting

Written 2026-10-23 18:40 +0530, before any DEV feedback run (`eval/results/` had no feedback results at that time).
It is copied unchanged into P7_REPORT.md section 2.

**Grid (12 points):** alpha = 1.0; beta ∈ {0.5, 0.75}; gamma ∈ {0, 0.15, 0.25}; lambda ∈ {0.3, 0.5}.
**Default:** beta = 0.75, gamma = 0.15, lambda = 0.3.
**References (reported, never chosen):** no feedback; lambda = 0 (literal pipeline: Rocchio re-scores dense, re-fuses
and re-runs the CE, final = minmax(CE)) at beta = 0.75, gamma = 0.15.

**Protocol (DEV only, retrieval only, no LLM):**
1. Round 1 = the P6 pipeline ranking (hybrid weighted soft + rerank, predicted query type, feedback off).
2. The simulated user votes on the round-1 top-5 from the component qrels: gold = up, every other item = down.
3. Round 2 = the same question with session Rocchio (the 5 votes).
4. Residual collection: the 5 judged items are removed from both rankings and from the qrels; questions with no gold
   component left are dropped. No-feedback = round-1 ranking on the same residual.
5. Component-level metrics: R@1, MRR, nDCG@5 (binary); 95% paired bootstrap CIs of (system - no feedback),
   2,000 resamples over questions, seed 42.

**Selection rule:** primary metric = DEV residual MRR. Keep the default unless another grid point has
DEV residual MRR ≥ default MRR + 0.01 AND DEV residual R@1 ≥ default R@1. If several qualify, take the highest MRR;
exact ties → smaller beta, then smaller gamma, then smaller lambda.

**TEST:** run once with the chosen setting (plus the two references), report the same metrics and CIs. No change after
seeing TEST. Memory parameters are not tuned (cap 0.15, min_votes 2, similarity 0.75, half-life 30 days).
