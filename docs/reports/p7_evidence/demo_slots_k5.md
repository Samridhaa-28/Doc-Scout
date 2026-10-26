# P7 step 0: demo questions at k = 5, slots off vs conditional (deferred from P6)

Command (same wording as the P6 files, doc p60-281, demo store, gpt-oss-20b, temperature 0):
`.venv/Scripts/python -m scripts.ask "<question>" --doc p60-281 --k 5 --slots {off|conditional}`
(`--slots` was added to ask.py in P7; without it ask.py uses `answer.slots` from the config, unchanged = conditional.)

Both sides are fresh k = 5 runs from this session. The saved P6 files are not a k = 5 baseline:
`ask_demo_numerical_k12.txt` is k = 12, and `ask_demo_textual.txt` / `ask_demo_visual_figure2.txt` list only E1-E3
(prompt est. 772 / 1,125 tokens), so the slots-off k = 5 prompts were cache misses for all three questions.

| question | slots | answer | citations | E5 (only position that differs) | LLM |
|---|---|---|---|---|---|
| numerical: Medicaid 2022, thousands (gold 62,050, table p8) | off | Not found | - | paragraph p22 | 1 call |
| | conditional | Not found | - | **table p6** (list of tables, front matter), reserved "best fused table"; locator row = "Table A-1..." title | 1 call |
| textual: most common subtype (gold employment-based) | off | employment-based insurance | E1 | heading p16 | 1 call |
| | conditional | employment-based insurance | E1 | same (textual: no reservation) | cache hit |
| visual: Figure 2, highest uninsured age group (gold figure p9) | off | 14.0 percent | E2 | paragraph p14 | 1 call |
| | conditional | 14.0 percent | E2 | **figure p16** (Figure 6, uninsured by income-to-poverty), reserved "best fused figure" | 1 call |

Result: conditional slots change E5 for the numerical and visual questions, but the reserved item is the wrong
table / figure in both cases (best fused, not the gold one), so no answer changes. The P6 default
(`answer.slots: conditional`) is unchanged. Full outputs: `demo_{numerical,textual,visual}_k5_{off,conditional}.txt`.

Note: the first attempt of these six runs failed with APIConnectionError (Avast TLS interception); they were rerun with
`SSL_CERT_FILE` pointing to a certifi + Windows-roots bundle in the session scratchpad (not in the repo).
