"""Answer quality (ANLS) on the TEST split with the P5 system (hybrid_weighted_soft_rerank) + P6 answering.

Settings:
  a       top-1 evidence only (no graph context, no localized cells; a table is shown as its first rows)
  b       top-3 evidence + graph context (+ localized cells for numerical queries)
  oracle  all components of the gold answer page (reading order) as evidence, on a fixed random sample of
          --oracle-n TEST questions (seed eval.seed): separates retrieval errors from LLM errors

Prediction for ANLS = the validated answer; "not found" (incl. answers without a valid citation) counts as "".
ANLS (eval.metrics.anls_single): per question max over gold answers of 1 - NL if NL < 0.5 else 0,
NL = Levenshtein(pred, gold) / max(len(pred), len(gold)), lowercased and whitespace-normalized.

Resumable: every finished question is appended to eval/results/answers_{setting}.jsonl and skipped next time;
LLM replies are also in the LLM disk cache. --max-minutes stops starting new questions after the budget;
--recompute ignores the jsonl and reruns everything (served from the LLM cache: shows 0 new calls).
--estimate builds every prompt without calling the API and prints the calls / tokens still needed.

Usage:
  .venv/Scripts/python -m eval.eval_answers --estimate
  .venv/Scripts/python -m eval.eval_answers --settings a b --limit 10
  .venv/Scripts/python -m eval.eval_answers --settings a b --max-minutes 15      # repeat until done
  .venv/Scripts/python -m eval.eval_answers --settings oracle --max-minutes 15
  .venv/Scripts/python -m eval.eval_answers --report                            # metrics + summary.md only
  .venv/Scripts/python -m eval.eval_answers --split dev --settings b --slots conditional   # slot decision (DEV)
  .venv/Scripts/python -m eval.eval_answers --split dev --compare-slots off conditional both
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from docscout.config import get_config
from docscout.schema import Component, Evidence, iter_jsonl
from eval.metrics import anls_single
from eval.split import load_split

SETTINGS = {"a": dict(top_k=1, use_context=False, use_localization=False),
            "b": dict(top_k=3, use_context=True, use_localization=True)}
ALL_SETTINGS = ["a", "b", "oracle"]
LABEL = {"a": "(a) top-1 evidence only", "b": "(b) top-3 + graph context + localized cells",
         "oracle": "(c) oracle: gold answer page as evidence"}


class DryClient:
    """Stands in for the LLM in --estimate: cached prompts are answered from the cache, others are recorded."""

    def __init__(self, real: Any) -> None:
        from docscout.reasoning.llm_client import LLMStats

        self.real, self.model, self.stats, self.misses = real, real.model, LLMStats(), []

    def complete_json(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        from docscout.reasoning.llm_client import cache_key

        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        hit = self.real.cache.get(cache_key(self.real.provider.name, self.real.model, msgs, self.real.temperature))
        if hit is not None:
            self.stats.cache_hits += 1
            return json.loads(hit["text"])
        self.misses.append(sum(len(m["content"]) for m in msgs))
        return {"answer": "", "citations": []}


def results_path(cfg: dict, setting: str, split: str = "test", slots: str = "off") -> Path:
    """answers_{setting}.jsonl for the main TEST run (slots off); answers_{split}_{setting}_slots-{slots}.jsonl else."""
    if split == "test" and slots == "off":
        return cfg["paths"]["results_dir"] / f"answers_{setting}.jsonl"
    return cfg["paths"]["results_dir"] / f"answers_{split}_{setting}_slots-{slots}.jsonl"


def load_done(cfg: dict, setting: str, split: str = "test", slots: str = "off") -> Dict[str, dict]:
    p = results_path(cfg, setting, split, slots)
    return {r["qid"]: r for r in iter_jsonl(p)} if p.exists() else {}


def oracle_sample(qids: List[str], n: int, seed: int) -> List[str]:
    return sorted(random.Random(seed).sample(sorted(qids), min(n, len(qids))), key=qids.index)


def oracle_evidence(q: dict, comps_by_page: Dict[tuple, List[Component]], pipe: Any, qtype: str) -> List[Evidence]:
    """Every component of the gold answer page, in reading order (tables: localized cells for numerical)."""
    from docscout.localization.table_locator import clean_table, df_markdown, locate

    out = []
    for c in sorted(comps_by_page.get((q["doc_id"], q["answer_page"]), []), key=lambda x: x.order):
        e = Evidence(component=c)
        if c.type == "table":
            df = pipe.table(c)
            if df is not None:
                if qtype == "numerical":
                    _, cells, flag = locate(q["question"], df)
                    e.localized_cells, e.meta["localize_flag"] = [tuple(x) for x in cells], flag
                e.meta["table_preview"] = df_markdown(clean_table(df), int(pipe.cfg["answer"]["max_table_rows"]))
        out.append(e)
    return out


def record(qid: str, q: dict, qtype: str, gold_type: str, setting: str, answer: str, not_found: bool, meta: dict,
           relevant_page_in_evidence: Optional[bool]) -> dict:
    pred = "" if not_found else answer
    return {"qid": qid, "setting": setting, "question": q["question"], "doc_id": q["doc_id"], "pred_type": qtype,
            "gold_type": gold_type, "answers": q["answers"], "answer": answer, "not_found": not_found,
            "anls": anls_single(pred, q["answers"]), **meta, "answer_page_in_evidence": relevant_page_in_evidence}


BOOTSTRAP_N = 2000


def bootstrap_ci(values: List[float], n: int = BOOTSTRAP_N, seed: int = 42, alpha: float = 0.05) -> List[float]:
    """Percentile bootstrap CI of the mean: resample questions with replacement n times."""
    if not values:
        return [0.0, 0.0]
    import numpy as np

    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def summarize(rows: List[dict]) -> Dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    by_type: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_type[r["gold_type"]].append(r["anls"])
    return {
        "n": n, "anls": statistics.mean(r["anls"] for r in rows),
        "anls_ci95": bootstrap_ci([r["anls"] for r in rows]),
        "anls_by_gold_type": {t: {"n": len(v), "anls": statistics.mean(v), "ci95": bootstrap_ci(v)}
                              for t, v in sorted(by_type.items())},
        "not_found_rate": sum(r["not_found"] for r in rows) / n,
        "not_found": sum(r["not_found"] for r in rows),
        "no_valid_citation": sum(r["no_valid_citation"] for r in rows),
        "questions_with_invalid_citations": sum(1 for r in rows if r["invalid_citations"]),
        "invalid_citations_total": sum(len(r["invalid_citations"]) for r in rows),
        "llm_errors": sum(1 for r in rows if r.get("error")),
        "answer_page_in_evidence": sum(1 for r in rows if r.get("answer_page_in_evidence")),
        "prompt_tokens_est_mean": statistics.mean(r["prompt_est_tokens"] for r in rows),
    }


def summary_section(results_dir: Path) -> List[str]:
    """Markdown for summary.md (called by eval.run_eval.write_summary too)."""
    f = results_dir / "answers.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text(encoding="utf-8"))
    lines = ["", "## TEST: answer quality (ANLS, P6)", "",
             f"System: hybrid_weighted_soft_rerank + grounded answering with `{d.get('model')}` "
             "(temperature 0, JSON output, citations validated). \"Not found\" and answers without a valid citation "
             "score 0. Gold types: DocVQA question_types mapping (P5).", "",
             "ANLS with 95% percentile bootstrap CI over questions "
             f"({BOOTSTRAP_N} resamples, seed 42); per-type n in brackets.", "",
             "| setting | n | ANLS [95% CI] | numerical | textual | visual | not found | no valid citation | invalid citation ids |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in ALL_SETTINGS:
        m = d["settings"].get(s)
        if not m or not m.get("n"):
            continue
        bt = m["anls_by_gold_type"]
        cell = lambda t: (f"{bt[t]['anls']:.3f} [{bt[t]['ci95'][0]:.3f}, {bt[t]['ci95'][1]:.3f}] ({bt[t]['n']})"  # noqa: E731
                          if t in bt else "-")
        ci = m["anls_ci95"]
        lines.append(f"| {LABEL[s]} | {m['n']} | **{m['anls']:.3f}** [{ci[0]:.3f}, {ci[1]:.3f}] | {cell('numerical')} | {cell('textual')} | "
                     f"{cell('visual')} | {m['not_found_rate']:.1%} ({m['not_found']}) | {m['no_valid_citation']} | "
                     f"{m['invalid_citations_total']} in {m['questions_with_invalid_citations']} q |")
    if d.get("a_vs_b"):
        x = d["a_vs_b"]
        ci = x["b_minus_a_ci95"]
        per = "; ".join(f"{t} {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] (n={v['n']})"
                        for t, v in x.get("b_minus_a_by_gold_type", {}).items())
        lines += ["", f"Paired (b) - (a) on the same {x['n']} questions: {x['mean_b_minus_a']:+.3f} "
                  f"[{ci[0]:+.3f}, {ci[1]:+.3f}]; {per}. (b) better on {x['b_better']}, (a) better on {x['a_better']}."]
    if d.get("oracle_compare"):
        o = d["oracle_compare"]
        lines += ["", f"Same {o['n']} oracle-sample questions: (a) {o['a']:.3f}, (b) {o['b']:.3f}, "
                  f"oracle {o['oracle']:.3f}."]
    return lines


def report(cfg: dict, qids_test: List[str], oracle_qids: List[str], model: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"model": model, "settings": {}}
    rows_by = {}
    for s in ALL_SETTINGS:
        done = load_done(cfg, s)
        want = oracle_qids if s == "oracle" else qids_test
        rows = [done[q] for q in want if q in done]
        rows_by[s] = {r["qid"]: r for r in rows}
        out["settings"][s] = {**summarize(rows), "expected": len(want)}
    if rows_by["oracle"]:
        common = [q for q in oracle_qids if all(q in rows_by[s] for s in ALL_SETTINGS)]
        if common:
            out["oracle_compare"] = {"n": len(common), **{s: statistics.mean(rows_by[s][q]["anls"] for q in common)
                                                         for s in ALL_SETTINGS}}
    if rows_by["a"] and rows_by["b"]:
        common = [q for q in qids_test if q in rows_by["a"] and q in rows_by["b"]]
        diffs = [rows_by["b"][q]["anls"] - rows_by["a"][q]["anls"] for q in common]
        out["a_vs_b"] = {"n": len(common), "mean_b_minus_a": statistics.mean(diffs) if diffs else 0.0,
                         "b_minus_a_ci95": bootstrap_ci(diffs),
                         "b_minus_a_by_gold_type": {
                             t: {"n": len(d), "mean": statistics.mean(d), "ci95": bootstrap_ci(d)}
                             for t in ("numerical", "textual", "visual")
                             for d in [[rows_by["b"][q]["anls"] - rows_by["a"][q]["anls"] for q in common
                                        if rows_by["a"][q]["gold_type"] == t]] if d},
                         "b_better": sum(rows_by["b"][q]["anls"] > rows_by["a"][q]["anls"] for q in common),
                         "a_better": sum(rows_by["a"][q]["anls"] > rows_by["b"][q]["anls"] for q in common),
                         "a_better_qids": [q for q in common if rows_by["a"][q]["anls"] > rows_by["b"][q]["anls"]]}
    res_dir = cfg["paths"]["results_dir"]
    (res_dir / "answers.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def slot_report(cfg: dict, split: str, qids: List[str], variants: List[str]) -> Dict[str, Any]:
    """Setting (b) per slot variant on one split: ANLS + CI, by type, and paired bootstrap of variant - off."""
    rows = {v: load_done(cfg, "b", split, v) for v in variants}
    out: Dict[str, Any] = {"split": split, "setting": "b", "variants": {}}
    base = rows.get("off", {})
    for v in variants:
        done = [rows[v][q] for q in qids if q in rows[v]]
        m = {**summarize(done), "expected": len(qids)}
        if v != "off" and base:
            common = [q for q in qids if q in rows[v] and q in base]
            diffs = [rows[v][q]["anls"] - base[q]["anls"] for q in common]
            m["minus_off"] = {"n": len(common), "mean": statistics.mean(diffs) if diffs else 0.0,
                              "ci95": bootstrap_ci(diffs), "better": sum(d > 0 for d in diffs),
                              "worse": sum(d < 0 for d in diffs),
                              "evidence_changed": sum(rows[v][q]["evidence_ids"] != base[q]["evidence_ids"]
                                                      for q in common)}
        out["variants"][v] = m
    (cfg["paths"]["results_dir"] / f"answers_slots_{split}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for v, m in out["variants"].items():
        if not m.get("n"):
            print(f"[{split}] slots={v}: no answers yet")
            continue
        bt = ", ".join(f"{t} {x['anls']:.3f} (n={x['n']})" for t, x in m["anls_by_gold_type"].items())
        line = (f"[{split}] (b) slots={v:<11} n={m['n']}/{m['expected']} ANLS={m['anls']:.3f} "
                f"[{m['anls_ci95'][0]:.3f},{m['anls_ci95'][1]:.3f}] | {bt} | not found {m['not_found_rate']:.1%}")
        if "minus_off" in m:
            d = m["minus_off"]
            line += (f" | minus off {d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] better {d['better']} "
                     f"worse {d['worse']} evidence changed {d['evidence_changed']}/{d['n']}")
        print(line)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="+", choices=ALL_SETTINGS, default=["a", "b"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N TEST questions")
    ap.add_argument("--max-minutes", type=float, default=0, help="no new question after this many minutes (0 = none)")
    ap.add_argument("--oracle-n", type=int, default=40)
    ap.add_argument("--estimate", action="store_true", help="count the calls / tokens still needed; no API calls")
    ap.add_argument("--recompute", action="store_true", help="ignore saved answers (LLM cache still used)")
    ap.add_argument("--report", action="store_true", help="only compute metrics from saved answers")
    ap.add_argument("--split", choices=["test", "dev"], default="test",
                    help="dev: only for the slot-reservation decision (settings a / b; no oracle)")
    ap.add_argument("--slots", choices=["off", "conditional", "both"], default="off",
                    help="modality slot reservation variant passed to the pipeline (retrieval/slots.py)")
    ap.add_argument("--compare-slots", nargs="+", choices=["off", "conditional", "both"], default=None,
                    help="only print the (b) comparison of these slot variants on --split from saved answers")
    ap.add_argument("--completion-tokens", type=int, default=0,
                    help="completion tokens per call for --estimate (default: mean of the cached answer calls, else 250)")
    args = ap.parse_args(argv)

    res_dir: Path = cfg["paths"]["results_dir"]
    split = load_split(cfg)
    if args.split == "dev" and "oracle" in args.settings:
        print("the oracle setting is TEST only")
        return 1
    test_ids = set(split[args.split]["question_ids"])
    questions = {q["question_id"]: q for q in iter_jsonl(cfg["paths"]["questions_file"]) if q["question_id"] in test_ids}
    qrels_page = json.loads((res_dir / "qrels_page.json").read_text(encoding="utf-8"))
    gold = json.loads((res_dir / "qtypes_gold.json").read_text(encoding="utf-8"))["gold"]
    qids = [q for q in questions if q in qrels_page]
    oracle_qids = oracle_sample(qids, args.oracle_n, int(cfg["eval"]["seed"]))
    if args.limit:
        qids = qids[: args.limit]
        oracle_qids = [q for q in oracle_qids if q in set(qids)] if "oracle" not in args.settings else oracle_qids[: args.limit]

    from docscout.reasoning.llm_client import get_client

    client = get_client()
    if args.compare_slots:
        slot_report(cfg, args.split, qids, args.compare_slots)
        return 0
    if args.report:
        out = report(cfg, [q for q in questions if q in qrels_page], oracle_sample(
            [q for q in questions if q in qrels_page], args.oracle_n, int(cfg["eval"]["seed"])), client.model)
        _print_report(out)
        _write_summary(cfg)
        return 0

    from docscout.pipeline import DocScoutPipeline
    from docscout.query.classifier import classify
    from docscout.reasoning.answerer import answer_question

    comps = {r["id"]: Component.from_dict(r) for r in iter_jsonl(cfg["paths"]["components_file"])}
    by_page: Dict[tuple, List[Component]] = defaultdict(list)
    for c in comps.values():
        by_page[(c.doc_id, c.page)].append(c)
    llm = DryClient(client) if args.estimate else client
    pipe = DocScoutPipeline(cfg=cfg, components=comps, client=llm)
    pipe.warm_up()
    qtypes = {qid: classify(questions[qid]["question"], client=client)["type"] for qid in qids + oracle_qids}
    print(f"{args.split.upper()} questions: {len(qids)} | slots {args.slots} | oracle sample: {len(oracle_qids)} | settings {args.settings} | "
          f"query types {dict(Counter(qtypes[q] for q in qids))} | model {client.model}"
          + (" | ESTIMATE (no API calls)" if args.estimate else ""))
    stats0 = dict(client.stats.as_dict())
    t_start = time.time()
    stopped = False
    consecutive_errors = 0
    for s in args.settings:
        done = {} if (args.recompute or args.estimate) else load_done(cfg, s, args.split, args.slots)
        todo = [q for q in (oracle_qids if s == "oracle" else qids) if q not in done]
        out_f = results_path(cfg, s, args.split, args.slots)
        if args.recompute and out_f.exists() and not args.estimate:
            out_f.unlink()
        new = 0
        for qid in tqdm(todo, desc=f"setting {s}", leave=False):
            if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
                stopped = True
                break
            q = questions[qid]
            page_key = f"{q['doc_id']}_p{q['answer_page']}"
            if s == "oracle":
                ev = oracle_evidence(q, by_page, pipe, qtypes[qid])
                ans = answer_question(q["question"], qtypes[qid], ev, llm, cfg)
                meta = {"citations": ans.citations, "invalid_citations": ans.invalid_citations,
                        "no_valid_citation": ans.no_valid_citation, "error": ans.error,
                        "raw_answer": ans.raw_answer, "prompt_est_tokens": ans.prompt_info.get("est_tokens", 0),
                        "dropped_items": ans.prompt_info.get("dropped_items", 0), "evidence_ids": [e.component.id for e in ev]}
                rec = record(qid, q, qtypes[qid], gold.get(qid, "textual"), s, ans.answer, ans.not_found, meta, True)
            else:
                res = pipe.answer(q["question"], q["doc_id"], qtype=qtypes[qid], slots=args.slots, **SETTINGS[s])
                v = res.meta["answer_validation"]
                meta = {"citations": res.meta["citations"], "invalid_citations": v["invalid_citations"],
                        "no_valid_citation": v["no_valid_citation"], "error": v["error"], "raw_answer": v["raw_answer"],
                        "prompt_est_tokens": res.meta["prompt"].get("est_tokens", 0),
                        "evidence_ids": [e.component.id for e in res.evidence],
                        "context_ids": [x for e in res.evidence for x in e.meta.get("context_ids", [])],
                        "localization": res.meta["localization"], "reserved": res.meta["slots"]["reserved"]}
                in_ev = any(f"{e.component.doc_id}_p{e.component.page}" == page_key for e in res.evidence)
                rec = record(qid, q, qtypes[qid], gold.get(qid, "textual"), s, res.answer, res.meta["not_found"], meta, in_ev)
            if args.estimate:
                continue
            if rec["error"]:  # not saved: retried on the next run
                consecutive_errors += 1
                tqdm.write(f"LLM error on {qid} ({s}): {rec['error']}")
                if consecutive_errors >= 3:
                    print("3 LLM errors in a row (rate limit or daily quota?): stopping; rerun later to resume.")
                    stopped = True
                    break
                continue
            consecutive_errors = 0
            with out_f.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            new += 1
        print(f"setting {s}: {new} new answers, {len(done) + new}/{len(oracle_qids if s == 'oracle' else qids)} done")
        if stopped:
            break

    if args.estimate:
        comp_tok = args.completion_tokens or _mean_completion_tokens(cfg) or 250
        n = len(llm.misses)
        ptok = sum(m // 4 for m in llm.misses)
        total = ptok + n * comp_tok
        tpm = int(cfg["llm"].get("tokens_per_minute", 0) or 8000)
        print(f"ESTIMATE: {n} LLM calls still needed ({llm.stats.cache_hits} prompts already cached) | prompt tokens ~{ptok:,} "
              f"(mean {ptok // max(n, 1)}) + completion ~{comp_tok}/call -> ~{total:,} tokens | at {tpm:,} tokens/min "
              f"~{total / tpm:.0f} min (and >= {n * float(cfg['llm']['min_interval_s']) / 60:.0f} min from the call gap) "
              f"| Groq daily request limit 1,000")
        return 0
    d = {k: client.stats.as_dict()[k] - stats0.get(k, 0) for k in ("calls", "cache_hits", "retries", "prompt_tokens",
                                                                   "completion_tokens", "total_tokens")}
    print(f"this run: LLM calls={d['calls']} cache_hits={d['cache_hits']} retries={d['retries']} tokens={d['total_tokens']} "
          f"(prompt {d['prompt_tokens']}, completion {d['completion_tokens']}) | wall {time.time() - t_start:.0f}s"
          + (" | STOPPED at the time budget: rerun the same command to resume" if stopped else ""))
    if args.split != "test" or args.slots != "off":
        slot_report(cfg, args.split, [q for q in questions if q in qrels_page],
                    [v for v in ("off", "conditional", "both") if results_path(cfg, "b", args.split, v).exists()])
        return 0
    all_q = [q for q in questions if q in qrels_page]
    out = report(cfg, all_q, oracle_sample(all_q, args.oracle_n, int(cfg["eval"]["seed"])), client.model)
    _print_report(out)
    _write_summary(cfg)
    return 0


def _mean_completion_tokens(cfg: dict) -> int:
    """Mean completion tokens of cached answer calls (identified by the answer prompt's first line)."""
    p = cfg["llm"]["cache_path"]
    vals = []
    if Path(p).exists():
        for r in iter_jsonl(Path(p)):
            if r.get("json_mode") and r.get("usage") and '"citations"' in r.get("text", ""):
                vals.append(r["usage"].get("completion_tokens", 0))
    return int(statistics.mean(vals)) if vals else 0


def _print_report(out: Dict[str, Any]) -> None:
    for s, m in out["settings"].items():
        if not m.get("n"):
            print(f"{LABEL[s]}: no answers yet (0/{m['expected']})")
            continue
        bt = ", ".join(f"{t} {v['anls']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}] (n={v['n']})"
                       for t, v in m["anls_by_gold_type"].items())
        ci = m["anls_ci95"]
        print(f"{LABEL[s]}: n={m['n']}/{m['expected']} ANLS={m['anls']:.3f} [{ci[0]:.3f},{ci[1]:.3f}] | {bt} | not found {m['not_found_rate']:.1%} "
              f"({m['not_found']}, of which no valid citation {m['no_valid_citation']}) | invalid citation ids "
              f"{m['invalid_citations_total']} in {m['questions_with_invalid_citations']} questions | answer page in "
              f"evidence {m['answer_page_in_evidence']}/{m['n']} | prompt est. mean {m['prompt_tokens_est_mean']:.0f} tokens")
    if out.get("oracle_compare"):
        o = out["oracle_compare"]
        print(f"oracle sample (n={o['n']}): a {o['a']:.3f} | b {o['b']:.3f} | oracle {o['oracle']:.3f}")
    if out.get("a_vs_b"):
        x = out["a_vs_b"]
        ci = x["b_minus_a_ci95"]
        print(f"a vs b per question (n={x['n']}): b-a {x['mean_b_minus_a']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] | "
              f"b better {x['b_better']}, a better {x['a_better']}")


def _write_summary(cfg: dict) -> None:
    from eval.run_eval import main as run_eval_main

    run_eval_main(["--summary-only"])


if __name__ == "__main__":
    sys.exit(main())
