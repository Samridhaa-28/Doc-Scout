"""Evaluate the query-type classifier (rules-only, llm-only, combined) on the 293 evaluation questions.

Ground truth: DocVQA `question_types` joined on questionId (lmms-lab/DocVQA, config DocVQA, split validation;
text columns only, streamed, no images). Mapped to our labels with priority visual > numerical > textual:
  figure/diagram or Image/Photo                      -> visual
  table/list, or a gold answer that is a number/date -> numerical
  everything else                                    -> textual
If the join covers < 80% of questions, PROXY labels are used instead (see proxy_label) and marked PROXY.

Outputs: eval/results/qtypes_gold.json (cached join + labels) and eval/results/classifier.json.

Usage: .venv/Scripts/python -m eval.eval_classifier [--limit N] [--refresh-gold]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from docscout.config import get_config
from docscout.query.classifier import LABELS, classify, llm_classify, rule_classify, rule_matches
from docscout.schema import iter_jsonl
from eval.build_qrels import is_numeric_answer, load_eval_docs

GOLD_DATASET = ("lmms-lab/DocVQA", "DocVQA", "validation")
VISUAL_TYPES = {"figure/diagram", "Image/Photo"}
TABLE_TYPES = {"table/list"}
_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE_RE = re.compile(rf"^\s*(?:\d{{1,2}}\s+{_MONTHS},?\s+\d{{2,4}}|{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{2,4}}"
                      rf"|{_MONTHS},?\s+\d{{4}})\s*$", re.IGNORECASE)
_PROXY_VISUAL_RE = re.compile(r"\b(?:charts?|graphs?|figures?|images?|logos?|photos?)\b", re.IGNORECASE)


def answer_is_numeric(answer: str) -> bool:
    """Number, amount, percentage or date ('0.28', '$1,234', '12/03/1998', 'March 3, 1998')."""
    return is_numeric_answer(answer) or bool(_DATE_RE.match(str(answer)))


def map_types(types: Sequence[str], answers: Sequence[str]) -> str:
    """DocVQA question_types + gold answers -> textual / numerical / visual (priority visual > numerical)."""
    ts = set(types)
    if ts & VISUAL_TYPES:
        return "visual"
    if ts & TABLE_TYPES or any(answer_is_numeric(a) for a in answers):
        return "numerical"
    return "textual"


def proxy_label(question: str, answers: Sequence[str]) -> str:
    """PROXY label (used only if the join coverage is < 80%)."""
    if any(answer_is_numeric(a) for a in answers):
        return "numerical"
    if _PROXY_VISUAL_RE.search(question):
        return "visual"
    return "textual"


def fetch_gold_types(qids: Sequence[str], cache_file: Path, refresh: bool = False) -> Dict[str, List[str]]:
    """{questionId: question_types} for our qids, streamed from DocVQA (text columns only), cached on disk."""
    if cache_file.exists() and not refresh:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if set(qids) <= set(cached.get("requested", [])):
            return cached["types"]
    from datasets import load_dataset
    from tqdm import tqdm

    name, config, split = GOLD_DATASET
    ds = load_dataset(name, config, split=split, streaming=True, columns=["questionId", "question_types"])
    want, found = set(qids), {}
    for row in tqdm(ds, desc=f"streaming {name} {split} (text columns)", unit="q"):
        qid = str(row["questionId"])
        if qid in want:
            found[qid] = list(row["question_types"] or [])
            if len(found) == len(want):
                break
    cache_file.write_text(json.dumps({"source": "/".join(GOLD_DATASET), "requested": sorted(want), "types": found}),
                          encoding="utf-8")
    return found


def scores(gold: List[str], pred: List[str]) -> Dict[str, object]:
    """Accuracy, macro-F1 and confusion matrix (rows = gold, cols = predicted, order LABELS)."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    return {"accuracy": float(accuracy_score(gold, pred)),
            "macro_f1": float(f1_score(gold, pred, labels=list(LABELS), average="macro", zero_division=0)),
            "confusion": confusion_matrix(gold, pred, labels=list(LABELS)).tolist()}


def print_confusion(title: str, s: Dict[str, object]) -> None:
    print(f"\n{title}: accuracy={s['accuracy']:.3f} macro-F1={s['macro_f1']:.3f}")
    head = "gold/pred"
    print(f"  {head:<12}" + "".join(f"{l:>10}" for l in LABELS))
    for lab, row in zip(LABELS, s["confusion"]):  # type: ignore[arg-type]
        print(f"  {lab:<12}" + "".join(f"{v:>10}" for v in row))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--refresh-gold", action="store_true", help="re-stream the DocVQA question types")
    args = ap.parse_args(argv)

    from docscout.reasoning.llm_client import LLMError, get_client

    cfg = get_config()
    results_dir: Path = cfg["paths"]["results_dir"]
    docs = set(load_eval_docs(cfg))
    qrels_page = json.loads((results_dir / "qrels_page.json").read_text(encoding="utf-8"))
    questions = [q for q in iter_jsonl(cfg["paths"]["questions_file"]) if q["doc_id"] in docs and q["question_id"] in qrels_page]
    if args.limit:
        questions = questions[: args.limit]
    qids = [q["question_id"] for q in questions]

    # ---- ground truth
    t0 = time.perf_counter()
    types = fetch_gold_types(qids, results_dir / "docvqa_qtypes_cache.json", args.refresh_gold)
    coverage = sum(1 for q in qids if q in types) / len(qids)
    proxy = coverage < 0.8
    print(f"gold join: {sum(1 for q in qids if q in types)}/{len(qids)} questions found in "
          f"{'/'.join(GOLD_DATASET)} -> coverage {coverage:.1%} ({time.perf_counter() - t0:.1f}s)")
    gold: Dict[str, str] = {}
    mapping_rows: Counter = Counter()
    for q in questions:
        if proxy:
            gold[q["question_id"]] = proxy_label(q["question"], q["answers"])
        else:
            ts = types.get(q["question_id"], [])
            lab = map_types(ts, q["answers"])
            gold[q["question_id"]] = lab
            num = any(answer_is_numeric(a) for a in q["answers"])
            mapping_rows[(" + ".join(sorted(ts)) or "(none)", "numeric" if num else "non-numeric", lab)] += 1
    tag = "PROXY" if proxy else "DocVQA question_types"
    if proxy:
        print("coverage < 80%: using PROXY labels (rules-vs-proxy accuracy is partly circular)")
    else:
        print("\nmapping table (DocVQA question_types, gold answer kind -> our label): count")
        for (ts, kind, lab), n in sorted(mapping_rows.items(), key=lambda kv: -kv[1]):
            print(f"  {ts:<40} {kind:<12} -> {lab:<10} {n}")
    lc = Counter(gold.values())
    print(f"label counts ({tag}): " + ", ".join(f"{l}={lc.get(l, 0)}" for l in LABELS))

    # ---- predictions
    try:
        client = get_client()
    except LLMError as e:
        print(f"LLM unavailable: {e}")
        return 3
    t_all = time.perf_counter()
    rules_pred, comb_pred, llm_pred, sources = {}, {}, {}, Counter()
    comb_lat, comb_lat_uncached, comb_lat_cached = [], [], []
    try:
        for q in questions:
            qid, text = q["question_id"], q["question"]
            r = rule_classify(text)
            rules_pred[qid] = r["type"] if r else cfg["classifier"].get("unsure_fallback", "textual")
            calls_before = client.stats.calls
            t = time.perf_counter()
            c = classify(text, client=client)
            dt = (time.perf_counter() - t) * 1000
            comb_lat.append(dt)
            if c["source"] == "llm":
                (comb_lat_uncached if client.stats.calls > calls_before else comb_lat_cached).append(dt)
            comb_pred[qid] = c["type"]
            sources[c["source"]] += 1
        for q in questions:
            llm_pred[q["question_id"]] = llm_classify(q["question"], client)["type"]
    except LLMError as e:
        print(f"\nSTOP: LLM calls keep failing after retries: {e}\n{client.stats}")
        return 3
    wall = time.perf_counter() - t_all

    g = [gold[q] for q in qids]
    res = {"labels": tag, "coverage": coverage, "n": len(qids), "label_counts": dict(lc),
           "rules_only": scores(g, [rules_pred[q] for q in qids]),
           "llm_only": scores(g, [llm_pred[q] for q in qids]),
           "combined": scores(g, [comb_pred[q] for q in qids]),
           "combined_sources": dict(sources), "llm_stats": client.stats.as_dict(), "wall_s": wall,
           "model": f"{client.provider.name}/{client.model}"}
    print(f"\nlabels: {tag} | n = {len(qids)} | combined sources: {dict(sources)} "
          f"(rules unsure -> LLM on {sources.get('llm', 0)})")
    print_confusion("rules-only (unsure -> textual)", res["rules_only"])
    print_confusion("llm-only", res["llm_only"])
    print_confusion("combined (rules, LLM when unsure)", res["combined"])
    print(f"\n{client.stats} | wall time {wall:.1f}s | model {res['model']}")

    out = results_dir / "classifier.json"
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    lat = {"combined_median_ms": statistics.median(comb_lat),
           "rules_decided_median_ms": statistics.median([d for d, q in zip(comb_lat, questions)
                                                         if rule_classify(q["question"]) is not None] or [0.0]),
           "llm_cached_median_ms": statistics.median(comb_lat_cached) if comb_lat_cached else None,
           "llm_uncached_median_ms": statistics.median(comb_lat_uncached) if comb_lat_uncached else None}
    # keep the uncached latency measured on the first (uncached) run when this run was fully cached
    if lat["llm_uncached_median_ms"] is None:
        lat["llm_uncached_median_ms"] = prev.get("latency", {}).get("llm_uncached_median_ms")
    res["latency"] = lat
    print("classifier latency (ms/query, median): " + ", ".join(
        f"{k}={'-' if v is None else f'{v:.2f}'}" for k, v in lat.items()))
    res["predictions"] = {q: {"rules": rules_pred[q], "llm": llm_pred[q], "combined": comb_pred[q], "gold": gold[q]}
                          for q in qids}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    (results_dir / "qtypes_gold.json").write_text(json.dumps({"labels": tag, "gold": gold}, indent=1), encoding="utf-8")
    examples = [(q, rule_matches(q["question"])) for q in questions if comb_pred[q["question_id"]] != gold[q["question_id"]]][:5]
    print("\nsample combined errors:")
    for q, m in examples:
        print(f"  {q['question_id']}: {q['question'][:80]!r} gold={gold[q['question_id']]} "
              f"pred={comb_pred[q['question_id']]} rules={m} answers={q['answers'][:2]} types={types.get(q['question_id'])}")
    print(f"wrote {out.relative_to(cfg['repo_root'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
