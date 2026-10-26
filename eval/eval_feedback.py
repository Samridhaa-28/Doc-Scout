"""P7 relevance-feedback evaluation (retrieval only, no LLM answer calls).

Simulated feedback (--split dev|test):
  round 1 = P6 pipeline ranking (feedback off); the simulated user votes on the top-5 with the component qrels
  (gold = up, others = down); round 2 = session Rocchio. Metrics on the residual collection (judged items removed
  from both rankings and the qrels; questions without residual gold dropped): R@1, MRR, nDCG@5 (component level),
  95% paired bootstrap CIs of (system - no feedback). The rule that picks the setting is pre-registered in
  docs/reports/P7_PREREGISTRATION.md and implemented in choose_setting().

Memory test (--memory-test, DEV): for every ordered pair (A, B) of questions on the same document, votes from A's
top-5 are recorded by two simulated users (min_votes = 2), then B is ranked with the memory prior vs without.
Related pairs share a gold component; unrelated pairs share none. Full ranking of B (B itself gave no votes).

  python -m eval.eval_feedback --split dev --grid          # 12 grid points + references, applies the rule
  python -m eval.eval_feedback --split test --beta 0.75 --gamma 0.15 --lam 0.3   # TEST once
  python -m eval.eval_feedback --memory-test               # DEV
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from docscout.config import get_config
from docscout.schema import Component, iter_jsonl
from eval.eval_answers import bootstrap_ci
from eval.split import load_split

GRID = {"beta": [0.5, 0.75], "gamma": [0.0, 0.15, 0.25], "lambda": [0.3, 0.5]}
DEFAULT = {"alpha": 1.0, "beta": 0.75, "gamma": 0.15, "lambda": 0.3}
JUDGED_K = 5
METRICS = ["r@1", "mrr", "ndcg@5"]


# ---------------------------------------------------------------- metrics (binary relevance)
def query_metrics(ranking: Sequence[str], gold: set) -> Dict[str, float]:
    """R@1 = |gold in top-1| / |gold|; MRR; nDCG@5 with ideal DCG over min(|gold|, 5) items."""
    if not gold:
        raise ValueError("empty gold set")
    r1 = len(gold & set(ranking[:1])) / len(gold)
    rr = next((1.0 / (i + 1) for i, c in enumerate(ranking) if c in gold), 0.0)
    dcg = sum(1.0 / math.log2(i + 2) for i, c in enumerate(ranking[:5]) if c in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 5)))
    return {"r@1": r1, "mrr": rr, "ndcg@5": dcg / idcg}


def simulated_votes(ranking: Sequence[str], gold: set, k: int = JUDGED_K) -> Dict[str, int]:
    return {c: (1 if c in gold else -1) for c in ranking[:k]}


def residual(ranking: Sequence[str], judged: set) -> List[str]:
    return [c for c in ranking if c not in judged]


def compare(per_q: Dict[str, Dict[str, float]], base: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Means and paired bootstrap CIs of (system - base) over the same questions."""
    qids = sorted(per_q)
    out: Dict[str, Any] = {"n": len(qids)}
    for m in METRICS:
        diff = [per_q[q][m] - base[q][m] for q in qids]
        out[m] = statistics.fmean(per_q[q][m] for q in qids) if qids else 0.0
        out[f"{m}_diff"] = statistics.fmean(diff) if diff else 0.0
        out[f"{m}_ci"] = bootstrap_ci(diff)
    return out


def choose_setting(grid: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    """The pre-registered rule (P7_PREREGISTRATION.md). Returns (chosen name, reason)."""
    d = grid[setting_name(DEFAULT)]
    ok = [(n, r) for n, r in grid.items()
          if n != setting_name(DEFAULT) and r["mrr"] >= d["mrr"] + 0.01 and r["r@1"] >= d["r@1"]]
    if not ok:
        return setting_name(DEFAULT), "default kept: no grid point has MRR >= default + 0.01 with R@1 >= default"
    ok.sort(key=lambda x: (-x[1]["mrr"], x[1]["params"]["beta"], x[1]["params"]["gamma"], x[1]["params"]["lambda"]))
    return ok[0][0], f"{ok[0][0]}: MRR {ok[0][1]['mrr']:.4f} vs default {d['mrr']:.4f}, R@1 {ok[0][1]['r@1']:.4f} >= {d['r@1']:.4f}"


def setting_name(p: Dict[str, float]) -> str:
    return f"b{p['beta']:g}_g{p['gamma']:g}_l{p['lambda']:g}"


# ---------------------------------------------------------------- data + round 1
class Bench:
    """Loads the pipeline once and caches round-1 candidate lists, query vectors and CE scores per question."""

    def __init__(self, cfg: dict, split: str) -> None:
        from docscout.feedback.rocchio import CachedScorer, component_vector_lookup
        from docscout.pipeline import DocScoutPipeline
        from docscout.query.classifier import classify
        from docscout.reasoning.llm_client import get_client

        self.cfg = cfg
        res_dir: Path = cfg["paths"]["results_dir"]
        ids = set(load_split(cfg)[split]["question_ids"])
        qrels = json.loads((res_dir / "qrels_component.json").read_text(encoding="utf-8"))
        self.questions = {q["question_id"]: q for q in iter_jsonl(cfg["paths"]["questions_file"])
                          if q["question_id"] in ids and qrels.get(q["question_id"])}
        self.gold = {q: set(qrels[q]) for q in self.questions}
        client = get_client()
        stats0 = dict(client.stats.as_dict())
        self.qtypes = {q: classify(v["question"], client=client)["type"] for q, v in self.questions.items()}
        self.classifier_calls = client.stats.as_dict()["calls"] - stats0.get("calls", 0)
        comps = {r["id"]: Component.from_dict(r) for r in iter_jsonl(cfg["paths"]["components_file"])}
        self.pipe = DocScoutPipeline(cfg=cfg, components=comps, client=client)
        self.pipe.warm_up()
        self.vec = component_vector_lookup(self.pipe.retriever)
        self._scorer_cls = CachedScorer
        self.round1: Dict[str, list] = {}
        self.qvec: Dict[str, np.ndarray] = {}
        self.scorers: Dict[str, Any] = {}

    def run_round1(self) -> None:
        from docscout.feedback.rocchio import FeedbackHook

        for qid, q in tqdm(self.questions.items(), desc="round 1", leave=False):
            cap = FeedbackHook(q["question"], {}, None, self.vec, self.pipe.texts, self.cfg)
            self.pipe.answer(q["question"], q["doc_id"], top_k=JUDGED_K, generate=False, use_context=False,
                             use_localization=False, slots="off", qtype=self.qtypes[qid], feedback_adjust=cap)
            self.round1[qid] = cap.ranked
            self.qvec[qid] = self.pipe.retriever._encode("dense_bge", q["question"])
            sc = self._scorer_cls(self.pipe.reranker.score)
            sc.seed(q["question"], cap.ranked, self.pipe.texts)
            self.scorers[qid] = sc

    def ids1(self, qid: str) -> List[str]:
        return [c.id for c in self.round1[qid]]

    def round2(self, qid: str, votes: Dict[str, int], params: Dict[str, float],
               memory: Optional[Dict[str, float]] = None) -> Tuple[List[str], float]:
        from docscout.feedback.rocchio import FeedbackHook

        q = self.questions[qid]
        h = FeedbackHook(q["question"], votes, self.qvec[qid], self.vec, self.pipe.texts, self.cfg,
                         scorer=self.scorers[qid], memory=memory, params=params)
        t0 = time.perf_counter()
        out = h(list(self.round1[qid]), self.qtypes[qid], None)
        return [c.id for c in out], (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------- simulated feedback
def feedback_eval(b: Bench, settings: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    judged = {q: set(b.ids1(q)[:JUDGED_K]) for q in b.questions}
    res_gold = {q: b.gold[q] - judged[q] for q in b.questions}
    qids = [q for q in b.questions if res_gold[q]]
    votes = {q: simulated_votes(b.ids1(q), b.gold[q]) for q in qids}
    base = {q: query_metrics(residual(b.ids1(q), judged[q]), res_gold[q]) for q in qids}
    full_nofb = {q: query_metrics(b.ids1(q), b.gold[q]) for q in b.questions}
    out: Dict[str, Any] = {
        "questions": len(b.questions), "residual_questions": len(qids),
        "dropped_no_residual_gold": len(b.questions) - len(qids),
        "with_up_vote": sum(1 for q in qids if any(v > 0 for v in votes[q].values())),
        "nofb_full_ranking": {m: statistics.fmean(full_nofb[q][m] for q in full_nofb) for m in METRICS},
        "classifier_llm_calls": b.classifier_calls,
        "no_feedback": {**compare(base, base), "per_q": base},
        "settings": {},
    }
    for name, p in settings.items():
        per_q, ms = {}, []
        for q in tqdm(qids, desc=name, leave=False):
            ranking, dt = b.round2(q, votes[q], p)
            per_q[q] = query_metrics(residual(ranking, judged[q]), res_gold[q])
            ms.append(dt)
        out["settings"][name] = {"params": p, **compare(per_q, base), "round2_ms_median": statistics.median(ms),
                                 "per_q": per_q}
    out["ce_cache_entries"] = sum(len(s.cache) for s in b.scorers.values())
    return out


def ranx_check(b: Bench) -> Dict[str, float]:
    """No-feedback residual metrics recomputed with ranx (sanity check of query_metrics)."""
    from ranx import Qrels, Run, evaluate

    qrels, run = {}, {}
    for q in b.questions:
        judged = set(b.ids1(q)[:JUDGED_K])
        g = b.gold[q] - judged
        if not g:
            continue
        r = residual(b.ids1(q), judged)
        qrels[q] = {c: 1 for c in g}
        run[q] = {c: float(len(r) - i) for i, c in enumerate(r)}
    return {k: float(v) for k, v in evaluate(Qrels(qrels), Run(run), ["recall@1", "mrr", "ndcg@5"]).items()}


# ---------------------------------------------------------------- memory test
def memory_eval(b: Bench) -> Dict[str, Any]:
    from docscout.feedback.memory import memory_prior, question_vector
    from docscout.feedback.session_store import FeedbackStore

    plain = {q: question_vector(b.pipe.retriever, b.questions[q]["question"]) for q in b.questions}
    by_doc: Dict[str, List[str]] = defaultdict(list)
    for q in b.questions:
        by_doc[b.questions[q]["doc_id"]].append(q)
    now = time.time()
    rows = []
    for doc, qs in by_doc.items():
        for a, bq in itertools.permutations(qs, 2):
            store = FeedbackStore()
            for user in ("sim-user-1", "sim-user-2"):
                for cid, v in simulated_votes(b.ids1(a), b.gold[a]).items():
                    store.record(f"{user}:{a}", doc, b.questions[a]["question"], b.qtypes[a], cid, v,
                                 query_vec=plain[a], timestamp=now)
            prior = memory_prior(store, doc, b.qtypes[bq], plain[bq], b.cfg, now=now)
            store.close()
            before = query_metrics(b.ids1(bq), b.gold[bq])
            after = query_metrics(b.round2(bq, {}, dict(DEFAULT), memory=prior)[0], b.gold[bq]) if prior else before
            rows.append({
                "a": a, "b": bq, "doc": doc, "related": bool(b.gold[a] & b.gold[bq]),
                "cos": float(plain[a] @ plain[bq]), "same_qtype": b.qtypes[a] == b.qtypes[bq],
                "prior_items": len(prior), "fired": bool(prior),
                "a_upvoted_shared_gold": bool(b.gold[a] & b.gold[bq] & set(b.ids1(a)[:JUDGED_K])),
                "before": before, "after": after,
            })

    def group(sel: List[dict]) -> Dict[str, Any]:
        g: Dict[str, Any] = {"pairs": len(sel), "fired": sum(r["fired"] for r in sel)}
        for m in METRICS:
            d = [r["after"][m] - r["before"][m] for r in sel]
            g[f"{m}_diff"] = statistics.fmean(d) if d else 0.0
            g[f"{m}_ci"] = bootstrap_ci(d)
        g["better_mrr"] = sum(r["after"]["mrr"] > r["before"]["mrr"] + 1e-12 for r in sel)
        g["worse_mrr"] = sum(r["after"]["mrr"] < r["before"]["mrr"] - 1e-12 for r in sel)
        return g

    rel = [r for r in rows if r["related"]]
    unrel = [r for r in rows if not r["related"]]
    return {
        "pairs": len(rows), "docs": len(by_doc),
        "related": group(rel), "related_fired": group([r for r in rel if r["fired"]]),
        "related_gate_ok": sum(1 for r in rel if r["cos"] >= float(b.cfg["feedback"]["min_similarity"]) and r["same_qtype"]),
        "related_with_shared_gold_upvoted": sum(r["a_upvoted_shared_gold"] for r in rel),
        "unrelated": group(unrel), "unrelated_fired": group([r for r in unrel if r["fired"]]),
        "cos_quantiles_same_doc": [float(x) for x in np.quantile([r["cos"] for r in rows], [0.1, 0.5, 0.9])] if rows else [],
        "rows": rows,
    }


# ---------------------------------------------------------------- summary.md section
def _fmt_row(name: str, r: Dict[str, Any]) -> str:
    cells = []
    for m in METRICS:
        lo, hi = r[f"{m}_ci"]
        cells.append(f"{r[m]:.3f} ({r[f'{m}_diff']:+.3f} [{lo:+.3f}, {hi:+.3f}])")
    return f"| {name} | " + " | ".join(cells) + " |"


def summary_section(results_dir: Path) -> List[str]:
    lines: List[str] = []
    for split in ("test", "dev"):
        f = results_dir / f"feedback_{split}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        lines += ["", f"## {split.upper()}: relevance feedback (P7, simulated, residual collection, component level)", "",
                  f"{d['residual_questions']} questions with residual gold (of {d['questions']}); top-{JUDGED_K} judged "
                  f"from qrels. Cells: score (diff vs no feedback [95% paired bootstrap CI]).", "",
                  "| system | R@1 | MRR | nDCG@5 |", "|---|---|---|---|",
                  _fmt_row("no feedback", d["no_feedback"])]
        for name, r in d["settings"].items():
            tag = " (chosen)" if name == d.get("chosen") else ""
            lines.append(_fmt_row(name + tag, r))
    return lines


def _strip(d: Any) -> Any:
    """Copy without the per-question tables (the brief file)."""
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if k not in ("per_q", "rows")}
    return d


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--grid", action="store_true", help="DEV: the 12 pre-registered grid points + rule")
    ap.add_argument("--beta", type=float, default=DEFAULT["beta"])
    ap.add_argument("--gamma", type=float, default=DEFAULT["gamma"])
    ap.add_argument("--lam", type=float, default=DEFAULT["lambda"])
    ap.add_argument("--memory-test", action="store_true", help="DEV memory test (A's votes -> question B)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke run; not saved)")
    args = ap.parse_args(argv)
    cfg = get_config()
    res_dir: Path = cfg["paths"]["results_dir"]
    if args.grid and args.split != "dev":
        print("the grid is DEV only")
        return 1
    if args.memory_test and args.split != "dev":
        print("the memory test is DEV only")
        return 1

    t0 = time.time()
    b = Bench(cfg, args.split)
    if args.limit:
        b.questions = dict(list(b.questions.items())[: args.limit])
    b.run_round1()
    print(f"{args.split.upper()}: {len(b.questions)} questions with component qrels | query types "
          f"{dict(Counter(b.qtypes[q] for q in b.questions))} | classifier LLM calls {b.classifier_calls} | "
          f"round 1 {time.time() - t0:.0f}s")

    if args.memory_test:
        out = memory_eval(b)
        for k in ("related", "related_fired", "unrelated", "unrelated_fired"):
            g = out[k]
            print(f"{k:16s} pairs {g['pairs']:4d} fired {g['fired']:4d} | MRR {g['mrr_diff']:+.4f} {g['mrr_ci']} | "
                  f"R@1 {g['r@1_diff']:+.4f} | nDCG@5 {g['ndcg@5_diff']:+.4f} | better {g['better_mrr']} worse {g['worse_mrr']}")
        print(f"pairs {out['pairs']} on {out['docs']} docs | related passing sim+qtype gate {out['related_gate_ok']} | "
              f"related with the shared gold up-voted by A {out['related_with_shared_gold_upvoted']} | "
              f"same-doc cosine q10/q50/q90 {['%.2f' % x for x in out['cos_quantiles_same_doc']]}")
        if not args.limit:
            (res_dir / "feedback_memory_dev.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        return 0

    lam0 = {**DEFAULT, "lambda": 0.0}
    if args.grid:
        grid = {setting_name(p): p for p in ({**DEFAULT, "beta": bb, "gamma": g, "lambda": la}
                                             for bb in GRID["beta"] for g in GRID["gamma"] for la in GRID["lambda"])}
    else:
        p = {"alpha": 1.0, "beta": args.beta, "gamma": args.gamma, "lambda": args.lam}
        grid = {setting_name(p): p}
    settings = {"lambda0_literal": lam0, **grid}
    out = feedback_eval(b, settings)
    out["ranx_check_nofb"] = ranx_check(b)
    if args.grid:
        chosen, why = choose_setting({k: v for k, v in out["settings"].items() if k != "lambda0_literal"})
        out["chosen"], out["rule"] = chosen, why
    else:
        out["chosen"] = next(iter(grid))
    print(f"residual questions {out['residual_questions']} / {out['questions']} (dropped {out['dropped_no_residual_gold']}), "
          f"with >= 1 up-vote {out['with_up_vote']} | no-fb full-ranking {_r(out['nofb_full_ranking'])}")
    print(f"ranx check (no fb, residual): {_r(out['ranx_check_nofb'])}")
    print("| system | R@1 | MRR | nDCG@5 | round-2 ms |")
    print(_fmt_row("no feedback", out["no_feedback"]) + " - |")
    for name, r in out["settings"].items():
        print(_fmt_row(name, r) + f" {r['round2_ms_median']:.0f} |")
    if args.grid:
        print("RULE:", out["rule"])
    print(f"total {time.time() - t0:.0f}s")
    if not args.limit:
        (res_dir / f"feedback_{args.split}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("wrote", res_dir / f"feedback_{args.split}.json")
        (res_dir / f"feedback_{args.split}_brief.json").write_text(json.dumps(_strip(out), indent=1), encoding="utf-8")
    return 0


def _r(d: Dict[str, float]) -> str:
    return ", ".join(f"{k} {v:.4f}" for k, v in d.items())


if __name__ == "__main__":
    sys.exit(main())
