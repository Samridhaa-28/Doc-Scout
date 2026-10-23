"""Evaluate retrieval systems at page and component level (within the question's document).

A system is a function (question, doc_id) -> ranked [(component_id, score)].
Page runs are built from component runs (page score = max of its components' scores).

Reported per system:
  page level  : all questions, and questions from documents with >= 2 pages (the main page number)
  component   : questions with >= 1 relevant component (see eval.build_qrels)
Plus the random page baseline: expected Recall@1 = mean(1 / num_pages).

Every system runs on all questions; metrics are stored per split (test / dev / all, see eval/split.py).
Main tables use TEST; "all" is secondary; dev is only for tuning (eval/tune_fusion.py).

Systems: bm25_all, dense_minilm_all, dense_bge_all (single retrievers over all branches) and the hybrids in
HYBRID_SYSTEMS (fusion over BM25 + BGE (+ CLIP), soft/hard routing, optional cross-encoder rerank).

Outputs: eval/results/{system}.json (metrics per split, latency median/p90, full runs),
eval/results/significance.json, eval/results/summary.md (regenerated from all result files).

Usage:
  .venv/Scripts/python -m eval.build_qrels
  .venv/Scripts/python -m eval.split
  .venv/Scripts/python -m eval.run_eval --systems bm25_all dense_minilm_all dense_bge_all
  .venv/Scripts/python -m eval.run_eval --systems hybrid_weighted_soft hybrid_weighted_soft_rerank --significance
  .venv/Scripts/python -m eval.run_eval --systems bm25_all --failures 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from docscout.config import get_config
from docscout.indexing.build_all import load_index, stale_indexes, stale_warning
from docscout.indexing.dense_index import DenseIndex
from docscout.schema import Component, iter_jsonl
from eval.build_qrels import load_eval_docs, page_key
from eval.metrics import DEFAULT_METRICS, evaluate_rankings, pages_from_components
from eval.split import load_split

Ranking = List[Tuple[str, float]]
System = Callable[[str, str], Ranking]
BRANCHES = ("text", "table", "figure")


class IndexCache:
    """Loads each (branch, retriever) index once."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], object] = {}

    def get(self, branch: str, retriever: str):
        key = (branch, retriever)
        if key not in self._cache:
            self._cache[key] = load_index(branch, retriever, warn=False)
        return self._cache[key]


def merge_max(rankings: Sequence[Ranking]) -> Ranking:
    """Union of rankings; a component seen twice keeps its max score."""
    best: Dict[str, float] = {}
    for r in rankings:
        for cid, s in r:
            if cid not in best or s > best[cid]:
                best[cid] = s
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))


def make_systems(cache: IndexCache, top_k: int) -> Dict[str, System]:
    def bm25_all(question: str, doc_id: str) -> Ranking:
        return merge_max([cache.get(b, "bm25").search(question, top_k=top_k, doc_id=doc_id)
                          for b in BRANCHES if cache.get(b, "bm25") is not None])

    def dense_all(retriever: str) -> System:
        def run(question: str, doc_id: str) -> Ranking:
            idxs = [cache.get(b, retriever) for b in BRANCHES if cache.get(b, retriever) is not None]
            q = idxs[0].encode_query(question)  # same model (and BGE instruction) for every branch
            return merge_max([i.search_vector(q, top_k=top_k, doc_id=doc_id) for i in idxs])
        return run

    return {"bm25_all": bm25_all, "dense_minilm_all": dense_all("dense_minilm"), "dense_bge_all": dense_all("dense_bge")}


def random_baseline(qids: Sequence[str], info: Dict[str, dict]) -> Dict[str, float]:
    """Expected page metrics of a uniformly random page ranking (one relevant page per question)."""
    if not qids:
        return {}
    n = [info[q]["num_pages"] for q in qids]
    harm = lambda k: sum(1.0 / i for i in range(1, k + 1))  # noqa: E731
    return {
        "recall@1": statistics.mean(1 / x for x in n),
        "recall@5": statistics.mean(min(5, x) / x for x in n),
        "recall@10": statistics.mean(min(10, x) / x for x in n),
        "mrr": statistics.mean(harm(x) / x for x in n),
    }


HYBRID_SYSTEMS = {
    # name: (fusion method, routing, rerank, query types)
    "hybrid_weighted_soft": ("weighted_sum", "soft", False, "pred"),
    "hybrid_rrf_soft": ("rrf", "soft", False, "pred"),
    "hybrid_weighted_hard": ("weighted_sum", "hard", False, "pred"),
    "hybrid_weighted_soft_rerank": ("weighted_sum", "soft", True, "pred"),
    "hybrid_weighted_soft_goldtype": ("weighted_sum", "soft", False, "gold"),  # analysis row
}
ANALYSIS_SYSTEMS = {"hybrid_weighted_soft_goldtype"}
QTypes = Dict[Tuple[str, str], str]  # (question, doc_id) -> textual | numerical | visual


def make_hybrid_system(retriever, method: str, routing: str, qtypes: QTypes, fusion_cfg: dict,
                       reranker=None, texts: Optional[Dict[str, str]] = None) -> System:
    """A hybrid system: score every component of the document, fuse, route, optionally rerank.

    The query type is looked up (precomputed), so the classifier is excluded from the timed call."""
    from docscout.retrieval.fusion import LABEL_TO_BRANCH, rank_candidates

    def run(question: str, doc_id: str) -> Ranking:
        qtype = qtypes[(question, doc_id)]
        branches = [LABEL_TO_BRANCH[qtype]] if routing == "hard" else None
        cands = retriever.retrieve(question, doc_id, branches=branches)
        ranked = rank_candidates(cands, qtype, method, routing, fusion_cfg["retriever_weights"],
                                 fusion_cfg["branch_weights"], int(fusion_cfg.get("rrf_k", 60)),
                                 fusion_cfg.get("normalization", "minmax"))
        if reranker is not None:
            ranked = reranker.rerank(question, ranked, texts or {})
        return [(c.id, c.final) for c in ranked]
    return run


def predicted_qtypes(questions: Dict[str, dict], qids: Sequence[str]) -> QTypes:
    """Classifier labels (rules, LLM when unsure; LLM replies come from the disk cache when present)."""
    from docscout.query.classifier import classify
    from docscout.reasoning.llm_client import get_client

    client = get_client()
    out = {}
    for qid in qids:
        q = questions[qid]
        out[(q["question"], q["doc_id"])] = classify(q["question"], client=client)["type"]
    print(f"query types: {dict(Counter(out.values()))} | {client.stats}")
    return out


def gold_qtypes(questions: Dict[str, dict], qids: Sequence[str], results_dir: Path) -> QTypes:
    gold = json.loads((results_dir / "qtypes_gold.json").read_text(encoding="utf-8"))["gold"]
    return {(questions[q]["question"], questions[q]["doc_id"]): gold[q] for q in qids}


def fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.3f}"


def _load_results(results_dir: Path) -> List[dict]:
    rows = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and "splits" in r and "system" in r:
            rows.append(r)
    order = ["bm25_all", "dense_minilm_all", "dense_bge_all"] + list(HYBRID_SYSTEMS)
    return sorted(rows, key=lambda r: (order.index(r["system"]) if r["system"] in order else 99, r["system"]))


def _table(rows: List[dict], split: str, level: str, baseline: Optional[Dict[str, float]], latency: bool) -> List[str]:
    hdr = "| system | R@1 | MRR | R@5 | R@10 | nDCG@10 |" + (" median ms | p90 ms |" if latency else "")
    out = [hdr, "|---" * (6 + (2 if latency else 0)) + "|"]
    if baseline:
        out.append(f"| random (expected) | {fmt(baseline.get('recall@1'))} | {fmt(baseline.get('mrr'))} | "
                   f"{fmt(baseline.get('recall@5'))} | {fmt(baseline.get('recall@10'))} | - |" + (" - | - |" if latency else ""))
    for r in rows:
        m = r["splits"][split][level]
        name = r["system"] + (" (analysis: gold types)" if r["system"] in ANALYSIS_SYSTEMS else "")
        line = (f"| {name} | **{fmt(m['recall@1'])}** | **{fmt(m['mrr'])}** | {fmt(m['recall@5'])} | "
                f"{fmt(m['recall@10'])} | {fmt(m['ndcg@10'])} |")
        if latency:
            line += f" {r['latency_ms_median']:.1f} | {r['latency_ms_p90']:.1f} |"
        out.append(line)
    return out


def write_summary(results_dir: Path, baseline: Dict[str, Dict[str, Dict[str, float]]],
                  counts: Dict[str, Dict[str, int]]) -> Path:
    """Regenerate summary.md from every eval/results/{system}.json (main tables on TEST)."""
    rows = _load_results(results_dir)
    ct, ca = counts["test"], counts["all"]
    lines = [
        "# Retrieval results (within-document search)",
        "",
        f"Dev/test split by document (`eval/results/split.json`). Weights were tuned on dev only. "
        f"**Main tables: TEST** ({ct['docs']} documents, {ct['all']} questions; {ct['multi']} from documents "
        f"with >= 2 pages; {ct['comp']} with >= 1 relevant component). Headline metrics: R@1 and MRR (bold). "
        "R@5 / R@10 / nDCG@10 are secondary: documents are short, so even random pages reach R@5 = 0.77.",
        "",
        "Latency: per query, models loaded, query classifier excluded (see classifier.json for its latency).",
        "",
        "## TEST: page level, documents with >= 2 pages (main)",
        "",
        *_table(rows, "test", "page_multi", baseline["test"]["page_multi"], latency=True),
        "",
        "## TEST: component level (questions with >= 1 relevant component)",
        "",
        *_table(rows, "test", "component", None, latency=False),
        "",
        "## TEST: page level, all questions",
        "",
        *_table(rows, "test", "page_all", baseline["test"]["page_all"], latency=False),
    ]
    sig_f = results_dir / "significance.json"
    if sig_f.exists():
        sig = json.loads(sig_f.read_text(encoding="utf-8"))
        lines += ["", f"## TEST: paired t-test vs {sig['baseline']} (ranx, p < {sig['max_p']})", "",
                  "| system | metric | system score | baseline score | p-value | significant |", "|---|---|---|---|---|---|"]
        for row in sig["rows"]:
            lines.append(f"| {row['system']} | {row['metric']} | {row['score']:.3f} | {row['baseline_score']:.3f} | "
                         f"{row['p_value']:.4f} | {'yes' if row['significant'] else 'no'} |")
    lines += [
        "",
        f"## Secondary: all {ca['all']} questions (dev + test; dev was used for tuning)",
        "",
        f"Page level, documents with >= 2 pages ({ca['multi']} questions):",
        "",
        *_table(rows, "all", "page_multi", baseline["all"]["page_multi"], latency=False),
        "",
        f"Component level ({ca['comp']} questions):",
        "",
        *_table(rows, "all", "component", None, latency=False),
    ]
    from eval.eval_answers import summary_section  # P6: ANLS table, if eval/results/answers.json exists

    lines += summary_section(results_dir)
    out = results_dir / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def significance(results_dir: Path, qrels_page: Dict[str, Dict[str, int]], qrels_comp: Dict[str, Dict[str, int]],
                 comp_page: Dict[str, str], test_multi: Sequence[str], test_comp: Sequence[str],
                 systems: Sequence[str], baseline: str = "dense_bge_all", max_p: float = 0.05) -> dict:
    """ranx paired Student's t-test of each system vs `baseline` on TEST, for page MRR and component MRR."""
    from ranx import Qrels, Run, compare

    def as_run(runs: Dict[str, Ranking], qids: Sequence[str], name: str) -> Run:
        d = {q: ({cid: float(s) for cid, s in runs.get(q, [])} or {"__none__": 0.0}) for q in qids}
        return Run(d, name=name)

    loaded = {}
    for name in [baseline, *systems]:
        f = results_dir / f"{name}.json"
        if f.exists():
            loaded[name] = json.loads(f.read_text(encoding="utf-8"))["runs"]
    rows = []
    for level, qids, qrels in (("page MRR (>=2 pages)", test_multi, qrels_page), ("component MRR", test_comp, qrels_comp)):
        runs = {}
        for name, r in loaded.items():
            runs[name] = ({q: pages_from_components(r.get(q, []), comp_page) for q in qids}
                          if level.startswith("page") else r)
        ranx_runs = [as_run(runs[n], qids, n) for n in runs]
        rep = compare(qrels=Qrels({q: qrels[q] for q in qids}), runs=ranx_runs, metrics=["mrr"],
                      stat_test="student", max_p=max_p, make_comparable=True).to_dict()
        for name in systems:
            if name not in runs:
                continue
            p = rep[name]["comparisons"][baseline]["mrr"]
            rows.append({"system": name, "metric": level, "score": rep[name]["scores"]["mrr"],
                         "baseline_score": rep[baseline]["scores"]["mrr"], "p_value": float(p),
                         "significant": bool(p < max_p)})
    out = {"baseline": baseline, "max_p": max_p, "test": "paired Student's t-test (ranx stat_test='student')",
           "rows": rows}
    (results_dir / "significance.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out

def show_failures(system: str, runs: Dict[str, Ranking], questions: Dict[str, dict], qrels_comp: Dict[str, Dict[str, int]],
                  comps: Dict[str, Component], n: int) -> None:
    """Print component-level failures: no relevant component in the top 10.

    Failures with at least 3 retrieved components are shown first (they have a top-3 to inspect)."""
    shown = 0
    cands = []
    for qid, rels in qrels_comp.items():
        ranking = runs.get(qid, [])
        ranks = [i for i, (cid, _) in enumerate(ranking, start=1) if cid in rels]
        if not ranks or ranks[0] > 10:
            cands.append((len(ranking) < 3, qid))
    print(f"\n{system}: {len(cands)} component-level failures (no relevant component in top 10), "
          f"{sum(1 for few, _ in cands if few)} of them with < 3 retrieved components")
    for _, qid in sorted(cands, key=lambda x: x[0]):
        rels = qrels_comp[qid]
        ranking = runs.get(qid, [])
        ranks = [i for i, (cid, _) in enumerate(ranking, start=1) if cid in rels]
        q = questions[qid]
        print(f"\n--- {system} failure: question {qid} (doc {q['doc_id']}, answer page {q['answer_page']})")
        print(f"Q: {q['question']}")
        print(f"gold answers: {q['answers']}")
        print(f"relevant components: {sorted(rels)} | first relevant rank: {ranks[0] if ranks else 'not retrieved'}")
        for i, (cid, s) in enumerate(ranking[:3], start=1):
            c = comps[cid]
            print(f"  top{i} {s:.3f} {cid} [{c.type}, p{c.page}] {c.text_form[:160]!r}")
        for cid in sorted(rels)[:1]:
            print(f"  relevant: {cid} {comps[cid].text_form[:200]!r}")
        shown += 1
        if shown >= n:
            break


def subset_metrics(qids: Sequence[str], info: Dict[str, dict], qrels_page: dict, qrels_comp: dict,
                   page_runs: Dict[str, Ranking], runs: Dict[str, Ranking]) -> Dict[str, Dict[str, float]]:
    multi = [q for q in qids if info[q]["num_pages"] >= 2]
    comp_q = [q for q in qids if q in qrels_comp]
    return {
        "page_all": evaluate_rankings({q: qrels_page[q] for q in qids}, page_runs, DEFAULT_METRICS),
        "page_multi": evaluate_rankings({q: qrels_page[q] for q in multi}, page_runs, DEFAULT_METRICS),
        "component": evaluate_rankings({q: qrels_comp[q] for q in comp_q}, runs, DEFAULT_METRICS),
    }


def main(argv: Optional[List[str]] = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", default=["bm25_all", "dense_minilm_all", "dense_bge_all"])
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N questions")
    ap.add_argument("--top-k", type=int, default=1000, help="candidates per branch (1000 = whole document)")
    ap.add_argument("--failures", type=int, default=0, help="print N component-level failures per system")
    ap.add_argument("--significance", action="store_true",
                    help="paired t-test (TEST) of hybrid_weighted_soft and _rerank vs dense_bge_all")
    ap.add_argument("--summary-only", action="store_true", help="only regenerate summary.md")
    args = ap.parse_args(argv)

    results_dir: Path = cfg["paths"]["results_dir"]
    try:
        qrels_page = json.loads((results_dir / "qrels_page.json").read_text(encoding="utf-8"))
        qrels_comp = json.loads((results_dir / "qrels_component.json").read_text(encoding="utf-8"))
        info = json.loads((results_dir / "qrels_info.json").read_text(encoding="utf-8"))
        split = load_split(cfg)
    except FileNotFoundError as e:
        print(f"missing {e.filename}. Run: .venv/Scripts/python -m eval.build_qrels and -m eval.split")
        return 1
    stale = stale_indexes(cfg)
    if stale:
        print(stale_warning(stale))
        return 2

    eval_docs = load_eval_docs(cfg)
    questions = {q["question_id"]: q for q in iter_jsonl(cfg["paths"]["questions_file"]) if q["doc_id"] in set(eval_docs)}
    qids = [qid for qid in questions if qid in qrels_page]
    if args.limit:
        qids = qids[: args.limit]
    subsets = {"test": [q for q in qids if q in set(split["test"]["question_ids"])],
               "dev": [q for q in qids if q in set(split["dev"]["question_ids"])], "all": qids}
    comps = {r["id"]: Component.from_dict(r) for r in iter_jsonl(cfg["paths"]["components_file"])}
    comp_page = {cid: page_key(c.doc_id, c.page) for cid, c in comps.items()}

    baseline, counts = {}, {}
    for name, qs in subsets.items():
        multi = [q for q in qs if info[q]["num_pages"] >= 2]
        baseline[name] = {"page_all": random_baseline(qs, info), "page_multi": random_baseline(multi, info)}
        counts[name] = {"docs": len({questions[q]["doc_id"] for q in qs}), "all": len(qs), "multi": len(multi),
                        "comp": sum(1 for q in qs if q in qrels_comp)}
        print(f"{name:<4}: questions {len(qs)} | docs>=2 pages: {len(multi)} | with relevant component: "
              f"{counts[name]['comp']} | random page R@1 (>=2 pages) = {baseline[name]['page_multi'].get('recall@1', 0):.3f}")
    (results_dir / "baseline.json").write_text(json.dumps({"baseline": baseline, "counts": counts}, indent=1),
                                               encoding="utf-8")
    if args.summary_only:
        out = write_summary(results_dir, baseline, counts)
        print(f"wrote {out.relative_to(cfg['repo_root'])}")
        return 0

    cache = IndexCache()
    systems = make_systems(cache, args.top_k)
    hybrid_needed = [n for n in args.systems if n in HYBRID_SYSTEMS]
    if hybrid_needed:
        from docscout.rerank.cross_encoder import CrossEncoderReranker
        from docscout.retrieval.retrievers import HybridRetriever

        retriever = HybridRetriever(load=cache.get, comp_meta=comps, cfg=cfg)
        qtypes = {"pred": predicted_qtypes(questions, qids)}
        if any(HYBRID_SYSTEMS[n][3] == "gold" for n in hybrid_needed):
            qtypes["gold"] = gold_qtypes(questions, qids, results_dir)
        reranker = CrossEncoderReranker(cfg=cfg) if any(HYBRID_SYSTEMS[n][2] for n in hybrid_needed) else None
        texts = {cid: c.text_form for cid, c in comps.items()}
        for n in hybrid_needed:
            method, routing, rr, types = HYBRID_SYSTEMS[n]
            systems[n] = make_hybrid_system(retriever, method, routing, qtypes[types], cfg["fusion"],
                                            reranker if rr else None, texts)
        print(f"fusion config: retriever_weights={cfg['fusion']['retriever_weights']} "
              f"branch_weights={cfg['fusion']['branch_weights']} rerank_top_n={cfg['retrieval']['rerank_top_n']}")

    for name in args.systems:
        if name not in systems:
            print(f"unknown system {name!r}; available: {sorted(set(systems) | set(HYBRID_SYSTEMS))}")
            return 1
        fn = systems[name]
        fn(questions[qids[0]]["question"], questions[qids[0]]["doc_id"])  # warm-up (model loading)
        runs: Dict[str, Ranking] = {}
        lat = []
        for qid in tqdm(qids, desc=name, unit="q", leave=False):
            q = questions[qid]
            t = time.perf_counter()
            runs[qid] = fn(q["question"], q["doc_id"])
            lat.append((time.perf_counter() - t) * 1000)
            assert all(comps[cid].doc_id == q["doc_id"] for cid, _ in runs[qid]), "doc filter leaked"
        page_runs = {qid: pages_from_components(r, comp_page) for qid, r in runs.items()}
        splits = {s: subset_metrics(qs, info, qrels_page, qrels_comp, page_runs, runs) for s, qs in subsets.items()}
        res = {"system": name, "n_questions": len(qids), "counts": counts, "splits": splits,
               "metrics": splits["test"], "latency_ms_median": statistics.median(lat),
               "latency_ms_p90": float(np.percentile(lat, 90)), "top_k": args.top_k,
               "empty_runs": sum(1 for r in runs.values() if not r),
               "runs": runs}
        (results_dir / f"{name}.json").write_text(json.dumps(res), encoding="utf-8")
        pm, cm = splits["test"]["page_multi"], splits["test"]["component"]
        print(f"[{name}] median {res['latency_ms_median']:.1f} ms, p90 {res['latency_ms_p90']:.1f} ms | empty runs "
              f"{res['empty_runs']} | TEST page>=2: R@1={pm['recall@1']:.3f} MRR={pm['mrr']:.3f} | "
              f"TEST comp: R@1={cm['recall@1']:.3f} MRR={cm['mrr']:.3f}")
        if args.failures:
            comp_q = [q for q in qids if q in qrels_comp]
            show_failures(name, runs, questions, {q: qrels_comp[q] for q in comp_q}, comps, args.failures)

    if args.significance:
        test = subsets["test"]
        sig = significance(results_dir, qrels_page, qrels_comp, comp_page,
                           [q for q in test if info[q]["num_pages"] >= 2], [q for q in test if q in qrels_comp],
                           ["hybrid_weighted_soft", "hybrid_weighted_soft_rerank"])
        for r in sig["rows"]:
            print(f"t-test {r['system']:<28} {r['metric']:<22} {r['score']:.3f} vs {r['baseline_score']:.3f} "
                  f"p={r['p_value']:.4f} {'SIGNIFICANT' if r['significant'] else 'n.s.'}")
    out = write_summary(results_dir, baseline, counts)
    print(f"wrote {out.relative_to(cfg['repo_root'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
