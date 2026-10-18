"""Evaluate retrieval systems at page and component level (within the question's document).

A system is a function (question, doc_id) -> ranked [(component_id, score)].
Page runs are built from component runs (page score = max of its components' scores).

Reported per system:
  page level  : all questions, and questions from documents with >= 2 pages (the main page number)
  component   : questions with >= 1 relevant component (see eval.build_qrels)
Plus the random page baseline: expected Recall@1 = mean(1 / num_pages).

Outputs: eval/results/{system}.json and eval/results/summary.md (regenerated from all result files).

Usage:
  .venv/Scripts/python -m eval.build_qrels
  .venv/Scripts/python -m eval.run_eval --systems bm25_all dense_minilm_all dense_bge_all
  .venv/Scripts/python -m eval.run_eval --systems bm25_all --failures 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from docscout.config import get_config
from docscout.indexing.build_all import load_index, stale_indexes, stale_warning
from docscout.indexing.dense_index import DenseIndex
from docscout.schema import Component, iter_jsonl
from eval.build_qrels import load_eval_docs, page_key
from eval.metrics import DEFAULT_METRICS, evaluate_rankings, pages_from_components

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


def fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.3f}"


def write_summary(results_dir: Path, baseline: Dict[str, Dict[str, float]], counts: Dict[str, int]) -> Path:
    """Regenerate summary.md from every eval/results/{system}.json."""
    rows = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name.startswith("qrels") or f.name == "baseline.json":
            continue
        rows.append(json.loads(f.read_text(encoding="utf-8")))
    lines = [
        "# Retrieval results (within-document search)",
        "",
        f"Evaluation set: {counts['docs']} documents from `eval_docs.txt`, {counts['all']} questions. "
        f"Page metrics on documents with >= 2 pages: {counts['multi']} questions (main page-level number). "
        f"Component metrics: {counts['comp']} questions with >= 1 relevant component.",
        "",
        "## Page level: documents with >= 2 pages (main)",
        "",
        "| system | R@1 | R@5 | R@10 | MRR | nDCG@10 | median ms/query |",
        "|---|---|---|---|---|---|---|",
    ]
    b = baseline.get("page_multi", {})
    lines.append(f"| random (expected) | {fmt(b.get('recall@1'))} | {fmt(b.get('recall@5'))} | "
                 f"{fmt(b.get('recall@10'))} | {fmt(b.get('mrr'))} | - | - |")
    for r in rows:
        m = r["metrics"]["page_multi"]
        lines.append(f"| {r['system']} | {fmt(m['recall@1'])} | {fmt(m['recall@5'])} | {fmt(m['recall@10'])} | "
                     f"{fmt(m['mrr'])} | {fmt(m['ndcg@10'])} | {r['latency_ms_median']:.1f} |")
    lines += ["", "## Page level: all questions", "", "| system | R@1 | R@5 | R@10 | MRR | nDCG@10 |", "|---|---|---|---|---|---|"]
    b = baseline.get("page_all", {})
    lines.append(f"| random (expected) | {fmt(b.get('recall@1'))} | {fmt(b.get('recall@5'))} | "
                 f"{fmt(b.get('recall@10'))} | {fmt(b.get('mrr'))} | - |")
    for r in rows:
        m = r["metrics"]["page_all"]
        lines.append(f"| {r['system']} | {fmt(m['recall@1'])} | {fmt(m['recall@5'])} | {fmt(m['recall@10'])} | "
                     f"{fmt(m['mrr'])} | {fmt(m['ndcg@10'])} |")
    lines += ["", "## Component level (questions with >= 1 relevant component)", "",
              "| system | R@1 | R@5 | R@10 | MRR | nDCG@10 |", "|---|---|---|---|---|---|"]
    for r in rows:
        m = r["metrics"]["component"]
        lines.append(f"| {r['system']} | {fmt(m['recall@1'])} | {fmt(m['recall@5'])} | {fmt(m['recall@10'])} | "
                     f"{fmt(m['mrr'])} | {fmt(m['ndcg@10'])} |")
    out = results_dir / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
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


def main(argv: Optional[List[str]] = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", default=["bm25_all", "dense_minilm_all", "dense_bge_all"])
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N questions")
    ap.add_argument("--top-k", type=int, default=1000, help="candidates per branch (1000 = whole document)")
    ap.add_argument("--failures", type=int, default=0, help="print N component-level failures per system")
    args = ap.parse_args(argv)

    results_dir: Path = cfg["paths"]["results_dir"]
    try:
        qrels_page = json.loads((results_dir / "qrels_page.json").read_text(encoding="utf-8"))
        qrels_comp = json.loads((results_dir / "qrels_component.json").read_text(encoding="utf-8"))
        info = json.loads((results_dir / "qrels_info.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        print("qrels not found. Run: .venv/Scripts/python -m eval.build_qrels")
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
    comps = {r["id"]: Component.from_dict(r) for r in iter_jsonl(cfg["paths"]["components_file"])}
    comp_page = {cid: page_key(c.doc_id, c.page) for cid, c in comps.items()}

    multi = [q for q in qids if info[q]["num_pages"] >= 2]
    comp_q = [q for q in qids if q in qrels_comp]
    baseline = {"page_all": random_baseline(qids, info), "page_multi": random_baseline(multi, info)}
    counts = {"docs": len(eval_docs), "all": len(qids), "multi": len(multi), "comp": len(comp_q)}
    print(f"questions: {len(qids)} | docs>=2 pages: {len(multi)} | with relevant component: {len(comp_q)}")
    print(f"random page Recall@1 (expected): all = {baseline['page_all']['recall@1']:.3f}, "
          f">=2 pages = {baseline['page_multi']['recall@1']:.3f}")
    (results_dir / "baseline.json").write_text(json.dumps({"baseline": baseline, "counts": counts}, indent=1), encoding="utf-8")

    cache = IndexCache()
    systems = make_systems(cache, args.top_k)
    for name in args.systems:
        if name not in systems:
            print(f"unknown system {name!r}; available: {sorted(systems)}")
            return 1
        fn = systems[name]
        fn(questions[qids[0]]["question"], questions[qids[0]]["doc_id"])  # warm-up (model loading)
        runs: Dict[str, Ranking] = {}
        lat = []
        for qid in qids:
            q = questions[qid]
            t = time.perf_counter()
            runs[qid] = fn(q["question"], q["doc_id"])
            lat.append((time.perf_counter() - t) * 1000)
            assert all(comps[cid].doc_id == q["doc_id"] for cid, _ in runs[qid]), "doc filter leaked"
        page_runs = {qid: pages_from_components(r, comp_page) for qid, r in runs.items()}
        metrics = {
            "page_all": evaluate_rankings({q: qrels_page[q] for q in qids}, page_runs, DEFAULT_METRICS),
            "page_multi": evaluate_rankings({q: qrels_page[q] for q in multi}, page_runs, DEFAULT_METRICS),
            "component": evaluate_rankings({q: qrels_comp[q] for q in comp_q}, runs, DEFAULT_METRICS),
        }
        res = {"system": name, "n_questions": len(qids), "counts": counts, "metrics": metrics,
               "latency_ms_median": statistics.median(lat), "top_k": args.top_k,
               "runs": {qid: r[:50] for qid, r in runs.items()}}
        (results_dir / f"{name}.json").write_text(json.dumps(res), encoding="utf-8")
        pm, pa, cm = metrics["page_multi"], metrics["page_all"], metrics["component"]
        print(f"\n[{name}] median {res['latency_ms_median']:.1f} ms/query")
        print(f"  page (>=2 pages): " + " ".join(f"{k}={v:.3f}" for k, v in pm.items()))
        print(f"  page (all)      : " + " ".join(f"{k}={v:.3f}" for k, v in pa.items()))
        print(f"  component       : " + " ".join(f"{k}={v:.3f}" for k, v in cm.items()))
        if args.failures:
            show_failures(name, runs, questions, {q: qrels_comp[q] for q in comp_q}, comps, args.failures)
    out = write_summary(results_dir, baseline, counts)
    print(f"\nwrote {out.relative_to(cfg['repo_root'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
