"""Grid-search the weighted-sum fusion weights on the DEV split only (test questions are never loaded).

Grid:
  retriever weights: BM25 in {0.3, 0.5, 0.7}, BGE = 1 - BM25, CLIP = 0.3 on figures (before renormalization)
  branch weights per query type: the type's own branch gets {0.4, 0.6, 0.8}; the rest is split in a fixed ratio
    textual   (own = text)  : table : figure = 5 : 3
    numerical (own = table) : text  : figure = 3 : 1
    visual    (own = figure): text  : table  = 5 : 3
  -> 3 x 3 x 3 x 3 = 81 settings. Query types come from the classifier (as in hybrid_weighted_soft).
Objective: mean(dev page MRR on documents with >= 2 pages, dev component MRR).

Retrieval + min-max normalization run once per dev question; only the fusion step is repeated per setting.
Output: eval/results/tuning.json (full grid). The chosen values are copied into configs/config.yaml by hand.

Usage: .venv/Scripts/python -m eval.tune_fusion
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from docscout.config import get_config
from docscout.retrieval.fusion import weighted_sum
from docscout.retrieval.normalize import normalize_candidates
from docscout.retrieval.retrievers import HybridRetriever
from docscout.schema import Component, iter_jsonl
from eval.build_qrels import page_key
from eval.metrics import evaluate_rankings, pages_from_components
from eval.run_eval import IndexCache, predicted_qtypes
from eval.split import load_split

BM25_GRID = (0.3, 0.5, 0.7)
CLIP_W = 0.3
OWN_GRID = (0.4, 0.6, 0.8)
REST_SPLIT = {  # type: (own branch, {other branch: share of the remainder})
    "textual": ("text", {"table": 5 / 8, "figure": 3 / 8}),
    "numerical": ("table", {"text": 3 / 4, "figure": 1 / 4}),
    "visual": ("figure", {"text": 5 / 8, "table": 3 / 8}),
}


def branch_preset(qtype: str, own: float) -> Dict[str, float]:
    b, rest = REST_SPLIT[qtype]
    out = {b: own}
    out.update({o: round((1 - own) * share, 4) for o, share in rest.items()})
    return {k: out[k] for k in ("text", "table", "figure")}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=15, help="print the best N settings")
    args = ap.parse_args(argv)
    cfg = get_config()
    results_dir: Path = cfg["paths"]["results_dir"]
    split = load_split(cfg)
    dev = split["dev"]["question_ids"]
    dev_docs = set(split["dev"]["doc_ids"])
    qrels_page = json.loads((results_dir / "qrels_page.json").read_text(encoding="utf-8"))
    qrels_comp = json.loads((results_dir / "qrels_component.json").read_text(encoding="utf-8"))
    info = json.loads((results_dir / "qrels_info.json").read_text(encoding="utf-8"))
    questions = {q["question_id"]: q for q in iter_jsonl(cfg["paths"]["questions_file"]) if q["question_id"] in set(dev)}
    assert all(q["doc_id"] in dev_docs for q in questions.values()), "a non-dev document leaked into tuning"
    multi = [q for q in dev if info[q]["num_pages"] >= 2]
    comp_q = [q for q in dev if q in qrels_comp]
    print(f"DEV only: {len(dev)} questions, {len(dev_docs)} documents | page>=2: {len(multi)} | component: {len(comp_q)}")

    comps = {r["id"]: Component.from_dict(r) for r in iter_jsonl(cfg["paths"]["components_file"])}
    comp_page = {cid: page_key(c.doc_id, c.page) for cid, c in comps.items()}
    retriever = HybridRetriever(load=IndexCache().get, comp_meta=comps, cfg=cfg)
    qtypes = predicted_qtypes(questions, dev)
    cands = {}
    for qid in tqdm(dev, desc="retrieve + normalize (dev)", leave=False):
        q = questions[qid]
        cands[qid] = normalize_candidates(retriever.retrieve(q["question"], q["doc_id"]), cfg["fusion"]["normalization"])
    by_type = {qid: qtypes[(questions[qid]["question"], questions[qid]["doc_id"])] for qid in dev}

    def score(rw: Dict[str, float], bw: Dict[str, Dict[str, float]]) -> Tuple[float, float, float]:
        runs = {qid: [(c.id, c.final) for c in weighted_sum(cands[qid], rw, bw[by_type[qid]])] for qid in dev}
        page_runs = {qid: pages_from_components(r, comp_page) for qid, r in runs.items()}
        pm = evaluate_rankings({q: qrels_page[q] for q in multi}, page_runs, ["mrr"])["mrr"]
        cm = evaluate_rankings({q: qrels_comp[q] for q in comp_q}, runs, ["mrr"])["mrr"]
        return (pm + cm) / 2, pm, cm

    base = score(cfg["fusion"]["retriever_weights"], cfg["fusion"]["branch_weights"])
    print(f"current config (before tuning): objective={base[0]:.4f} page_mrr={base[1]:.4f} comp_mrr={base[2]:.4f}")
    grid = []
    for bm25, ot, on, ov in tqdm(list(itertools.product(BM25_GRID, OWN_GRID, OWN_GRID, OWN_GRID)), desc="grid", leave=False):
        rw = {"bm25": bm25, "dense": round(1 - bm25, 4), "clip": CLIP_W}
        bw = {"textual": branch_preset("textual", ot), "numerical": branch_preset("numerical", on),
              "visual": branch_preset("visual", ov)}
        obj, pm, cm = score(rw, bw)
        grid.append({"bm25": bm25, "textual_text": ot, "numerical_table": on, "visual_figure": ov,
                     "objective": obj, "page_mrr": pm, "comp_mrr": cm, "retriever_weights": rw, "branch_weights": bw})
    # best objective; ties -> the earlier grid point (smaller weights), deterministic
    grid_sorted = sorted(enumerate(grid), key=lambda kv: (-round(kv[1]["objective"], 6), kv[0]))
    print(f"\n{'rank':>4} {'bm25':>5} {'txt.text':>8} {'num.table':>9} {'vis.fig':>7} {'objective':>9} {'pageMRR':>8} {'compMRR':>8}")
    for rank, (_, g) in enumerate(grid_sorted[: args.top], start=1):
        print(f"{rank:>4} {g['bm25']:>5} {g['textual_text']:>8} {g['numerical_table']:>9} {g['visual_figure']:>7} "
              f"{g['objective']:>9.4f} {g['page_mrr']:>8.4f} {g['comp_mrr']:>8.4f}")
    # marginal view: best objective per value of each knob
    for knob, vals in (("bm25", BM25_GRID), ("textual_text", OWN_GRID), ("numerical_table", OWN_GRID),
                       ("visual_figure", OWN_GRID)):
        best = {v: max(g["objective"] for g in grid if g[knob] == v) for v in vals}
        print(f"best objective by {knob:<16}: " + "  ".join(f"{v}->{o:.4f}" for v, o in best.items()))
    best = grid_sorted[0][1]
    types_count = {t: sum(1 for v in by_type.values() if v == t) for t in REST_SPLIT}
    print(f"\ndev questions per predicted type: {types_count}")
    print(f"CHOSEN: retriever_weights={best['retriever_weights']}")
    for t, w in best["branch_weights"].items():
        print(f"        {t:<9} {w}")
    (results_dir / "tuning.json").write_text(json.dumps({"split": "dev", "n_questions": len(dev), "before": base,
                                                         "dev_type_counts": types_count, "grid": grid,
                                                         "chosen": best}, indent=1), encoding="utf-8")
    print(f"wrote {(results_dir / 'tuning.json').relative_to(cfg['repo_root'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
