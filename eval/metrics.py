"""Evaluation metrics: ranx wrappers for ranking metrics, and ANLS for answers."""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from rapidfuzz.distance import Levenshtein

DEFAULT_METRICS = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]

Ranking = Sequence[Tuple[str, float]]


def evaluate_rankings(qrels: Mapping[str, Mapping[str, int]], runs: Mapping[str, Ranking],
                      metrics: Sequence[str] = DEFAULT_METRICS) -> Dict[str, float]:
    """Mean metrics over the queries in `qrels` (ranx). Queries missing from `runs` count as empty runs."""
    from ranx import Qrels, Run, evaluate

    q = Qrels({qid: dict(rels) for qid, rels in qrels.items()})
    run_dict = {}
    for qid in qrels:
        ranking = runs.get(qid) or []
        # ranx needs >= 1 doc per query; an empty ranking gets a dummy non-relevant doc
        run_dict[qid] = {doc: float(s) for doc, s in ranking} if ranking else {"__none__": 0.0}
    r = Run(run_dict)
    res = evaluate(q, r, list(metrics), make_comparable=True)
    if isinstance(res, float):  # single metric
        res = {metrics[0]: res}
    return {m: float(v) for m, v in res.items()}


def pages_from_components(ranking: Ranking, comp_page: Mapping[str, str]) -> List[Tuple[str, float]]:
    """Page ranking from a component ranking: page score = max score of its components."""
    best: Dict[str, float] = {}
    for cid, score in ranking:
        p = comp_page[cid]
        if p not in best or score > best[p]:
            best[p] = score
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))


def anls_single(prediction: str, golds: Iterable[str], threshold: float = 0.5) -> float:
    """ANLS for one question (DocVQA definition): max over gold answers of 1 - NL if NL < threshold else 0,
    where NL = Levenshtein(pred, gold) / max(len(pred), len(gold)) on lowercased, stripped strings."""
    p = " ".join(str(prediction).lower().split())
    best = 0.0
    for g in golds:
        g = " ".join(str(g).lower().split())
        if not p and not g:
            return 1.0
        nl = Levenshtein.distance(p, g) / max(len(p), len(g), 1)
        best = max(best, 1.0 - nl if nl < threshold else 0.0)
    return best


def anls(predictions: Sequence[str], golds: Sequence[Sequence[str]], threshold: float = 0.5) -> float:
    """Mean ANLS over questions."""
    if not predictions:
        return 0.0
    return sum(anls_single(p, g, threshold) for p, g in zip(predictions, golds)) / len(predictions)
