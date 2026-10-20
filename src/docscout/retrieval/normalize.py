"""Per-list score normalization (one list = one retriever within one branch, within one document).

minmax: (s - min) / (max - min). If all scores are equal: every item gets 0, or 1 if the list has exactly one item.
zscore: (s - mean) / std (population std); all-equal -> 0 (one item -> 1, as for minmax).

Note: min-max maps the best item of EVERY list to 1.0, even when that branch is weak for the query
(e.g. the only table of a document always gets 1.0 from both of its retrievers). Branch weights are what
keeps a weak branch's 1.0 from dominating.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

from docscout.retrieval.retrievers import Candidate, lists_by_branch_retriever


def minmax(scores: Mapping[str, float]) -> Dict[str, float]:
    """Min-max normalize {id: score} to [0, 1]."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        v = 1.0 if len(scores) == 1 else 0.0
        return {k: v for k in scores}
    return {k: (s - lo) / (hi - lo) for k, s in scores.items()}


def zscore(scores: Mapping[str, float]) -> Dict[str, float]:
    """Z-score normalize {id: score}."""
    if not scores:
        return {}
    vals = list(scores.values())
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    if std == 0:
        v = 1.0 if len(scores) == 1 else 0.0
        return {k: v for k in scores}
    return {k: (s - mean) / std for k, s in scores.items()}


METHODS = {"minmax": minmax, "zscore": zscore}


def normalize(scores: Mapping[str, float], method: str = "minmax") -> Dict[str, float]:
    if method not in METHODS:
        raise ValueError(f"unknown normalization {method!r}; use one of {sorted(METHODS)}")
    return METHODS[method](scores)


def normalize_candidates(cands: Sequence[Candidate], method: str = "minmax") -> Sequence[Candidate]:
    """Fill cand.norm[retriever], normalizing each (branch, retriever) list separately. Returns `cands`."""
    for (_, r), lst in lists_by_branch_retriever(cands).items():
        normed = normalize({c.id: c.raw[r] for c in lst}, method)
        for c in lst:
            c.norm[r] = normed[c.id]
    return cands
