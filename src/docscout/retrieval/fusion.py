"""Score fusion across branches and retrievers.

weighted_sum (main):  fused(c) = w_branch[branch(c)] * sum_r w'_r * norm_r(c)
    w'_r = w_ret[r] / sum of w_ret over the retrievers the candidate's branch has in this document
    (text/table: BM25 + BGE; figure: BM25 + BGE + CLIP). A missing score counts as 0.
rrf (comparison):     rrf(c) = sum over (branch, retriever) lists containing c of w_branch[branch] / (k + rank)
    Ranks come from RAW scores (no normalization), 1-based, ties broken by component id.
    A BM25 score of 0 means "no query term matched": that item is not ranked in the BM25 list.

Ties in every final ranking are broken deterministically by component id.
Hard routing keeps only the predicted branch's candidates; soft routing keeps all and only sets weights.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from docscout.retrieval.retrievers import RETRIEVER_WEIGHT_KEY, Candidate, lists_by_branch_retriever


def sort_candidates(cands: Sequence[Candidate]) -> List[Candidate]:
    """Best first by .final; ties by component id."""
    return sorted(cands, key=lambda c: (-c.final, c.id))


def branch_retrievers(cands: Sequence[Candidate]) -> Dict[str, List[str]]:
    """Retrievers present per branch in this candidate set."""
    out: Dict[str, set] = {}
    for c in cands:
        out.setdefault(c.branch, set()).update(c.raw)
    return {b: sorted(rs) for b, rs in out.items()}


def effective_retriever_weights(retrievers: Sequence[str], retriever_weights: Mapping[str, float]) -> Dict[str, float]:
    """Retriever weights renormalized over the retrievers a branch actually has."""
    w = {r: float(retriever_weights[RETRIEVER_WEIGHT_KEY.get(r, r)]) for r in retrievers}
    tot = sum(w.values())
    return {r: (v / tot if tot > 0 else 0.0) for r, v in w.items()}


def weighted_sum(cands: Sequence[Candidate], retriever_weights: Mapping[str, float],
                 branch_weights: Mapping[str, float]) -> List[Candidate]:
    """Set c.fused (and c.final) from normalized scores; return candidates sorted best first.

    Requires normalize_candidates() to have filled c.norm."""
    eff = {b: effective_retriever_weights(rs, retriever_weights) for b, rs in branch_retrievers(cands).items()}
    for c in cands:
        s = sum(w * c.norm.get(r, 0.0) for r, w in eff[c.branch].items())
        c.fused = float(branch_weights.get(c.branch, 0.0)) * s
        c.final = c.fused
    return sort_candidates(cands)


def rrf(cands: Sequence[Candidate], branch_weights: Mapping[str, float], k: int = 60) -> List[Candidate]:
    """Set c.rrf (and c.final) from per-list RANKS of raw scores; return candidates sorted best first."""
    for c in cands:
        c.rrf = 0.0
    for (branch, r), lst in lists_by_branch_retriever(cands).items():
        ranked = [c for c in lst if not (r == "bm25" and c.raw[r] <= 0.0)]
        ranked.sort(key=lambda c: (-c.raw[r], c.id))
        w = float(branch_weights.get(branch, 0.0))
        for rank, c in enumerate(ranked, start=1):
            c.rrf += w / (k + rank)
    for c in cands:
        c.final = c.rrf
    return sort_candidates(cands)


def hard_route(cands: Sequence[Candidate], branch: str) -> List[Candidate]:
    """Hard routing: keep only the predicted branch's candidates."""
    return [c for c in cands if c.branch == branch]


LABEL_TO_BRANCH = {"textual": "text", "numerical": "table", "visual": "figure"}


def rank_candidates(cands: Sequence[Candidate], qtype: str, method: str, routing: str,
                    retriever_weights: Mapping[str, float], branch_weights: Mapping[str, Mapping[str, float]],
                    rrf_k: int = 60, normalization: str = "minmax") -> List[Candidate]:
    """Normalize (weighted_sum only) + fuse + route one document's candidates for a query of type `qtype`.

    routing "soft": every branch, weights branch_weights[qtype]; "hard": only LABEL_TO_BRANCH[qtype]'s
    candidates. Normalization runs over the full lists before hard routing filters them."""
    from docscout.retrieval.normalize import normalize_candidates

    bw = branch_weights[qtype]
    if method == "weighted_sum":
        normalize_candidates(cands, normalization)
        ranked = weighted_sum(cands, retriever_weights, bw)
    elif method == "rrf":
        ranked = rrf(cands, bw, rrf_k)
    else:
        raise ValueError(f"unknown fusion method {method!r}")
    if routing == "hard":
        ranked = hard_route(ranked, LABEL_TO_BRANCH[qtype])
    elif routing != "soft":
        raise ValueError(f"unknown routing {routing!r}")
    return ranked
