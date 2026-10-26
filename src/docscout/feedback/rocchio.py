"""Session Rocchio, plugged into the pipeline's feedback_adjust hook (step 6).

With votes for this (session, document, question):
  1. q' = alpha*q + beta*mean(BGE vectors of up-votes) - gamma*mean(BGE vectors of down-votes), L2-normalized.
  2. Dense (BGE) raw scores of every component of the document are recomputed with q'; then min-max normalization
     and weighted-sum fusion run again exactly as in steps 3-4.
  3. The cross-encoder reranks the new top-N (original query text; scores cached per (query, text)).
  4. Over the top-N: final = (1 - lambda) * minmax(CE) + lambda * minmax(cos(q', d)). Items below the top-N stay
     below it (final = -1 - i).
  5. The cross-session memory prior (memory.py), if given, is added to final.
  6. Up-voted items are pinned above everything (their order among themselves kept); down-voted items are removed.

Without votes and without a memory prior the hook returns the ranking untouched: final stays the raw CE score (P6).
With a memory prior only, lambda is 0: final = minmax(CE) + prior (the prior lives on the [0, 1] scale).

`hook.deltas` = feedback change per item on the normalized scale (final - minmax(CE), pin offset excluded) and
`hook.tags` = item labels ("rocchio", "memory", "pinned"); the pipeline puts both in the why-line.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from docscout.retrieval.fusion import sort_candidates, weighted_sum
from docscout.retrieval.normalize import minmax, normalize_candidates
from docscout.retrieval.retrievers import Candidate

PIN_OFFSET = 10.0  # pinned items: final = PIN_OFFSET + their feedback score (always above the [-1, 1.15] range)


def rocchio_query(q: np.ndarray, pos: Sequence[np.ndarray], neg: Sequence[np.ndarray],
                  alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.15) -> np.ndarray:
    """alpha*q + beta*mean(pos) - gamma*mean(neg), L2-normalized (an empty group contributes nothing)."""
    v = alpha * np.asarray(q, dtype=np.float64)
    if len(pos):
        v = v + beta * np.mean(np.asarray(pos, dtype=np.float64), axis=0)
    if len(neg):
        v = v - gamma * np.mean(np.asarray(neg, dtype=np.float64), axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def component_vector_lookup(retriever: Any, name: str = "dense_bge") -> Callable[[str], Optional[np.ndarray]]:
    """cid -> stored (normalized) BGE vector, from the retriever's branch indexes; the id map is built once."""
    rows = getattr(retriever, "_fb_vec_rows", None)
    if rows is None:
        rows = {}
        for rets in retriever.indexes.values():
            idx = rets.get(name)
            if idx is not None and getattr(idx, "emb", None) is not None:
                for i, cid in enumerate(idx.ids):
                    rows[cid] = (idx, i)
        retriever._fb_vec_rows = rows

    def get(cid: str) -> Optional[np.ndarray]:
        hit = rows.get(cid)
        return None if hit is None else hit[0].emb[hit[1]]
    return get


class CachedScorer:
    """Cross-encoder score function with a (query, text) -> score cache shared across feedback rounds."""

    def __init__(self, score_fn: Callable[[str, Sequence[str]], List[float]]) -> None:
        self.score_fn = score_fn
        self.cache: Dict[Tuple[str, str], float] = {}
        self.calls = 0

    def seed(self, query: str, ranked: Sequence[Candidate], texts: Mapping[str, str]) -> None:
        for c in ranked:
            if c.rerank is not None:
                self.cache.setdefault((query, texts[c.id]), float(c.rerank))

    def __call__(self, query: str, texts: Sequence[str]) -> List[float]:
        todo = [t for t in dict.fromkeys(texts) if (query, t) not in self.cache]
        if todo:
            self.calls += 1
            for t, s in zip(todo, self.score_fn(query, todo)):
                self.cache[(query, t)] = float(s)
        return [self.cache[(query, t)] for t in texts]


class FeedbackHook:
    """Callable for DocScoutPipeline.answer(feedback_adjust=...). Build one per (query, document) call."""

    def __init__(self, query: str, votes: Mapping[str, int], q_vec: Optional[np.ndarray],
                 vec: Callable[[str], Optional[np.ndarray]], texts: Mapping[str, str], cfg: Mapping[str, Any],
                 scorer: Optional[CachedScorer] = None, memory: Optional[Mapping[str, float]] = None,
                 params: Optional[Mapping[str, float]] = None, top_n: Optional[int] = None) -> None:
        self.query, self.votes, self.q_vec, self.vec, self.texts, self.cfg = query, dict(votes), q_vec, vec, texts, cfg
        self.scorer, self.memory = scorer, dict(memory or {})
        p = dict(cfg["feedback"]["rocchio"])
        p.update(params or {})
        self.params = p
        self.top_n = int(top_n or cfg["retrieval"]["rerank_top_n"])
        self.deltas: Dict[str, float] = {}
        self.tags: Dict[str, str] = {}
        self.q_prime: Optional[np.ndarray] = None
        self.ranked: List[Candidate] = []   # the returned ranking (for "show next" without a new retrieval)

    @property
    def active(self) -> bool:
        return bool(self.votes) or any(self.memory.values())

    def __call__(self, ranked: List[Candidate], qtype: str, session_id: Optional[str]) -> List[Candidate]:
        if not self.active:
            self.ranked = list(ranked)
            return ranked
        fcfg = self.cfg["fusion"]
        cands = [c.copy() for c in ranked]
        use_rocchio = bool(self.votes) and self.q_vec is not None
        lam = float(self.params["lambda"]) if use_rocchio else 0.0
        if use_rocchio:
            pos = [v for v in (self.vec(c) for c, s in self.votes.items() if s > 0) if v is not None]
            neg = [v for v in (self.vec(c) for c, s in self.votes.items() if s < 0) if v is not None]
            qp = rocchio_query(self.q_vec, pos, neg, float(self.params["alpha"]), float(self.params["beta"]),
                               float(self.params["gamma"]))
            self.q_prime = qp
            if self.scorer is None:
                raise ValueError("Rocchio needs a cross-encoder scorer")
            self.scorer.seed(self.query, ranked, self.texts)  # round-1 CE scores: only new top-N items are scored
            for c in cands:
                v = self.vec(c.id)
                if "dense_bge" in c.raw and v is not None:
                    c.raw["dense_bge"] = float(v @ qp)
                c.norm = {}
                c.rerank = None
            normalize_candidates(cands, fcfg.get("normalization", "minmax"))
            bw = {b: float(w) for b, w in fcfg["branch_weights"][qtype].items()}
            cands = weighted_sum(cands, fcfg["retriever_weights"], bw)
            from docscout.rerank.cross_encoder import rerank_candidates

            cands = rerank_candidates(self.query, cands, self.texts, self.scorer, self.top_n)
        else:
            cands = sort_candidates(cands)
        head = [c for c in cands if c.rerank is not None]
        tail = [c for c in cands if c.rerank is None]
        ce_n = minmax({c.id: float(c.rerank) for c in head})
        cos_n: Dict[str, float] = {}
        if lam > 0:
            cos = {}
            for c in head:
                v = self.vec(c.id)
                cos[c.id] = float(v @ self.q_prime) if v is not None else 0.0
            cos_n = minmax(cos)
        for c in head:
            c.final = (1.0 - lam) * ce_n[c.id] + lam * cos_n.get(c.id, 0.0)
        for i, c in enumerate(tail):
            c.final = -1.0 - i
        for c in cands:
            prior = self.memory.get(c.id, 0.0)
            if prior:
                c.final += prior
                self.tags[c.id] = "memory"
            base = ce_n.get(c.id, c.final - prior)
            self.deltas[c.id] = c.final - base
            if use_rocchio and c.id in ce_n:
                self.tags[c.id] = "rocchio" + ("+memory" if prior else "")
        out = []
        for c in cands:
            s = self.votes.get(c.id, 0)
            if s < 0:
                continue  # down-voted: excluded from the next results
            if s > 0:
                c.final += PIN_OFFSET
                self.tags[c.id] = "pinned"
            out.append(c)
        self.ranked = sort_candidates(out)
        return self.ranked


def make_feedback_hook(pipe: Any, query: str, votes: Mapping[str, int], memory: Optional[Mapping[str, float]] = None,
                       params: Optional[Mapping[str, float]] = None, scorer: Optional[CachedScorer] = None,
                       q_vec: Optional[np.ndarray] = None) -> FeedbackHook:
    """FeedbackHook wired to a DocScoutPipeline (its BGE index vectors, query encoder and cross-encoder)."""
    if q_vec is None and (votes or memory):
        q_vec = pipe.retriever._encode("dense_bge", query)
    if scorer is None:
        scorer = CachedScorer(pipe.reranker.score)
    return FeedbackHook(query, votes, q_vec, component_vector_lookup(pipe.retriever), pipe.texts, pipe.cfg,
                        scorer=scorer, memory=memory, params=params)
