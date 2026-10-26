"""Cross-session memory: a small capped prior per component from votes on similar past questions.

A past vote counts for the current question only if it is on the same document, has the same query type and its
question has BGE cosine >= min_similarity to the current one. A component needs >= min_votes such votes; its prior is
step * sum(vote * 0.5 ** (age_days / half_life_days)), clipped to [-cap, +cap], and is added to `final`
after rerank (the [0, 1] normalized scale used whenever feedback is active, see rocchio.py).
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


def question_vector(retriever: Any, question: str, name: str = "dense_bge") -> np.ndarray:
    """Plain BGE embedding of a question (no retrieval instruction: question-question similarity is symmetric,
    and the shared instruction prefix would inflate every cosine). Stored with each vote."""
    for rets in retriever.indexes.values():
        idx = rets.get(name)
        if idx is not None:
            return np.asarray(idx.model.encode([question], normalize_embeddings=True, convert_to_numpy=True,
                                               show_progress_bar=False)[0], dtype=np.float32)
    raise ValueError(f"no {name} index loaded")


def memory_prior(store: Any, doc_id: str, qtype: str, q_vec: np.ndarray, cfg: Mapping[str, Any],
                 now: Optional[float] = None, exclude_session: Optional[str] = None) -> Dict[str, float]:
    """{component_id: prior} for the current question (empty when no component passes the gates)."""
    fb = cfg["feedback"]
    step, cap = float(fb["step"]), float(fb["cap"])
    min_votes, min_sim = int(fb["min_votes"]), float(fb["min_similarity"])
    half_life_s = float(fb["half_life_days"]) * 86400.0
    now = time.time() if now is None else now
    q = np.asarray(q_vec, dtype=np.float32)
    q = q / (np.linalg.norm(q) or 1.0)
    weights: Dict[str, List[float]] = defaultdict(list)
    for v in store.doc_votes(doc_id, qtype, exclude_session=exclude_session):
        if v.query_vec is None or v.component_id is None:
            continue
        p = v.query_vec / (np.linalg.norm(v.query_vec) or 1.0)
        if float(p @ q) < min_sim:
            continue
        age = max(0.0, now - v.timestamp)
        weights[v.component_id].append(v.vote * 0.5 ** (age / half_life_s))
    return {cid: float(np.clip(step * sum(ws), -cap, cap)) for cid, ws in weights.items()
            if len(ws) >= min_votes and sum(ws) != 0}
