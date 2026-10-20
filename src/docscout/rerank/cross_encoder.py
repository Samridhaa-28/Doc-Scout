"""Cross-encoder rerank of the top-N fused candidates on (query, text_form).

The top `rerank_top_n` (config retrieval.rerank_top_n) candidates are scored by
cross-encoder/ms-marco-MiniLM-L-6-v2 and re-sorted by that score (ties by component id); c.rerank keeps it.
Everything below the top N keeps its fused order, placed after the reranked block. c.final is set so the list is
strictly decreasing: head = CE score, tail = (lowest head CE score) - 1 - i.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, List, Mapping, Optional, Sequence

from docscout.config import get_config
from docscout.retrieval.retrievers import Candidate

ScoreFn = Callable[[str, Sequence[str]], List[float]]


@lru_cache(maxsize=2)
def load_cross_encoder(name: str, device: str = "cpu") -> Any:
    """Load (once per process) a sentence-transformers CrossEncoder: HF cache first, else download."""
    from sentence_transformers import CrossEncoder

    try:
        return CrossEncoder(name, device=device, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached yet
        return CrossEncoder(name, device=device)


def rerank_candidates(query: str, ranked: Sequence[Candidate], texts: Mapping[str, str], score_fn: ScoreFn,
                      top_n: int) -> List[Candidate]:
    """Rerank the first `top_n` of an already-fused ranking with score_fn(query, texts) -> scores."""
    head, tail = list(ranked[:top_n]), list(ranked[top_n:])
    if not head:
        return []
    scores = score_fn(query, [texts[c.id] for c in head])
    for c, s in zip(head, scores):
        c.rerank = float(s)
        c.final = float(s)
    head.sort(key=lambda c: (-c.final, c.id))
    floor = min(c.final for c in head)
    for i, c in enumerate(tail):
        c.rerank = None
        c.final = floor - 1.0 - i
    return head + tail


class CrossEncoderReranker:
    """Loads the cross-encoder once; batched scoring."""

    def __init__(self, model_name: Optional[str] = None, batch_size: Optional[int] = None,
                 top_n: Optional[int] = None, model: Any = None, cfg: Optional[dict] = None) -> None:
        cfg = cfg or get_config()
        self.model_name = model_name or cfg["models"]["cross_encoder"]
        self.batch_size = batch_size or int(cfg["models"].get("batch_size", 32))
        self.top_n = top_n or int(cfg["retrieval"]["rerank_top_n"])
        self.device = cfg["models"].get("device", "cpu")
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = load_cross_encoder(self.model_name, self.device)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        out = self.model.predict([(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False)
        return [float(x) for x in out]

    def rerank(self, query: str, ranked: Sequence[Candidate], texts: Mapping[str, str]) -> List[Candidate]:
        return rerank_candidates(query, ranked, texts, self.score, self.top_n)
