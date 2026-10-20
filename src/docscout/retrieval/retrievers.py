"""Score EVERY component of one document with every retriever of its branch.

Branch retrievers: text / table / figure -> BM25 + BGE; figure additionally CLIP (figure crops).
Each query is encoded once per model (BGE once for all three branches, CLIP once).
A component with no BM25 term match gets BM25 = 0 (it is kept, not dropped).

The result is a list of Candidate objects carrying the raw score per retriever; normalize.py, fusion.py and
rerank fill in the later stage scores, which stay on the candidate for provenance (P6).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from docscout.config import get_config
from docscout.indexing.bm25_index import tokenize

BRANCH_RETRIEVERS: Dict[str, List[str]] = {
    "text": ["bm25", "dense_bge"],
    "table": ["bm25", "dense_bge"],
    "figure": ["bm25", "dense_bge", "clip"],
}
# retriever name -> key in config fusion.retriever_weights
RETRIEVER_WEIGHT_KEY = {"bm25": "bm25", "dense_bge": "dense", "dense_minilm": "dense", "clip": "clip"}


@dataclass
class Candidate:
    """One component of the searched document, with every stage score."""

    id: str
    doc_id: str
    page: int
    type: str
    branch: str
    raw: Dict[str, float] = field(default_factory=dict)    # retriever -> raw score
    norm: Dict[str, float] = field(default_factory=dict)   # retriever -> normalized score
    fused: Optional[float] = None                          # weighted-sum score
    rrf: Optional[float] = None                            # reciprocal-rank-fusion score
    rerank: Optional[float] = None                         # cross-encoder score (top-N only)
    final: float = 0.0                                     # score the ranking is sorted by

    def stage_scores(self) -> Dict[str, Any]:
        """All stage scores, for provenance."""
        return {"raw": dict(self.raw), "norm": dict(self.norm), "fused": self.fused, "rrf": self.rrf,
                "rerank": self.rerank, "final": self.final}

    def copy(self) -> "Candidate":
        return Candidate(self.id, self.doc_id, self.page, self.type, self.branch, dict(self.raw), dict(self.norm),
                         self.fused, self.rrf, self.rerank, self.final)


def branch_of_type(cfg: Optional[dict] = None) -> Dict[str, str]:
    """Component type -> branch, from config `branches`."""
    cfg = cfg or get_config()
    return {t: b for b, types in cfg["branches"].items() for t in types}


class HybridRetriever:
    """Loads the branch indexes once and scores all components of a document.

    `load` is a function (branch, retriever) -> index or None (default: indexing.build_all.load_index);
    `comp_meta` maps component id -> object with doc_id / page / type (default: components.jsonl).
    """

    def __init__(self, load: Optional[Callable[[str, str], Any]] = None,
                 comp_meta: Optional[Mapping[str, Any]] = None, branch_retrievers: Optional[Dict[str, List[str]]] = None,
                 cfg: Optional[dict] = None) -> None:
        self.cfg = cfg or get_config()
        if load is None:
            from docscout.indexing.build_all import load_index

            def load(b: str, r: str) -> Any:  # noqa: E306
                return load_index(b, r, warn=False)
        if comp_meta is None:
            from docscout.schema import Component, iter_jsonl

            comp_meta = {r["id"]: Component.from_dict(r) for r in iter_jsonl(self.cfg["paths"]["components_file"])}
        self.comp_meta = comp_meta
        self.branch_retrievers = branch_retrievers or BRANCH_RETRIEVERS
        self.type_branch = branch_of_type(self.cfg)
        self.indexes: Dict[str, Dict[str, Any]] = {}
        for b, rets in self.branch_retrievers.items():
            for r in rets:
                idx = load(b, r)
                if idx is not None:
                    self.indexes.setdefault(b, {})[r] = idx

    def _encode(self, retriever: str, query: str) -> Optional[np.ndarray]:
        for rets in self.indexes.values():
            if retriever in rets:
                return rets[retriever].encode_query(query)
        return None

    @staticmethod
    def _bm25_scores(idx: Any, tokens: List[str], rows: np.ndarray) -> np.ndarray:
        if not tokens or idx.bm25 is None:
            return np.zeros(len(rows), dtype=np.float64)
        return np.asarray(idx.bm25.get_batch_scores(tokens, rows.tolist()), dtype=np.float64)

    @staticmethod
    def _dense_scores(idx: Any, q: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return (idx.emb[rows] @ q).astype(np.float64)

    def retrieve(self, query: str, doc_id: str, branches: Optional[Iterable[str]] = None) -> List[Candidate]:
        """Every component of `doc_id` in the given branches (default all), with raw scores per retriever."""
        branches = list(branches) if branches is not None else list(self.indexes)
        needed = {r for b in branches for r in self.indexes.get(b, {})}
        tokens = tokenize(query)
        vecs = {r: self._encode(r, query) for r in needed if r != "bm25"}  # one encoding per model
        cands: Dict[str, Candidate] = {}
        for b in branches:
            for r, idx in self.indexes.get(b, {}).items():
                rows = idx.doc_rows.get(doc_id)
                if rows is None or len(rows) == 0:
                    continue
                scores = self._bm25_scores(idx, tokens, rows) if r == "bm25" else self._dense_scores(idx, vecs[r], rows)
                for row, s in zip(rows.tolist(), scores.tolist()):
                    cid = idx.ids[row]
                    c = cands.get(cid)
                    if c is None:
                        m = self.comp_meta[cid]
                        c = cands[cid] = Candidate(cid, m.doc_id, int(m.page), m.type, b)
                    c.raw[r] = float(s)
        return sorted(cands.values(), key=lambda c: c.id)


def lists_by_branch_retriever(cands: Sequence[Candidate]) -> Dict[tuple, List[Candidate]]:
    """{(branch, retriever): [candidates that have that raw score]}: the units normalization and RRF work on."""
    out: Dict[tuple, List[Candidate]] = {}
    for c in cands:
        for r in c.raw:
            out.setdefault((c.branch, r), []).append(c)
    return out
