"""BM25 (rank_bm25 BM25Okapi) over Component.text_form."""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from docscout.indexing.base import BaseIndex, SearchResult
from docscout.schema import Component

STOPWORDS = frozenset("""
a an the and or but if of at by for with about against between into through during before after above below
to from up down in out on off over under again further then once here there when where why how all any both
each few more most other some such no nor not only own same so than too very s t can will just don should now
is are was were be been being have has had having do does did doing i me my we our you your he him his she her
it its they them their what which who whom this that these those am as until while
""".split())

# numbers with internal separators ("10,700", "26.9", "1.2.3") or plain alphanumeric runs
_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)+|[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase; split on punctuation; keep numbers ('2023', '26.9', '10,700' -> '10700'); drop stopwords."""
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok[0].isdigit() and "," in tok:
            tok = tok.replace(",", "")
        if tok not in STOPWORDS:
            out.append(tok)
    return out


class BM25Index(BaseIndex):
    """Okapi BM25 over tokenized text_form; doc-filtered search scores only that document's rows."""

    kind = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__()
        self.k1, self.b = k1, b
        self.bm25: Optional[BM25Okapi] = None

    def build(self, components: Sequence[Component]) -> "BM25Index":
        self._set_rows(components)
        corpus = [tokenize(c.text_form) or ["__empty__"] for c in components]
        self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        self.info = {"k1": self.k1, "b": self.b}
        return self

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "bm25.pkl").open("wb") as f:
            pickle.dump(self.bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._write_meta(out_dir, {})

    @classmethod
    def load(cls, src_dir: Path, **kwargs: Any) -> "BM25Index":
        idx = cls()
        meta = idx._read_meta(src_dir)
        idx.k1, idx.b = meta.get("k1", 1.5), meta.get("b", 0.75)
        with (src_dir / "bm25.pkl").open("rb") as f:
            idx.bm25 = pickle.load(f)
        return idx

    def search(self, query: str, top_k: int = 10, doc_id: Optional[str] = None) -> SearchResult:
        q = tokenize(query)
        if not q or self.bm25 is None:
            return []
        if doc_id is not None:
            rows = self.doc_rows.get(doc_id)
            if rows is None:
                return []
            scores = np.asarray(self.bm25.get_batch_scores(q, rows.tolist()), dtype=np.float64)
        else:
            rows = np.arange(len(self.ids))
            scores = np.asarray(self.bm25.get_scores(q), dtype=np.float64)
        keep = scores > 0  # no query term matched -> not a result
        return self._top(rows[keep], scores[keep], top_k)
