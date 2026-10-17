"""Dense retrieval: sentence-transformers embeddings (normalized) in a FAISS IndexFlatIP.

The BGE query instruction is prepended to QUERIES only (`query_text`); documents are encoded
as their plain text_form (`doc_text`).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Sequence

import faiss
import numpy as np

from docscout.config import get_config
from docscout.indexing.base import BaseIndex, SearchResult
from docscout.schema import Component


@lru_cache(maxsize=4)
def load_st_model(name: str, device: str = "cpu"):
    """Load (once per process) a sentence-transformers model: HF cache first (no network), else download."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(name, device=device, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached yet (or incomplete): fetch from the Hub
        return SentenceTransformer(name, device=device)


def query_instruction_for(model_name: str, cfg: Optional[dict] = None) -> str:
    """The query prefix for a model: the BGE instruction for the configured BGE model, else ''."""
    cfg = cfg or get_config()
    return cfg["models"]["bge_query_instruction"] if model_name == cfg["models"]["bge"] else ""


class DenseIndex(BaseIndex):
    """Bi-encoder index. `model` can be injected (tests); otherwise loaded by name."""

    kind = "dense"

    def __init__(self, model_name: str, query_instruction: Optional[str] = None, model: Any = None,
                 batch_size: int = 32, device: str = "cpu") -> None:
        super().__init__()
        self.model_name = model_name
        self.query_instruction = query_instruction_for(model_name) if query_instruction is None else query_instruction
        self._model = model
        self.batch_size, self.device = batch_size, device
        self.emb: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_st_model(self.model_name, self.device)
        return self._model

    # exact strings that get encoded
    def doc_text(self, c: Component) -> str:
        return c.text_form

    def query_text(self, query: str) -> str:
        return f"{self.query_instruction}{query}"

    def _encode(self, texts: List[str], progress: bool) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=progress,
                                normalize_embeddings=True, convert_to_numpy=True)
        return np.ascontiguousarray(emb, dtype=np.float32)

    def build(self, components: Sequence[Component], progress: bool = True) -> "DenseIndex":
        self._set_rows(components)
        self.emb = self._encode([self.doc_text(c) for c in components], progress)
        self.faiss_index = faiss.IndexFlatIP(self.emb.shape[1])
        self.faiss_index.add(self.emb)
        self.info = {"model": self.model_name, "query_instruction": self.query_instruction, "dim": int(self.emb.shape[1])}
        return self

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "embeddings.npy", self.emb)
        faiss.write_index(self.faiss_index, str(out_dir / "faiss.index"))
        self._write_meta(out_dir, {})

    @classmethod
    def load(cls, src_dir: Path, model: Any = None, **kwargs: Any) -> "DenseIndex":
        meta = json.loads((src_dir / "meta.json").read_text(encoding="utf-8"))
        idx = cls(meta["model"], query_instruction=meta["query_instruction"], model=model)
        idx._read_meta(src_dir)
        idx.emb = np.load(src_dir / "embeddings.npy")
        idx.faiss_index = faiss.read_index(str(src_dir / "faiss.index"))
        return idx

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([self.query_text(query)], progress=False)[0]

    def search(self, query: str, top_k: int = 10, doc_id: Optional[str] = None) -> SearchResult:
        if not self.ids:
            return []
        q = self.encode_query(query)
        return self.search_vector(q, top_k, doc_id)

    def search_vector(self, q: np.ndarray, top_k: int = 10, doc_id: Optional[str] = None) -> SearchResult:
        """Search with an already-encoded, normalized query vector."""
        if doc_id is not None:
            rows = self.doc_rows.get(doc_id)
            if rows is None:
                return []
            scores = self.emb[rows] @ q  # exact dot products over this document's rows only
            return self._top(rows, scores.astype(np.float64), top_k)
        k = min(top_k, len(self.ids))
        scores, rows = self.faiss_index.search(q.reshape(1, -1).astype(np.float32), k)
        return [(self.ids[int(r)], float(s)) for r, s in zip(rows[0], scores[0]) if r >= 0]
