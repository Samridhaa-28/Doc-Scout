"""Common interface for all indexes: build / save / load / search, with exact doc_id filtering.

Every index keeps, per row, the component id and its doc_id, plus `doc_rows[doc_id]` (row
numbers of that document) so a doc-filtered search only touches that document's rows.
"""
from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from docscout.schema import Component

SearchResult = List[Tuple[str, float]]  # (component_id, score), best first


def file_fingerprint(path: Path) -> Dict[str, Any]:
    """Fingerprint of components.jsonl: line count + sha256 of its bytes."""
    h = hashlib.sha256()
    lines = 0
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            lines += chunk.count(b"\n")
    return {"lines": lines, "sha256": h.hexdigest()}


class BaseIndex(ABC):
    """Abstract index over a list of Components."""

    kind: str = "base"

    def __init__(self) -> None:
        self.ids: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_rows: Dict[str, np.ndarray] = {}
        self.info: Dict[str, Any] = {}

    # -- bookkeeping shared by all indexes
    def _set_rows(self, components: Sequence[Component]) -> None:
        self.ids = [c.id for c in components]
        self.doc_ids = [c.doc_id for c in components]
        rows: Dict[str, List[int]] = {}
        for i, d in enumerate(self.doc_ids):
            rows.setdefault(d, []).append(i)
        self.doc_rows = {d: np.asarray(r, dtype=np.int64) for d, r in rows.items()}

    def __len__(self) -> int:
        return len(self.ids)

    def _top(self, rows: np.ndarray, scores: np.ndarray, top_k: int) -> SearchResult:
        """Sort (rows, scores) descending; ties broken by row order for determinism."""
        if len(rows) == 0:
            return []
        order = np.lexsort((rows, -scores))[:top_k]
        return [(self.ids[int(rows[i])], float(scores[i])) for i in order]

    def _write_meta(self, out: Path, extra: Dict[str, Any]) -> None:
        meta = {"kind": self.kind, "n": len(self.ids), "ids": self.ids, "doc_ids": self.doc_ids, **self.info, **extra}
        (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def _read_meta(self, src: Path) -> Dict[str, Any]:
        meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        self.ids, self.doc_ids = meta.pop("ids"), meta.pop("doc_ids")
        rows: Dict[str, List[int]] = {}
        for i, d in enumerate(self.doc_ids):
            rows.setdefault(d, []).append(i)
        self.doc_rows = {d: np.asarray(r, dtype=np.int64) for d, r in rows.items()}
        self.info = meta
        return meta

    # -- interface
    @abstractmethod
    def build(self, components: Sequence[Component]) -> "BaseIndex":
        """Index the components (their text_form, or image for CLIP)."""

    @abstractmethod
    def save(self, out_dir: Path) -> None:
        """Persist to out_dir (meta.json + index files)."""

    @classmethod
    @abstractmethod
    def load(cls, src_dir: Path, **kwargs: Any) -> "BaseIndex":
        """Load a saved index without rebuilding."""

    @abstractmethod
    def search(self, query: str, top_k: int = 10, doc_id: Optional[str] = None) -> SearchResult:
        """Return [(component_id, score)] best first; with doc_id, only that document's components."""


class Timer:
    """Context manager measuring wall time in seconds (`.secs`)."""

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.secs = time.perf_counter() - self.t0
