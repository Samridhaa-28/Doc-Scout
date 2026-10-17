"""Shared data model: Component, Evidence, QueryResult, plus JSONL helpers."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

COMPONENT_TYPES: Tuple[str, ...] = ("paragraph", "heading", "table", "figure", "caption")
QUERY_TYPES: Tuple[str, ...] = ("textual", "numerical", "visual")

BBox = Tuple[float, float, float, float]  # (x0, y0, x1, y1), page pixels, top-left origin
Cell = Tuple[str, str, Any]  # (row_label, column, value)


def make_component_id(doc_id: str, page: int, ctype: str, n: int) -> str:
    """Build a component id in the format '{doc_id}_p{page}_{type}_{n}'."""
    return f"{doc_id}_p{page}_{ctype}_{n}"


@dataclass
class Component:
    """One retrievable unit of a document (paragraph, heading, table, figure or caption)."""

    id: str
    doc_id: str
    page: int  # 1-based
    type: str  # one of COMPONENT_TYPES
    text: str = ""
    text_form: str = ""  # serialized text used for indexing
    bbox: BBox = (0.0, 0.0, 0.0, 0.0)
    page_width: float = 0.0
    page_height: float = 0.0
    order: int = 0  # reading order within the document
    section_id: Optional[str] = None
    table_path: Optional[str] = None  # parquet path for tables
    image_path: Optional[str] = None  # png crop for figures
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in COMPONENT_TYPES:
            raise ValueError(f"unknown component type: {self.type!r}")
        self.bbox = tuple(float(v) for v in self.bbox)  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Component":
        d = dict(d)
        d["bbox"] = tuple(d.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        d["meta"] = dict(d.get("meta") or {})
        return cls(**d)


@dataclass
class Evidence:
    """A retrieved component with per-stage scores, graph context and explanation."""

    component: Component
    scores: Dict[str, float] = field(default_factory=dict)  # e.g. bm25, dense, fused, rerank
    context: List[Component] = field(default_factory=list)  # graph-expanded neighbours
    localized_cells: List[Cell] = field(default_factory=list)
    why: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)  # e.g. graph_path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component.to_dict(),
            "scores": dict(self.scores),
            "context": [c.to_dict() for c in self.context],
            "localized_cells": [list(c) for c in self.localized_cells],
            "why": self.why,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Evidence":
        return cls(
            component=Component.from_dict(d["component"]),
            scores={k: float(v) for k, v in (d.get("scores") or {}).items()},
            context=[Component.from_dict(c) for c in d.get("context") or []],
            localized_cells=[tuple(c) for c in d.get("localized_cells") or []],  # type: ignore[misc]
            why=d.get("why", ""),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class QueryResult:
    """Answer to one query plus the ranked evidence that supports it."""

    query: str
    query_type: str
    answer: str = ""
    evidence: List[Evidence] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)  # e.g. query_id, doc_id, latency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "answer": self.answer,
            "evidence": [e.to_dict() for e in self.evidence],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QueryResult":
        return cls(
            query=d["query"],
            query_type=d["query_type"],
            answer=d.get("answer", ""),
            evidence=[Evidence.from_dict(e) for e in d.get("evidence") or []],
            meta=dict(d.get("meta") or {}),
        )


# ---------------------------------------------------------------- JSONL helpers

def write_jsonl(path: Path, records: Iterable[Dict[str, Any]], append: bool = False) -> int:
    """Write dicts to a JSONL file (UTF-8). Returns the number of lines written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield dicts from a JSONL file, skipping blank lines."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a whole JSONL file into a list of dicts."""
    return list(iter_jsonl(path))


def write_components(path: Path, components: Iterable[Component], append: bool = False) -> int:
    """Write Components to JSONL."""
    return write_jsonl(path, (c.to_dict() for c in components), append=append)


def read_components(path: Path) -> List[Component]:
    """Read Components from JSONL."""
    return [Component.from_dict(d) for d in iter_jsonl(path)]
