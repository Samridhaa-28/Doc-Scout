"""Provenance: the per-stage scores of an evidence item and its one-line "why retrieved" explanation.

why_line format (fields separated by " | "):
    {query type} ({branch} {branch weight})
    {retriever} {raw}->{normalized}, ... , fused {fused}, CE {cross-encoder}[, fb {feedback delta}]
    terms: {query BM25 tokens found in the component's text_form, or "-"}
    {graph path}
    bbox {x0},{y0},{x1},{y1}
Example: "numerical (table 0.6) | bm25 11.95->1.00, bge 0.52->1.00, fused 0.600, CE 8.51 | terms: trrf, session |
          Page 9 -> Table 0 | bbox 90,297,520,610"
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from docscout.indexing.bm25_index import tokenize
from docscout.retrieval.retrievers import Candidate

RETRIEVER_LABEL = {"bm25": "bm25", "dense_bge": "bge", "dense_minilm": "minilm", "clip": "clip"}
RETRIEVER_ORDER = ["bm25", "dense_bge", "dense_minilm", "clip"]


def matched_terms(query: str, text: str) -> List[str]:
    """Query BM25 tokens (query order, no duplicates) that also occur in the component text."""
    toks = set(tokenize(text))
    return [t for t in dict.fromkeys(tokenize(query)) if t in toks]


def flat_scores(c: Candidate, feedback_delta: float = 0.0) -> Dict[str, float]:
    """{bm25_raw, bm25_norm, dense_bge_raw, ..., fused, rerank, feedback, final} (only scores that exist)."""
    out: Dict[str, float] = {}
    for r in RETRIEVER_ORDER:
        if r in c.raw:
            out[f"{r}_raw"] = float(c.raw[r])
        if r in c.norm:
            out[f"{r}_norm"] = float(c.norm[r])
    for k in ("fused", "rrf", "rerank"):
        v = getattr(c, k)
        if v is not None:
            out[k] = float(v)
    if feedback_delta:
        out["feedback"] = float(feedback_delta)
    out["final"] = float(c.final)
    return out


def _num(v: float) -> str:
    return f"{v:.2f}" if abs(v) < 1000 else f"{v:.0f}"


def why_line(c: Candidate, qtype: str, branch_weights: Mapping[str, float], terms: Sequence[str], path: str,
             bbox: Optional[Sequence[float]] = None, feedback_delta: float = 0.0) -> str:
    """One readable line explaining why the candidate was retrieved (format in the module docstring)."""
    parts = [f"{qtype} ({c.branch} {float(branch_weights.get(c.branch, 0.0)):g})"]
    sc = [f"{RETRIEVER_LABEL.get(r, r)} {_num(c.raw[r])}->{c.norm.get(r, 0.0):.2f}"
          for r in RETRIEVER_ORDER if r in c.raw]
    if c.fused is not None:
        sc.append(f"fused {c.fused:.3f}")
    sc.append(f"CE {c.rerank:.2f}" if c.rerank is not None else "CE -")
    if feedback_delta:
        sc.append(f"fb {feedback_delta:+.2f}")
    parts.append(", ".join(sc))
    parts.append("terms: " + (", ".join(terms) if terms else "-"))
    if path:
        parts.append(path)
    if bbox is not None and any(bbox):
        parts.append("bbox " + ",".join(f"{v:.0f}" for v in bbox))
    return " | ".join(parts)


def provenance_record(c: Candidate, comp: Any, qtype: str, branch_weights: Mapping[str, float], query: str,
                      path: str, feedback_delta: float = 0.0) -> Dict[str, Any]:
    """Everything the UI / API needs to show where an evidence item came from and why."""
    terms = matched_terms(query, comp.text_form)
    return {
        "page": int(comp.page), "type": comp.type, "bbox": list(comp.bbox),
        "page_size": [comp.page_width, comp.page_height], "graph_path": path, "branch": c.branch,
        "branch_weight": float(branch_weights.get(c.branch, 0.0)), "matched_terms": terms,
        "scores": flat_scores(c, feedback_delta),
        "why": why_line(c, qtype, branch_weights, terms, path, comp.bbox, feedback_delta),
    }
