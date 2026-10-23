"""Modality slot reservation: make sure the evidence shown to the LLM contains a table and a figure candidate.

The cross-encoder (ms-marco, trained on web prose) ranks serialized tables and OCR'd chart text low, so for
numerical / visual questions the best table or figure can fall out of the top-k after rerank. Slot reservation
keeps the reranked ORDER (and the full reranked list, so P5 ranking metrics are unchanged) and only changes which
k items become evidence:

  1. evidence = reranked[:k]
  2. for each modality in `modalities` (table, then figure): take the best candidate of that type in the
     PRE-RERANK hybrid ranking (highest c.fused; ties by id). If it is not already in the evidence, it replaces the
     lowest-ranked PROSE item (paragraph / heading / caption) at evidence position >= 2 - the reranker's top-1 is
     never replaced, and reserved items are never replaced by later ones.
  3. evidence is returned in reranked order (so a reserved item usually comes last); k stays the same.
     With k = 1 nothing changes.

Variants (`slot_modalities`):
  "conditional" (proposed default): numerical -> table slot only; visual -> figure slot only; textual -> none
  "both": table and figure slots for every query type
  "off": no reservation
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from docscout.retrieval.retrievers import Candidate

PROSE_TYPES = ("paragraph", "heading", "caption")
DEFAULT_MODALITIES = ("table", "figure")
VARIANTS = ("off", "conditional", "both")
CONDITIONAL = {"numerical": ("table",), "visual": ("figure",), "textual": ()}


def slot_modalities(variant: str, qtype: str) -> Tuple[str, ...]:
    """Modalities to reserve for a query type under a variant (see module docstring)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown slot variant {variant!r}; choose from {VARIANTS}")
    if variant == "off":
        return ()
    if variant == "both":
        return DEFAULT_MODALITIES
    return CONDITIONAL.get(qtype, ())


def best_by_fused(cands: Sequence[Candidate], ctype: str) -> Candidate | None:
    """Best candidate of a component type in the pre-rerank (fused) ranking, or None."""
    pool = [c for c in cands if c.type == ctype and c.fused is not None]
    if not pool:
        return None
    return min(pool, key=lambda c: (-float(c.fused), c.id))


def reserve_modality_slots(reranked: Sequence[Candidate], k: int,
                           modalities: Sequence[str] = DEFAULT_MODALITIES) -> Tuple[List[Candidate], Dict[str, str]]:
    """(evidence candidates of length min(k, len(reranked)), {reserved component id: modality}).

    Does not modify `reranked` or any candidate."""
    evidence = list(reranked[:k])
    if k < 2:
        return evidence, {}
    pos = {c.id: i for i, c in enumerate(reranked)}
    reserved: Dict[str, str] = {}
    for m in modalities:
        best = best_by_fused(reranked, m)
        if best is None or any(c.id == best.id for c in evidence):
            continue
        # lowest-ranked prose item at evidence position >= 2 that is not itself reserved
        victims = [i for i, c in enumerate(evidence) if i >= 1 and c.type in PROSE_TYPES and c.id not in reserved]
        if not victims:
            continue
        evidence[victims[-1]] = best
        reserved[best.id] = m
        evidence.sort(key=lambda c: pos[c.id])
    return evidence, reserved
