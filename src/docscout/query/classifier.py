"""Query-type classifier: textual / numerical / visual. Rules first, the LLM decides when the rules are unsure.

Decision flow:
  1. Word-boundary regexes. NUMERICAL: digits, how much/many, total, percent/%, amount, number of, date, year,
     cost, price. VISUAL: chart, graph, figure, diagram, image, logo, photo, picture.
  2. Exactly one group matches -> that label, confidence = classifier.rule_confidence, source "rules".
  3. Both or neither match -> rules are "unsure": the LLM returns {"type", "confidence"} (source "llm").
     If the LLM is off or fails, the fallback label classifier.unsure_fallback is used (source "rules").

The label only sets branch weights (soft routing); see weights_for().
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from docscout.config import get_config

LABELS = ("textual", "numerical", "visual")
LABEL_TO_BRANCH = {"textual": "text", "numerical": "table", "visual": "figure"}

NUMERICAL_PATTERNS = [
    r"\d",
    r"\bhow\s+(?:much|many)\b",
    r"\btotals?\b",
    r"\bpercent(?:age)?s?\b",
    r"%",
    r"\bamounts?\b",
    r"\bnumber\s+of\b",
    r"\bdates?\b",
    r"\byears?\b",
    r"\bcosts?\b",
    r"\bprices?\b",
]
VISUAL_PATTERNS = [
    r"\bcharts?\b",
    r"\bgraphs?\b",
    r"\bfigures?\b",
    r"\bdiagrams?\b",
    r"\bimages?\b",
    r"\blogos?\b",
    r"\bphotos?(?:graphs?)?\b",
    r"\bpictures?\b",
]
_NUM_RE = [re.compile(p, re.IGNORECASE) for p in NUMERICAL_PATTERNS]
_VIS_RE = [re.compile(p, re.IGNORECASE) for p in VISUAL_PATTERNS]

SYSTEM_PROMPT = "You classify questions asked about a document. Reply with a JSON object only."
LLM_PROMPT = """Classify the question into exactly one type:
- textual: the answer is words from the running text (a name, title, place, organisation, statement).
- numerical: the answer is a number, amount, date, time, percentage or a value read from a table or form.
- visual: the answer depends on a chart, graph, figure, diagram, photo, logo or other image.

Question: {query}

Return JSON: {{"type": "textual" | "numerical" | "visual", "confidence": <number between 0 and 1>}}"""


def rule_matches(query: str) -> Dict[str, List[str]]:
    """Matched pattern texts per group: {"numerical": [...], "visual": [...]}."""
    return {
        "numerical": [m.group(0) for r in _NUM_RE for m in [r.search(query)] if m],
        "visual": [m.group(0) for r in _VIS_RE for m in [r.search(query)] if m],
    }


def rule_classify(query: str, confidence: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Rules only. Returns a result dict, or None when the rules are unsure (both or neither group match)."""
    conf = float(get_config()["classifier"]["rule_confidence"]) if confidence is None else confidence
    m = rule_matches(query)
    num, vis = bool(m["numerical"]), bool(m["visual"])
    if num == vis:
        return None
    label = "numerical" if num else "visual"
    return {"type": label, "confidence": conf, "source": "rules", "matched": m[label]}


def llm_classify(query: str, client: Any) -> Dict[str, Any]:
    """Ask the LLM (JSON mode). Raises on an invalid reply."""
    data = client.complete_json(LLM_PROMPT.format(query=query), system=SYSTEM_PROMPT)
    label = str(data.get("type", "")).strip().lower()
    if label not in LABELS:
        raise ValueError(f"LLM returned an unknown type: {data!r}")
    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return {"type": label, "confidence": min(1.0, max(0.0, conf)), "source": "llm"}


def classify(query: str, client: Any = None, use_llm: Optional[bool] = None) -> Dict[str, Any]:
    """classify(query) -> {"type": textual|numerical|visual, "confidence": float, "source": "rules"|"llm"}.

    `client` is an LLMClient-like object (complete_json); if None and the LLM is enabled, get_client() is used.
    """
    cfg = get_config()["classifier"]
    use_llm = cfg.get("use_llm", True) if use_llm is None else use_llm
    res = rule_classify(query, float(cfg["rule_confidence"]))
    if res is not None:
        return res
    fallback = {"type": cfg.get("unsure_fallback", "textual"), "confidence": 0.5, "source": "rules",
                "unsure": True}
    if not use_llm:
        return fallback
    if client is None:
        from docscout.reasoning.llm_client import get_client

        client = get_client()
    out = llm_classify(query, client)
    out["unsure_rules"] = True
    return out


def weights_for(qtype: str, cfg: Optional[dict] = None) -> Dict[str, float]:
    """Branch weights {text, table, figure} for a query type, from config fusion.branch_weights."""
    cfg = cfg or get_config()
    bw = cfg["fusion"]["branch_weights"]
    if qtype not in bw:
        raise KeyError(f"no branch weights for query type {qtype!r}")
    return {b: float(w) for b, w in bw[qtype].items()}
