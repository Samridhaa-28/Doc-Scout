"""Grounded answer generation over retrieved evidence, with output validation.

answer_question(query, qtype, evidence, client) -> AnswerResult
  1. build_prompt (prompts.py): question + type + numbered evidence [E1].. (never the whole document)
  2. client.complete_json -> {"answer", "citations"}
  3. validate_answer:
     - citations are normalized ("[E1]", "e1", "E1, E2" -> "E1", "E2"); ids that were not in the prompt are
       dropped and counted (invalid_citations)
     - an answer equal to the not-found sentence (any case / trailing dot) or empty -> not found
     - an answer with NO valid citation -> treated as not found (no_valid_citation = True); the model's text
       is kept in raw_answer
  4. An LLM failure (after retries) or a non-JSON reply -> not found with `error` set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from docscout.config import get_config
from docscout.reasoning.prompts import PromptInfo, build_prompt
from docscout.schema import Evidence

_ID_RE = re.compile(r"E\d+(?:\.ctx\d+)?", re.IGNORECASE)


@dataclass
class AnswerResult:
    answer: str
    citations: List[str] = field(default_factory=list)
    not_found: bool = False
    raw_answer: str = ""
    invalid_citations: List[str] = field(default_factory=list)
    no_valid_citation: bool = False
    error: Optional[str] = None
    prompt_info: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", " ".join(str(s).lower().split())).strip()


def is_not_found(answer: str, not_found: str) -> bool:
    """Empty, or the not-found sentence (any case, trailing punctuation)."""
    a = _norm_text(answer)
    return not a or a.startswith(_norm_text(not_found))


def normalize_citations(raw: Any) -> List[str]:
    """Citation ids from a list or string, normalized to 'E1' / 'E1.ctx2', order kept, no duplicates."""
    items = raw if isinstance(raw, list) else [raw]
    out: List[str] = []
    for it in items:
        for m in _ID_RE.findall(str(it)):
            cid = "E" + m[1:].lower()
            if cid not in out:
                out.append(cid)
    return out


def _raw_tokens(raw: Any) -> List[str]:
    items = raw if isinstance(raw, list) else ([] if raw is None else [raw])
    return [str(x).strip() for x in items if str(x).strip()]


def validate_answer(data: Dict[str, Any], valid_ids: Sequence[str], not_found: str) -> AnswerResult:
    """Apply the validation rules in the module docstring to a parsed LLM reply."""
    raw_answer = " ".join(str(data.get("answer", "") or "").split())
    cited = normalize_citations(data.get("citations", []))
    valid = [c for c in cited if c in set(valid_ids)]
    invalid = [c for c in cited if c not in set(valid_ids)]
    # strings that contain no id-like text at all are invalid too ("page 3", "table")
    invalid += [t for t in _raw_tokens(data.get("citations")) if not _ID_RE.search(t)]
    if is_not_found(raw_answer, not_found):
        return AnswerResult(not_found, [], True, raw_answer, invalid)
    if not valid:
        return AnswerResult(not_found, [], True, raw_answer, invalid, no_valid_citation=True)
    return AnswerResult(raw_answer, valid, False, raw_answer, invalid)


def answer_question(query: str, qtype: str, evidence: Sequence[Evidence], client: Any,
                    cfg: Optional[dict] = None) -> AnswerResult:
    """Build the prompt, call the LLM (JSON mode, cached), validate. Never raises on LLM failure."""
    cfg = cfg or get_config()
    not_found = cfg["answer"]["not_found"]
    system, user, info = build_prompt(query, qtype, evidence, cfg)
    if not evidence:
        res = AnswerResult(not_found, [], True, error="no evidence")
        res.prompt_info = info.as_dict()
        return res
    try:
        data = client.complete_json(user, system=system)
        res = validate_answer(data, info.ids, not_found)
    except Exception as e:  # noqa: BLE001 - LLMError, JSON errors: report as not found
        res = AnswerResult(not_found, [], True, error=f"{type(e).__name__}: {str(e)[:200]}")
    res.prompt_info = info.as_dict()
    return res


def evidence_for_citation(cid: str, evidence: Sequence[Evidence]) -> Optional[str]:
    """Component id behind an id like 'E2' or 'E2.ctx1' (None if out of range)."""
    m = re.fullmatch(r"E(\d+)(?:\.ctx(\d+))?", cid)
    if not m:
        return None
    i = int(m.group(1)) - 1
    if not 0 <= i < len(evidence):
        return None
    if m.group(2) is None:
        return evidence[i].component.id
    j = int(m.group(2)) - 1
    ctx = evidence[i].context
    return ctx[j].id if 0 <= j < len(ctx) else None


__all__ = ["AnswerResult", "PromptInfo", "answer_question", "validate_answer", "normalize_citations",
           "is_not_found", "evidence_for_citation"]
