"""Prompt for grounded answering: the question, its type and numbered evidence only (never the whole document).

Evidence block format (one per item, in rank order):
    [E1] table | page 9
    Table: MONDAY, MAY 15 ...
    Localized cells:
    | row | column | value |
    ...
    [E1.ctx1] heading | page 9: MONDAY, MAY 15
Tables show the localized cells as a small markdown table when there are any; otherwise their first rows
(answer.max_table_rows). Text is cut at answer.max_item_chars (context items: answer.max_ctx_chars).

Size cap: the whole prompt (system + user) must stay within answer.max_prompt_tokens, estimated as
chars / answer.chars_per_token. Main items come first: if they do not fit, their character cap is shrunk
(x0.8 per step, down to 200), then the lowest-ranked items are dropped. Context items fill what is left,
in rank order. A final hard cut guarantees the estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from docscout.config import get_config
from docscout.localization.table_locator import cells_markdown
from docscout.schema import Evidence

SYSTEM_PROMPT = ("You answer questions about a document using ONLY the numbered evidence excerpts you are given. "
                 "Reply with a JSON object only.")

ANSWER_TEMPLATE = """Question: {query}
Question type: {qtype}

Evidence (excerpts retrieved from the document; you have not seen the rest of it):
{evidence}

Rules:
1. Use ONLY the evidence above. Do not use outside knowledge and do not guess.
2. Give a short extractive answer: copy the exact words or number from the evidence (a name, value, date or short phrase), not a full sentence and no explanation.
3. Cite the ids of the evidence you used, for example ["E1"] or ["E2.ctx1"].
4. If the evidence does not contain the answer, set "answer" to exactly "{not_found}" and "citations" to [].

Return JSON: {{"answer": "<short answer>", "citations": ["E1"]}}"""


@dataclass
class PromptInfo:
    """What went into the prompt: ids shown, size estimate, what was cut."""

    ids: List[str] = field(default_factory=list)            # every id shown: E1, E1.ctx1, E2, ...
    component_ids: List[str] = field(default_factory=list)  # the document components the LLM saw
    est_tokens: int = 0
    item_char_cap: int = 0
    dropped_items: int = 0
    dropped_ctx: int = 0
    hard_cut: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Rough token estimate (no tokenizer download): ceil(chars / chars_per_token)."""
    return int(-(-len(text) // chars_per_token))


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split()) if "\n|" not in str(text) else str(text).strip()
    return text if len(text) <= n else text[: max(0, n - 3)] + "..."


def evidence_body(e: Evidence, cap: int) -> str:
    """Text shown for one main evidence item (tables: title + cells, or title + first rows)."""
    c = e.component
    if c.type == "table":
        title = (c.text_form or "").split("\n", 1)[0]
        if e.localized_cells:
            body = f"{title}\nLocalized cells:\n{cells_markdown(e.localized_cells)}"
        elif e.meta.get("table_preview"):
            body = f"{title}\n{e.meta['table_preview']}"
        else:
            body = c.text_form
        return body if len(body) <= cap else body[: max(0, cap - 3)] + "..."
    text = c.text if c.type in ("paragraph", "heading", "caption") and c.text else c.text_form
    return _clip(text, cap)


def main_block(eid: str, e: Evidence, cap: int) -> str:
    return f"[{eid}] {e.component.type} | page {e.component.page}\n{evidence_body(e, cap)}"


def ctx_block(cid: str, comp: Any, cap: int) -> str:
    return f"[{cid}] {comp.type} | page {comp.page}: {_clip(comp.text or comp.text_form, cap)}"


def build_prompt(query: str, qtype: str, evidence: Sequence[Evidence], cfg: Optional[dict] = None
                 ) -> Tuple[str, str, PromptInfo]:
    """(system, user prompt, info). Evidence items are numbered E1..En; context items E{i}.ctx{j}."""
    acfg = (cfg or get_config())["answer"]
    max_tokens = int(acfg["max_prompt_tokens"])
    cpt = float(acfg.get("chars_per_token", 4.0))
    cap = int(acfg["max_item_chars"])
    ctx_cap = int(acfg["max_ctx_chars"])
    not_found = acfg["not_found"]

    fixed = SYSTEM_PROMPT + ANSWER_TEMPLATE.format(query=query, qtype=qtype, evidence="", not_found=not_found)
    budget = int(max_tokens * cpt) - len(fixed)
    info = PromptInfo()
    items = list(evidence)

    def mains(n_cap: int) -> List[str]:
        return [main_block(f"E{i}", e, n_cap) for i, e in enumerate(items, start=1)]

    blocks = mains(cap)
    while sum(len(b) + 1 for b in blocks) > budget and cap > 200:
        cap = max(200, int(cap * 0.8))
        blocks = mains(cap)
    while sum(len(b) + 1 for b in blocks) > budget and len(items) > 1:
        items.pop()
        info.dropped_items += 1
        blocks = mains(cap)
    info.item_char_cap = cap

    used = sum(len(b) + 1 for b in blocks)
    per_item_ctx: List[List[str]] = []
    for i, e in enumerate(items, start=1):
        kept = []
        for j, comp in enumerate(e.context, start=1):
            b = ctx_block(f"E{i}.ctx{j}", comp, ctx_cap)
            if used + len(b) + 1 <= budget:
                kept.append(b)
                used += len(b) + 1
                info.ids.append(f"E{i}.ctx{j}")
                info.component_ids.append(comp.id)
            else:
                info.dropped_ctx += 1
        per_item_ctx.append(kept)
    info.ids = [f"E{i}" for i in range(1, len(items) + 1)] + info.ids
    info.component_ids = [e.component.id for e in items] + info.component_ids

    lines: List[str] = []
    for b, ctx in zip(blocks, per_item_ctx):
        lines.append(b)
        lines.extend(ctx)
    ev_text = "\n".join(lines) if lines else "(no evidence retrieved)"
    user = ANSWER_TEMPLATE.format(query=query, qtype=qtype, evidence=ev_text, not_found=not_found)
    if len(SYSTEM_PROMPT) + len(user) > max_tokens * cpt:  # hard guarantee
        keep = int(max_tokens * cpt) - len(fixed) - 4
        user = ANSWER_TEMPLATE.format(query=query, qtype=qtype, evidence=ev_text[:max(0, keep)] + "...",
                                      not_found=not_found)
        info.hard_cut = True
    info.est_tokens = estimate_tokens(SYSTEM_PROMPT + user, cpt)
    return SYSTEM_PROMPT, user, info
