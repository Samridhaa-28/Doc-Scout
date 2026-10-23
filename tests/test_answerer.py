"""Prompt building (numbering, cells, size cap, evidence only) and answer validation."""
import json

import pytest

from docscout.config import get_config
from docscout.reasoning.answerer import answer_question, evidence_for_citation, normalize_citations, validate_answer
from docscout.reasoning.prompts import build_prompt, estimate_tokens
from docscout.schema import Component, Evidence

NF = "Not found in the retrieved evidence"


def comp(cid, typ="paragraph", page=1, text="some text"):
    return Component(id=cid, doc_id="d", page=page, type=typ, text=text, text_form=text)


class FakeClient:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete_json(self, prompt, system=None):
        self.prompts.append((system, prompt))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_prompt_numbering_cells_and_context_labels():
    t = comp("d_p9_table_0", "table", 9, "Table: Budget\nColumns: Item | 2023")
    e1 = Evidence(component=t, localized_cells=[("Revenue", "2023", "1,450")])
    e1.context = [comp("d_p9_heading_0", "heading", 9, "Budget")]
    e2 = Evidence(component=comp("d_p3_paragraph_1", page=3, text="Presiding: Lee A. Waller"))
    system, user, info = build_prompt("revenue 2023?", "numerical", [e1, e2])
    assert "[E1] table | page 9" in user and "[E2] paragraph | page 3" in user
    assert "[E1.ctx1] heading | page 9: Budget" in user
    assert "| Revenue | 2023 | 1,450 |" in user
    assert info.ids == ["E1", "E2", "E1.ctx1"]
    assert NF in user and "JSON" in user


def test_prompt_cap_and_llm_never_sees_whole_document():
    cfg = get_config()
    cfg["answer"]["max_prompt_tokens"] = 600
    doc = [comp(f"d_p{i}_paragraph_0", page=i, text=f"UNIQUE{i} " + "lorem ipsum " * 150) for i in range(1, 41)]
    evidence = [Evidence(component=c, context=[doc[(i + 10) % 40]]) for i, c in enumerate(doc[:5])]
    system, user, info = build_prompt("q?", "textual", evidence, cfg)
    assert estimate_tokens(system + user) <= 600
    assert info.est_tokens <= 600
    seen = [i for i in range(1, 41) if f"UNIQUE{i} " in user]
    assert set(info.component_ids) <= {c.id for c in doc[:5]} | {doc[(i + 10) % 40].id for i in range(5)}
    assert len(seen) < len(doc) and all(i <= 15 for i in seen)  # only evidence (and its context), never the rest
    assert info.dropped_items + info.dropped_ctx > 0 or info.item_char_cap < cfg["answer"]["max_item_chars"]


def test_validate_good_answer_and_drop_invalid_citations():
    r = validate_answer({"answer": "1,450", "citations": ["[E1]", "E7", "page 3", "e2.CTX1"]}, ["E1", "E2", "E2.ctx1"], NF)
    assert r.answer == "1,450" and r.citations == ["E1", "E2.ctx1"] and not r.not_found
    assert r.invalid_citations == ["E7", "page 3"]


def test_answer_without_valid_citation_is_not_found():
    r = validate_answer({"answer": "1,450", "citations": ["E9"]}, ["E1"], NF)
    assert r.not_found and r.no_valid_citation and r.answer == NF and r.raw_answer == "1,450"
    r = validate_answer({"answer": "1,450"}, ["E1"], NF)
    assert r.not_found and r.no_valid_citation


@pytest.mark.parametrize("ans", [NF, NF.lower() + ".", "", "  "])
def test_not_found_variants(ans):
    r = validate_answer({"answer": ans, "citations": ["E1"]}, ["E1"], NF)
    assert r.not_found and r.citations == [] and not r.no_valid_citation and r.answer == NF


def test_normalize_citations_string_and_list():
    assert normalize_citations("E1, [E2] and E1") == ["E1", "E2"]
    assert normalize_citations(["E3.ctx2"]) == ["E3.ctx2"]


def test_answer_question_with_mock_and_llm_failure():
    ev = [Evidence(component=comp("d_p1_paragraph_0", text="Presiding: Lee A. Waller"))]
    client = FakeClient({"answer": "Lee A. Waller", "citations": ["E1"]})
    r = answer_question("Who is presiding?", "textual", ev, client)
    assert r.answer == "Lee A. Waller" and r.citations == ["E1"]
    assert "Presiding: Lee A. Waller" in client.prompts[0][1]
    r = answer_question("q", "textual", ev, FakeClient(RuntimeError("boom")))
    assert r.not_found and r.error.startswith("RuntimeError")
    r = answer_question("q", "textual", [], FakeClient({"answer": "x"}))
    assert r.not_found and r.error == "no evidence"


def test_evidence_for_citation():
    e = Evidence(component=comp("a"), context=[comp("b")])
    assert evidence_for_citation("E1", [e]) == "a"
    assert evidence_for_citation("E1.ctx1", [e]) == "b"
    assert evidence_for_citation("E2", [e]) is None


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    from eval.eval_answers import bootstrap_ci

    vals = [0.0, 1.0] * 50
    lo, hi = bootstrap_ci(vals)
    assert lo < 0.5 < hi and bootstrap_ci(vals) == [lo, hi]
    assert bootstrap_ci([0.7] * 5) == [0.7, 0.7]
