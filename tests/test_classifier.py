"""Query classifier: word-boundary rules, unsure cases, LLM fallback via a mock (no network)."""
import pytest

from docscout.query.classifier import classify, rule_classify, weights_for
from eval.eval_classifier import answer_is_numeric, map_types, proxy_label


class MockLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete_json(self, prompt, system=None):
        self.prompts.append(prompt)
        return self.reply


@pytest.mark.parametrize("q,label", [
    ("How many pages are there?", "numerical"),
    ("What is the total amount?", "numerical"),
    ("What is the date mentioned?", "numerical"),
    ("What was the value in 1975?", "numerical"),
    ("What percentage of sales?", "numerical"),
    ("What does the logo show?", "visual"),
    ("What is shown in the pictures?", "visual"),
    ("Which diagram is on top?", "visual"),
])
def test_rules_single_group(q, label):
    r = rule_classify(q, confidence=0.9)
    assert r is not None and r["type"] == label and r["source"] == "rules" and r["confidence"] == 0.9


@pytest.mark.parametrize("q", [
    "Who is the author?",                    # neither group
    "What is the title of the graph in 1990?",  # both groups
    "What is photosynthesis?",               # 'photo' only as part of a word
    "Totally unrelated question",            # 'total' only as part of a word
    "He figured it out?",
])
def test_rules_unsure(q):
    assert rule_classify(q, confidence=0.9) is None


def test_llm_used_only_when_rules_unsure():
    llm = MockLLM({"type": "visual", "confidence": 0.7})
    out = classify("What is the title of the graph in 1990?", client=llm, use_llm=True)
    assert out == {"type": "visual", "confidence": 0.7, "source": "llm", "unsure_rules": True}
    assert len(llm.prompts) == 1 and "graph in 1990" in llm.prompts[0]
    out = classify("How many items?", client=llm, use_llm=True)
    assert out["source"] == "rules" and len(llm.prompts) == 1  # rules decided, no LLM call


def test_llm_off_falls_back_and_bad_reply_raises():
    out = classify("Who is the author?", client=None, use_llm=False)
    assert out["type"] == "textual" and out["source"] == "rules" and out["unsure"]
    with pytest.raises(ValueError):
        classify("Who is the author?", client=MockLLM({"type": "tabular"}), use_llm=True)
    clamp = classify("Who is the author?", client=MockLLM({"type": "Textual", "confidence": 3}), use_llm=True)
    assert clamp["type"] == "textual" and clamp["confidence"] == 1.0


def test_weights_for_reads_config():
    w = weights_for("numerical")
    assert set(w) == {"text", "table", "figure"} and abs(sum(w.values()) - 1.0) < 1e-6
    with pytest.raises(KeyError):
        weights_for("other")


def test_gold_mapping_priority():
    assert map_types(["figure/diagram", "table/list"], ["0.28"]) == "visual"
    assert map_types(["table/list"], ["Smith"]) == "numerical"
    assert map_types(["form"], ["March 3, 1998"]) == "numerical"
    assert map_types(["others"], ["university of california"]) == "textual"
    assert answer_is_numeric("$1,234") and answer_is_numeric("12/03/1998") and not answer_is_numeric("Jan Smith")
    assert proxy_label("What is in the photo?", ["dog"]) == "visual"
