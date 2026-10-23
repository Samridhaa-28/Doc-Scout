"""One question end to end through DocScoutPipeline with fake retriever / cross-encoder / LLM (no models, no network)."""
import pandas as pd

from docscout.evidence.provenance import why_line
from docscout.graph.builder import build_graph
from docscout.pipeline import DocScoutPipeline
from docscout.rerank.cross_encoder import rerank_candidates
from docscout.retrieval.retrievers import Candidate
from docscout.schema import Component


def make_components():
    def c(cid, page, typ, order, text, section=None, **kw):
        return Component(id=cid, doc_id="d", page=page, type=typ, text=text, text_form=text, order=order,
                         section_id=section, bbox=(10, 20, 300, 400), page_width=600, page_height=800, **kw)
    comps = [
        c("d_p1_heading_0", 1, "heading", 0, "Revenue"),
        c("d_p1_table_0", 1, "table", 1, "Table: Revenue by year\nColumns: Segment | 2022 | 2023", "d_p1_heading_0",
          table_path="fake.parquet"),
        c("d_p1_paragraph_0", 1, "paragraph", 2, "Revenue grew strongly in 2023.", "d_p1_heading_0"),
        c("d_p2_paragraph_0", 2, "paragraph", 3, "Unrelated text about staff."),
        c("d_p2_figure_0", 2, "figure", 4, "Figure: bar chart of revenue"),
        c("x_p1_paragraph_0", 1, "paragraph", 0, "Another document"),
    ]
    comps[-1].doc_id = "x"
    return {x.id: x for x in comps}


class FakeRetriever:
    RAW = {"d_p1_heading_0": ("text", 2.0, 0.6), "d_p1_table_0": ("table", 5.0, 0.7),
           "d_p1_paragraph_0": ("text", 4.0, 0.65), "d_p2_paragraph_0": ("text", 0.0, 0.2),
           "d_p2_figure_0": ("figure", 1.0, 0.5)}

    def __init__(self, comps):
        self.comps, self.calls = comps, []

    def retrieve(self, query, doc_id, branches=None):
        self.calls.append((query, doc_id))
        out = []
        for cid, (b, bm, de) in self.RAW.items():
            raw = {"bm25": bm, "dense_bge": de}
            if b == "figure":
                raw["clip"] = 0.3
            out.append(Candidate(cid, doc_id, self.comps[cid].page, self.comps[cid].type, b, raw=raw))
        return sorted(out, key=lambda c: c.id)


class FakeReranker:
    """Fixed cross-encoder scores per text (table best)."""
    SCORES = {"Table": 8.5, "Revenue grew": 2.0, "Revenue": 1.0, "Figure": -1.0}

    def rerank(self, query, ranked, texts):
        def score(q, ts):
            return [next((v for k, v in self.SCORES.items() if t.startswith(k)), -5.0) for t in ts]
        return rerank_candidates(query, ranked, texts, score, top_n=30)


class FakeLLM:
    model = "fake-model"

    def __init__(self):
        from docscout.reasoning.llm_client import LLMStats
        self.stats, self.prompts = LLMStats(), []

    def complete_json(self, prompt, system=None):
        self.prompts.append(prompt)
        self.stats.calls += 1
        self.stats.prompt_tokens += len(prompt) // 4
        return {"answer": "1,450", "citations": ["E1", "E99"]}


def make_pipeline():
    comps = make_components()
    table = pd.DataFrame({"Segment": ["Revenue", "Costs"], "2022": ["1,200", "900"], "2023": ["1,450", "1,000"]})
    llm = FakeLLM()
    pipe = DocScoutPipeline(
        retriever=FakeRetriever(comps), reranker=FakeReranker(), client=llm, components=comps,
        graph_loader=lambda doc: build_graph([c for c in comps.values() if c.doc_id == doc]),
        table_loader=lambda comp: table,
        classify_fn=lambda q, client=None, use_llm=None: {"type": "numerical", "confidence": 0.9, "source": "rules"})
    return pipe, llm


def test_pipeline_end_to_end():
    pipe, llm = make_pipeline()
    res = pipe.answer("What was the revenue in 2023?", "d", top_k=3)
    # answer + validated citations (E99 dropped)
    assert res.answer == "1,450" and res.meta["citations"] == ["E1"]
    assert res.meta["answer_validation"]["invalid_citations"] == ["E99"]
    assert res.query_type == "numerical"
    # full reranked list = every component of the document, best first
    rr = res.meta["reranked"]
    assert {r["id"] for r in rr} == {c for c in FakeRetriever.RAW}
    assert [r["final"] for r in rr] == sorted((r["final"] for r in rr), reverse=True)
    assert len(res.evidence) == 3 and [e.component.id for e in res.evidence] == [r["id"] for r in rr[:3]]
    # the table is first and localized to one cell
    e1 = res.evidence[0]
    assert e1.component.id == "d_p1_table_0"
    assert e1.localized_cells == [("Revenue", "2023", "1,450")] and e1.meta["localize_flag"] == "cell"
    # provenance fields
    for e in res.evidence:
        for k in ("eid", "graph_path", "page", "bbox", "matched_terms", "branch_weight"):
            assert k in e.meta
        assert {"bm25_raw", "bm25_norm", "dense_bge_raw", "fused", "rerank", "final"} <= set(e.scores)
        assert e.why.startswith("numerical (") and "fused" in e.why and "CE" in e.why and "Page" in e.why
    assert e1.meta["graph_path"] == "Page 1 -> Section 'Revenue' -> Table 0"
    assert "revenue" in e1.meta["matched_terms"] and "2023" in e1.meta["matched_terms"]
    # graph context is attached and never repeats an evidence item
    ev_ids = {e.component.id for e in res.evidence}
    assert all(x.id not in ev_ids for e in res.evidence for x in e.context)
    # the LLM saw only evidence of this document, one call
    assert len(llm.prompts) == 1 and "Another document" not in llm.prompts[0]
    shown = set(res.meta["prompt"]["component_ids"])
    allowed = ev_ids | {x.id for e in res.evidence for x in e.context}
    assert shown <= allowed and all(pipe.components[x].doc_id == "d" for x in shown)
    assert "| Revenue | 2023 | 1,450 |" in llm.prompts[0]
    assert res.meta["llm"]["calls"] == 1 and res.meta["model"] == "fake-model"
    steps = ["classify", "retrieve", "normalize", "fuse", "rerank", "feedback", "localize", "expand", "answer",
             "provenance"]
    assert list(res.meta["timings_ms"])[:10] == steps


def test_feedback_hook_reorders_and_shows_in_why():
    pipe, _ = make_pipeline()

    def boost_paragraphs(ranked, qtype, session_id):
        for c in ranked:
            if c.type == "paragraph":
                c.final += 100.0
        return ranked
    res = pipe.answer("What was the revenue in 2023?", "d", session_id="s1", top_k=2, feedback_adjust=boost_paragraphs)
    assert res.evidence[0].component.type == "paragraph"
    assert res.evidence[0].scores["feedback"] == 100.0 and "fb +100.00" in res.evidence[0].why
    assert res.meta["session_id"] == "s1"


def test_ablation_switches():
    pipe, llm = make_pipeline()
    res = pipe.answer("What was the revenue in 2023?", "d", top_k=1, use_context=False, use_localization=False)
    assert res.evidence[0].context == [] and res.evidence[0].localized_cells == []
    assert "table_preview" in res.evidence[0].meta  # table still shown as its first rows


def test_why_line_format():
    c = Candidate("d_p9_table_0", "d", 9, "table", "table", raw={"bm25": 11.954, "dense_bge": 0.52},
                  norm={"bm25": 1.0, "dense_bge": 1.0}, fused=0.6, rerank=8.512, final=8.512)
    line = why_line(c, "numerical", {"text": 0.3, "table": 0.6}, ["trrf", "session"], "Page 9 -> Table 0", (90, 297, 520, 610))
    assert line == ("numerical (table 0.6) | bm25 11.95->1.00, bge 0.52->1.00, fused 0.600, CE 8.51 | "
                    "terms: trrf, session | Page 9 -> Table 0 | bbox 90,297,520,610")


def test_classifier_llm_failure_falls_back_to_rules():
    pipe, _ = make_pipeline()

    def flaky(q, client=None, use_llm=None):
        if use_llm is False:
            return {"type": "textual", "confidence": 0.5, "source": "rules", "unsure": True}
        raise RuntimeError("connection error")
    pipe._classify = flaky
    res = pipe.answer("Who wrote it?", "d", top_k=1)
    assert res.query_type == "textual" and "llm_error" in res.meta["classification"]


def test_slots_flag_changes_evidence_not_ranking():
    pipe, _ = make_pipeline()
    q = "What was the revenue in 2023?"
    off = pipe.answer(q, "d", top_k=2, slots="off")
    pipe2, _ = make_pipeline()
    pipe2.reranker.SCORES = {"Revenue grew": 9.0, "Revenue": 8.0, "Table": -2.0, "Figure": -3.0}  # table reranks low
    plain = pipe2.answer(q, "d", top_k=2, slots="off")
    assert all(e.component.type != "table" for e in plain.evidence) and plain.meta["slots"]["variant"] == "off"
    pipe3, _ = make_pipeline()
    pipe3.reranker.SCORES = dict(pipe2.reranker.SCORES)
    cond = pipe3.answer(q, "d", top_k=2, slots="conditional")
    assert [r["id"] for r in cond.meta["reranked"]] == [r["id"] for r in plain.meta["reranked"]]  # ranking unchanged
    assert [e.component.type for e in cond.evidence] == ["paragraph", "table"]
    assert cond.meta["slots"]["reserved"] == {"d_p1_table_0": "table"}
    assert cond.evidence[1].meta["reserved_slot"] == "table" and "slot: best fused table" in cond.evidence[1].why
    assert off.meta["slots"] == {"variant": "off", "reserved": {}}
    # no slots argument -> the configured default (answer.slots: conditional since the P6 DEV decision)
    pipe4, _ = make_pipeline()
    pipe4.reranker.SCORES = dict(pipe2.reranker.SCORES)
    default = pipe4.answer(q, "d", top_k=2)
    assert default.meta["slots"]["variant"] == pipe4.cfg["answer"]["slots"] == "conditional"
    assert default.meta["slots"]["reserved"] == {"d_p1_table_0": "table"}
