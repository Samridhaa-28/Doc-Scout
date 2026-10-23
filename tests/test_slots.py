"""Modality slot reservation (opt-in): keeps reranked order, k fixed, top-1 never replaced, input not modified."""
from docscout.retrieval.retrievers import Candidate
from docscout.retrieval.slots import best_by_fused, reserve_modality_slots


def cand(cid, typ, fused, final):
    branch = {"table": "table", "figure": "figure"}.get(typ, "text")
    return Candidate(cid, "d", 1, typ, branch, fused=fused, final=final)


def reranked():
    # already in reranked order (final decreasing); fused = pre-rerank hybrid score
    return [cand("p1", "paragraph", 0.30, 9.0), cand("p2", "paragraph", 0.25, 8.0), cand("h1", "heading", 0.10, 7.0),
            cand("p3", "paragraph", 0.20, 6.0), cand("p4", "paragraph", 0.15, 5.0),
            cand("t1", "table", 0.60, 1.0), cand("t2", "table", 0.40, 2.0), cand("f1", "figure", 0.50, 0.5)]


def ids(cs):
    return [c.id for c in cs]


def test_missing_table_and_figure_replace_lowest_prose_and_keep_order():
    rr = reranked()
    ev, res = reserve_modality_slots(rr, 5)
    assert len(ev) == 5
    # best table by fused is t1 (0.60, not t2 which reranks higher); best figure f1. They replace p4, then p3.
    assert ids(ev) == ["p1", "p2", "h1", "t1", "f1"]
    assert res == {"t1": "table", "f1": "figure"}
    assert ids(rr) == ["p1", "p2", "h1", "p3", "p4", "t1", "t2", "f1"]  # full reranked list untouched


def test_already_present_modality_is_not_duplicated():
    rr = reranked()
    rr.insert(1, rr.pop(5))  # t1 now at rank 2
    ev, res = reserve_modality_slots(rr, 5)
    assert ids(ev).count("t1") == 1 and res == {"f1": "figure"}
    assert ids(ev) == ["p1", "t1", "p2", "h1", "f1"]


def test_top1_never_replaced_and_k1_is_noop():
    rr = [cand("p1", "paragraph", 0.3, 9.0), cand("t1", "table", 0.6, 1.0)]
    assert ids(reserve_modality_slots(rr, 1)[0]) == ["p1"]
    ev, res = reserve_modality_slots([rr[0], cand("p2", "paragraph", 0.1, 5.0), rr[1]], 2)
    assert ids(ev) == ["p1", "t1"] and res == {"t1": "table"}  # p2 (rank 2) replaced, p1 (rank 1) kept


def test_no_prose_to_replace_or_no_modality_in_document():
    rr = [cand("t2", "table", 0.4, 9.0), cand("t3", "table", 0.3, 8.0), cand("f1", "figure", 0.5, 1.0)]
    ev, res = reserve_modality_slots(rr, 2)  # only tables in the evidence: nothing to replace
    assert ids(ev) == ["t2", "t3"] and res == {}
    prose_only = [cand("p1", "paragraph", 0.3, 9.0), cand("p2", "paragraph", 0.2, 8.0)]
    assert reserve_modality_slots(prose_only, 5) == (prose_only, {})


def test_best_by_fused_ties_by_id():
    assert best_by_fused([cand("b", "table", 0.5, 0), cand("a", "table", 0.5, 0)], "table").id == "a"
    assert best_by_fused([cand("p", "paragraph", 0.5, 0)], "table") is None


def test_type_conditional_variant():
    import pytest
    from docscout.retrieval.slots import slot_modalities

    assert slot_modalities("conditional", "numerical") == ("table",)
    assert slot_modalities("conditional", "visual") == ("figure",)
    assert slot_modalities("conditional", "textual") == ()
    assert slot_modalities("both", "textual") == ("table", "figure")
    assert slot_modalities("off", "numerical") == ()
    with pytest.raises(ValueError):
        slot_modalities("always", "numerical")
    rr = reranked()
    ev_num, res_num = reserve_modality_slots(rr, 5, slot_modalities("conditional", "numerical"))
    assert res_num == {"t1": "table"} and ids(ev_num) == ["p1", "p2", "h1", "p3", "t1"]
    ev_vis, res_vis = reserve_modality_slots(rr, 5, slot_modalities("conditional", "visual"))
    assert res_vis == {"f1": "figure"} and ids(ev_vis) == ["p1", "p2", "h1", "p3", "f1"]
    ev_txt, res_txt = reserve_modality_slots(rr, 5, slot_modalities("conditional", "textual"))
    assert res_txt == {} and ids(ev_txt) == ids(rr[:5])
