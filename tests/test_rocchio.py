"""Session Rocchio: the math on hand-made vectors, pinning / exclusion, the lambda blend and the identity default."""
import numpy as np
import pytest

from docscout.feedback.rocchio import PIN_OFFSET, CachedScorer, FeedbackHook, rocchio_query
from docscout.retrieval.normalize import minmax
from tests.test_pipeline_smoke import FakeReranker, make_pipeline

Q = "What was the revenue in 2023?"
VECS = {  # unit vectors in a 3-d "BGE" space
    "d_p1_table_0": np.array([1.0, 0.0, 0.0]),
    "d_p1_paragraph_0": np.array([0.0, 1.0, 0.0]),
    "d_p1_heading_0": np.array([0.6, 0.8, 0.0]),
    "d_p2_paragraph_0": np.array([0.0, 0.0, 1.0]),
    "d_p2_figure_0": np.array([0.6, 0.0, 0.8]),
}
Q_VEC = np.array([1.0, 0.0, 0.0])


def fake_ce(query, texts):
    return [next((v for k, v in FakeReranker.SCORES.items() if t.startswith(k)), -5.0) for t in texts]


def hook(pipe, votes, memory=None, **params):
    return FeedbackHook(Q, votes, Q_VEC, VECS.get, pipe.texts, pipe.cfg, scorer=CachedScorer(fake_ce),
                        memory=memory, params=params or None)


# ---------------------------------------------------------------- math
def test_rocchio_query_hand_computed():
    pos = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    neg = [np.array([0.0, 0.0, 1.0])]
    got = rocchio_query(np.array([1.0, 0.0, 0.0]), pos, neg, alpha=1.0, beta=0.75, gamma=0.15)
    raw = np.array([1.0, 0.75 * 0.5, 0.75 * 0.5 - 0.15])  # beta*mean(pos) - gamma*mean(neg)
    assert np.allclose(got, raw / np.linalg.norm(raw))
    assert np.isclose(np.linalg.norm(got), 1.0)


def test_rocchio_query_empty_groups_and_gamma_zero():
    q = np.array([3.0, 4.0])
    assert np.allclose(rocchio_query(q, [], []), [0.6, 0.8])
    assert np.allclose(rocchio_query(q, [], [np.array([1.0, 0.0])], gamma=0.0), [0.6, 0.8])


# ---------------------------------------------------------------- identity default (P5/P6 unchanged)
def test_no_votes_no_memory_is_identity_and_final_is_raw_ce():
    pipe, _ = make_pipeline()
    base = pipe.answer(Q, "d", top_k=3, generate=False)
    h = hook(pipe, {})
    res = pipe.answer(Q, "d", top_k=3, generate=False, feedback_adjust=h)
    assert res.meta["reranked"] == base.meta["reranked"]
    top = res.meta["reranked"][0]
    assert top["id"] == "d_p1_table_0" and top["final"] == top["rerank"] == 8.5  # raw CE, as in P6
    assert h.tags == {} and "fb" not in res.evidence[0].why


# ---------------------------------------------------------------- votes
def test_upvote_pinned_downvote_excluded():
    pipe, _ = make_pipeline()
    h = hook(pipe, {"d_p2_paragraph_0": 1, "d_p1_table_0": -1})
    res = pipe.answer(Q, "d", session_id="s", top_k=3, generate=False, feedback_adjust=h)
    ids = [r["id"] for r in res.meta["reranked"]]
    assert ids[0] == "d_p2_paragraph_0"          # CE -5 (worst), but pinned at the top
    assert "d_p1_table_0" not in ids              # excluded from the results
    assert res.meta["reranked"][0]["final"] >= PIN_OFFSET
    assert h.tags["d_p2_paragraph_0"] == "pinned"
    assert "| pinned" in res.evidence[0].why


def test_rocchio_rescoring_and_lambda_blend():
    pipe, _ = make_pipeline()
    h = hook(pipe, {"d_p2_paragraph_0": 1}, beta=0.75, gamma=0.15, **{"lambda": 0.5})
    pipe.answer(Q, "d", top_k=3, generate=False, feedback_adjust=h)
    qp = rocchio_query(Q_VEC, [VECS["d_p2_paragraph_0"]], [], 1.0, 0.75, 0.15)
    assert np.allclose(h.q_prime, qp)
    rest = [c for c in h.ranked if c.id != "d_p2_paragraph_0"]
    head = {c.id: c for c in h.ranked if c.rerank is not None}
    ce_n = minmax({cid: c.rerank for cid, c in head.items()})
    cos_n = minmax({cid: float(VECS[cid] @ qp) for cid in head})
    for c in rest:
        assert c.final == pytest.approx(0.5 * ce_n[c.id] + 0.5 * cos_n[c.id])
        assert c.raw["dense_bge"] == pytest.approx(float(VECS[c.id] @ qp))  # dense re-scored with q'
        assert h.deltas[c.id] == pytest.approx(c.final - ce_n[c.id])
        assert h.tags[c.id] == "rocchio"
    # the figure (closest to the up-voted direction among the rest) moves above the paragraph it trailed by CE
    order = [c.id for c in rest]
    assert order.index("d_p2_figure_0") < order.index("d_p1_paragraph_0")


def test_lambda_zero_is_the_literal_pipeline_order():
    pipe, _ = make_pipeline()
    h = hook(pipe, {"d_p1_heading_0": -1}, **{"lambda": 0.0})
    pipe.answer(Q, "d", top_k=3, generate=False, feedback_adjust=h)
    head = [c for c in h.ranked if c.rerank is not None]
    assert [c.id for c in head] == [c.id for c in sorted(head, key=lambda c: (-c.rerank, c.id))]
    assert all(0.0 <= c.final <= 1.0 for c in head)


def test_why_line_shows_fb_and_rocchio():
    pipe, _ = make_pipeline()
    h = hook(pipe, {"d_p2_figure_0": 1})
    res = pipe.answer(Q, "d", top_k=3, generate=False, feedback_adjust=h)
    e = next(e for e in res.evidence if h.tags.get(e.component.id) == "rocchio")
    assert " fb " in e.why or ", fb " in e.why
    assert "| rocchio" in e.why and e.meta["feedback_tag"] == "rocchio"
    assert e.scores["feedback"] == pytest.approx(h.deltas[e.component.id])


def test_cached_scorer_scores_only_new_texts():
    calls = []

    def ce(q, ts):
        calls.append(list(ts))
        return [float(len(t)) for t in ts]
    s = CachedScorer(ce)
    assert s("q", ["a", "bb"]) == [1.0, 2.0]
    assert s("q", ["bb", "ccc", "a"]) == [2.0, 3.0, 1.0]
    assert calls == [["a", "bb"], ["ccc"]]


def test_memory_only_uses_normalized_ce_plus_prior():
    pipe, _ = make_pipeline()
    h = hook(pipe, {}, memory={"d_p1_paragraph_0": 0.15})
    pipe.answer(Q, "d", top_k=3, generate=False, feedback_adjust=h)
    by = {c.id: c for c in h.ranked}
    ce_n = minmax({c.id: c.rerank for c in h.ranked if c.rerank is not None})
    assert by["d_p1_paragraph_0"].final == pytest.approx(ce_n["d_p1_paragraph_0"] + 0.15)
    assert by["d_p1_table_0"].final == pytest.approx(1.0)
    assert h.tags == {"d_p1_paragraph_0": "memory"} and h.q_prime is None
