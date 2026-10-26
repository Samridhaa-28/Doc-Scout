"""Feedback store and the cross-session memory gates: cap, min votes, similarity, doc / qtype match, decay."""
import copy

import numpy as np
import pytest

from docscout.config import get_config
from docscout.feedback.memory import memory_prior
from docscout.feedback.session_store import FeedbackStore

DAY = 86400.0
NOW = 1_000 * DAY
A = np.array([1.0, 0.0])                 # past question
NEAR = np.array([0.8, 0.6])              # cosine 0.8 with A (>= 0.75)
FAR = np.array([0.6, 0.8])               # cosine 0.6 with A (< 0.75)


@pytest.fixture
def cfg():
    c = copy.deepcopy(get_config())
    c["feedback"].update({"step": 0.05, "cap": 0.15, "min_votes": 2, "min_similarity": 0.75, "half_life_days": 30})
    return c


def votes(store, cid, n, vote=1, doc="d", qtype="numerical", vec=A, age_days=0.0, session="old"):
    for _ in range(n):
        store.record(session, doc, "past question", qtype, cid, vote, query_vec=vec, timestamp=NOW - age_days * DAY)


def test_store_session_votes_latest_wins_and_same_question_only():
    s = FeedbackStore()
    s.record("s1", "d", "What is X?", "textual", "c1", 1, timestamp=1)
    s.record("s1", "d", "what is  x?", "textual", "c1", -1, timestamp=2)   # same question (normalized)
    s.record("s1", "d", "What is X?", "textual", "c2", 1, timestamp=3)
    s.record("s1", "d", "Other question", "textual", "c3", 1, timestamp=4)
    s.record("s2", "d", "What is X?", "textual", "c4", 1, timestamp=5)
    s.record("s1", "d", "What is X?", "textual", None, 1, target="answer", timestamp=6)
    assert s.session_votes("s1", "d", "What is X?") == {"c1": -1, "c2": 1}
    assert s.count("answer") == 1 and s.count() == 6


def test_store_rejects_bad_votes():
    s = FeedbackStore()
    with pytest.raises(ValueError):
        s.record("s", "d", "q", "textual", "c", 0)
    with pytest.raises(ValueError):
        s.record("s", "d", "q", "textual", None, 1)            # evidence vote without component


def test_store_persists_to_file(tmp_path):
    p = tmp_path / "fb.sqlite"
    s = FeedbackStore(p)
    s.record("s", "d", "q", "textual", "c", 1, query_vec=A)
    s.close()
    v = FeedbackStore(p).doc_votes("d", "textual")
    assert len(v) == 1 and np.allclose(v[0].query_vec, A)


def test_prior_needs_min_votes(cfg):
    s = FeedbackStore()
    votes(s, "c1", 1)
    assert memory_prior(s, "d", "numerical", NEAR, cfg, now=NOW) == {}
    votes(s, "c1", 1)
    assert memory_prior(s, "d", "numerical", NEAR, cfg, now=NOW) == {"c1": pytest.approx(0.10)}


def test_prior_is_capped(cfg):
    s = FeedbackStore()
    votes(s, "up", 10)
    votes(s, "down", 10, vote=-1)
    assert memory_prior(s, "d", "numerical", NEAR, cfg, now=NOW) == {"up": 0.15, "down": -0.15}


def test_prior_similarity_gate(cfg):
    s = FeedbackStore()
    votes(s, "c1", 3)
    assert memory_prior(s, "d", "numerical", FAR, cfg, now=NOW) == {}
    assert "c1" in memory_prior(s, "d", "numerical", NEAR, cfg, now=NOW)


def test_prior_doc_and_qtype_must_match(cfg):
    s = FeedbackStore()
    votes(s, "c1", 3, doc="other")
    votes(s, "c2", 3, qtype="textual")
    assert memory_prior(s, "d", "numerical", A, cfg, now=NOW) == {}
    assert memory_prior(s, "other", "numerical", A, cfg, now=NOW) == {"c1": pytest.approx(0.15)}


def test_prior_decays_with_half_life(cfg):
    s = FeedbackStore()
    votes(s, "c1", 1, age_days=0)
    votes(s, "c1", 1, age_days=30)       # weight 0.5
    assert memory_prior(s, "d", "numerical", A, cfg, now=NOW)["c1"] == pytest.approx(0.05 * 1.5)


def test_prior_excludes_current_session_and_answer_votes(cfg):
    s = FeedbackStore()
    votes(s, "c1", 3, session="me")
    s.record("old", "d", "q", "numerical", None, 1, target="answer", query_vec=A, timestamp=NOW)
    assert memory_prior(s, "d", "numerical", A, cfg, now=NOW, exclude_session="me") == {}
    assert "c1" in memory_prior(s, "d", "numerical", A, cfg, now=NOW)
