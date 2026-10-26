"""API endpoints with FastAPI's TestClient on a fake pipeline (no models, no network)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import AppState, create_app
from docscout.feedback.session_store import FeedbackStore
from tests.test_pipeline_smoke import make_pipeline
from tests.test_rocchio import VECS

Q = "What was the revenue in 2023?"


class FakeModel:
    """Every question encodes to the same unit vector (question cosine 1)."""

    def encode(self, texts, **kw):
        return np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(texts), 1))


class FakeIndex:
    def __init__(self):
        self.ids = list(VECS)
        self.emb = np.stack([VECS[i] for i in self.ids])
        self.model = FakeModel()


@pytest.fixture
def client(tmp_path):
    pipe, llm = make_pipeline()
    # Rocchio needs BGE vectors, a query encoder and a cross-encoder score function
    pipe.retriever.indexes = {"text": {"dense_bge": FakeIndex()}}
    pipe.retriever._encode = lambda r, q: np.array([1.0, 0.0, 0.0])
    from tests.test_rocchio import fake_ce
    pipe.reranker.score = fake_ce
    pipe.doc_sizes = {"d": 5}
    img = tmp_path / "page_1.png"
    Image.new("RGB", (60, 80), "white").save(img)
    manifests = {"d": {"doc_id": "d", "source": "mpdocvqa", "store": "eval",
                       "pages": [{"page": 1, "width": 600, "height": 800, "image_path": str(img)}]}}
    state = AppState({"eval": pipe}, pipe.cfg, FeedbackStore(), manifests, {"d": [Q]})
    with TestClient(create_app(state)) as c:
        c.llm, c.state = llm, state
        yield c


def ask(client, **kw):
    r = client.post("/ask", json={"doc_id": "d", "query": Q, "top_k": 3, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_documents_and_page_image(client):
    docs = client.get("/documents").json()
    assert docs[0]["doc_id"] == "d" and docs[0]["pages"][0]["width"] == 600 and docs[0]["sample_questions"] == [Q]
    r = client.get("/page-image/d/1")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get("/page-image/d/9").status_code == 404
    assert client.get("/page-image/zz/1").status_code == 404


def test_ask_returns_answer_cards_and_session(client):
    out = ask(client)
    assert out["session_id"] and out["query_id"] and out["query_type"] == "numerical"
    assert out["answer"]["text"] == "1,450" and out["answer"]["citations"] == ["E1"]
    assert out["answer"]["confidence"]["cited_sources"] == 1 and out["answer"]["confidence"]["not_found"] is False
    ev = out["evidence"]
    assert [e["eid"] for e in ev] == ["E1", "E2", "E3"] and ev[0]["id"] == "d_p1_table_0"
    for k in ("bbox", "page_size", "graph_path", "why", "scores", "localized_cells", "context"):
        assert k in ev[0]
    assert ev[0]["localized_cells"] == [["Revenue", "2023", "1,450"]]
    assert out["feedback_active"] is False and client.llm.stats.calls == 1
    assert client.post("/ask", json={"doc_id": "nope", "query": Q}).status_code == 404


def test_feedback_reranks_without_new_answer(client):
    out = ask(client)
    sid, qid = out["session_id"], out["query_id"]
    calls = client.llm.stats.calls
    r = client.post("/feedback", json={"session_id": sid, "query_id": qid, "component_id": "d_p1_table_0", "vote": -1})
    assert r.status_code == 200
    fb = r.json()
    ids = [e["id"] for e in fb["evidence"]]
    assert "d_p1_table_0" not in ids                       # down-voted: excluded
    assert fb["answer"]["text"] == "1,450" and fb["answer_stale"] is True
    assert client.llm.stats.calls == calls                 # no LLM call for a vote
    assert fb["votes"] == {"d_p1_table_0": -1} and fb["feedback_active"] is True
    assert any("rocchio" in e["why"] for e in fb["evidence"])
    up = ids[-1]
    fb2 = client.post("/feedback", json={"session_id": sid, "query_id": qid, "component_id": up, "vote": 1}).json()
    assert fb2["evidence"][0]["id"] == up and fb2["evidence"][0]["feedback_tag"] == "pinned"   # up-voted: pinned
    assert client.state.store.count("evidence") == 2


def test_answer_vote_is_logged_only(client):
    out = ask(client)
    r = client.post("/feedback", json={"session_id": out["session_id"], "query_id": out["query_id"],
                                       "target": "answer", "vote": 1})
    assert r.json() == {"recorded": True, "target": "answer", "query_id": out["query_id"]}
    assert client.state.store.count("answer") == 1
    assert client.state.store.session_votes(out["session_id"], "d", Q) == {}   # not used for ranking


def test_feedback_validation(client):
    out = ask(client)
    base = {"session_id": out["session_id"], "query_id": out["query_id"]}
    assert client.post("/feedback", json={**base, "vote": 1}).status_code == 422          # no component
    assert client.post("/feedback", json={**base, "component_id": "x", "vote": 2}).status_code == 422
    assert client.post("/feedback", json={**base, "session_id": "other", "component_id": "x", "vote": 1}).status_code == 404


def test_next_pages_through_ranking_without_retrieval(client):
    out = ask(client)
    sid, qid = out["session_id"], out["query_id"]
    pipe = client.state.pipelines["eval"]
    n_retrieve = len(pipe.retriever.calls)
    shown = [e["id"] for e in out["evidence"]]
    seen = []
    while True:
        r = client.get("/next", params={"session_id": sid, "query_id": qid}).json()
        if r["item"] is None:
            assert r["message"] == "no further matches"
            break
        seen.append(r["item"]["id"])
        assert r["item"]["why"] and r["item"]["graph_path"]
    assert set(seen).isdisjoint(shown) and len(seen) == out["ranking_size"] - len(shown)
    assert len(pipe.retriever.calls) == n_retrieve          # no new retrieval
    assert client.get("/next", params={"session_id": "x", "query_id": qid}).status_code == 404


def test_memory_flag_carries_votes_across_sessions(client):
    for _ in range(2):  # two earlier sessions up-vote the heading
        o = ask(client, memory=True)
        r = client.post("/feedback", json={"session_id": o["session_id"], "query_id": o["query_id"],
                                           "component_id": "d_p1_heading_0", "vote": 1})
        assert r.status_code == 200, r.text
    new = ask(client, memory=True)
    off = ask(client, memory=False)
    assert new["memory"]["prior_items"] == 1 and new["feedback_active"] is True
    assert off["memory"] == {"on": False, "prior_items": 0} and off["feedback_active"] is False
    h = next(e for e in new["evidence"] if e["id"] == "d_p1_heading_0")
    assert h["feedback_tag"] == "memory" and h["scores"]["feedback"] == pytest.approx(0.10)


def _fake_upload_env(client, tmp_path):
    st = client.state
    st.upload_dir = tmp_path / "pdfs"
    calls = []

    def runner(pdf, log):
        calls.append(pdf.name)
        log("fake ingest " + pdf.name)

    def reload():  # like reload_demo_store: register the demo pipeline and its new document
        from types import SimpleNamespace
        st.pipelines["demo"] = SimpleNamespace(doc_sizes={"my-report": 3})
        st.doc_store["my-report"] = "demo"
        st.manifests["my-report"] = {"doc_id": "my-report", "source": "pdf", "pages": [], "store": "demo"}
    st.ingest_runner, st.reload_demo = runner, reload
    return calls


def test_ingest_upload_runs_in_background_and_adds_document(client, tmp_path):
    import time
    calls = _fake_upload_env(client, tmp_path)
    r = client.post("/ingest", files={"file": ("My Report.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert r.status_code == 202 and r.json()["doc_id"] == "my-report"
    for _ in range(100):
        s = client.get("/ingest/status").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert s["status"] == "done", s
    assert calls == ["My Report.pdf"] and (tmp_path / "pdfs" / "My Report.pdf").read_bytes() == b"%PDF-1.4 fake"
    assert any("fake ingest" in line for line in s["log_tail"])
    docs = [d["doc_id"] for d in client.get("/documents").json()]
    assert "my-report" in docs and "d" in docs          # the new document is added; existing ones stay


def test_ingest_rejects_non_pdf_and_concurrent_uploads(client, tmp_path):
    calls = _fake_upload_env(client, tmp_path)
    assert client.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")}).status_code == 415
    assert client.post("/ingest", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")}).status_code == 415
    client.state.ingest["status"] = "running"            # one upload at a time
    r = client.post("/ingest", files={"file": ("second.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 409 and calls == []
    assert not (tmp_path / "pdfs" / "second.pdf").exists()
    assert client.get("/ingest/status").json()["status"] == "running"
