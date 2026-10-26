"""DocScout-X API (FastAPI). Pipelines are loaded once at startup; votes go to the SQLite feedback store.

  POST /ask                    answer + evidence cards for (doc_id, query); session id created if missing
  POST /feedback               vote on an evidence item (re-ranks with session Rocchio, no new answer) or on the answer
  GET  /next                   next item of the current ranking (no new retrieval)
  GET  /documents              documents of the eval and demo stores (+ sample questions)
  GET  /page-image/{doc}/{page} the page PNG (highlights are drawn by the UI from bbox / cell_boxes)

Run: .venv/Scripts/python -m uvicorn api.main:app --port 8000      (Swagger UI at /docs)
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

SNIPPET_CHARS = 700


# ---------------------------------------------------------------- request / response models
class AskRequest(BaseModel):
    doc_id: str
    query: str = Field(min_length=1)
    session_id: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)
    memory: Optional[bool] = None          # cross-session memory (default: config feedback.memory)
    generate: bool = True                  # False: retrieval only, no LLM call


class FeedbackRequest(BaseModel):
    session_id: str
    query_id: str
    vote: Literal[1, -1]
    target: Literal["evidence", "answer"] = "evidence"
    component_id: Optional[str] = None


# ---------------------------------------------------------------- state
@dataclass
class QueryState:
    session_id: str
    doc_id: str
    store: str
    query: str
    qtype: str
    top_k: int
    memory: bool
    answer: Dict[str, Any]
    ranked: List[Any] = field(default_factory=list)       # Candidates after feedback (best first)
    shown: List[str] = field(default_factory=list)        # component ids already shown (top-k, then /next)
    scorer: Any = None
    q_vec: Any = None                                     # BGE query vector (with instruction): Rocchio
    plain_vec: Any = None                                 # plain question vector: memory similarity
    bw: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    deltas: Dict[str, float] = field(default_factory=dict)
    classification: Dict[str, Any] = field(default_factory=dict)


class CachingReranker:
    """Cross-encoder with a shared (query, text) score cache: repeated feedback rounds re-score only new items."""

    def __init__(self, inner: Any) -> None:
        from docscout.feedback.rocchio import CachedScorer

        self.inner, self.top_n = inner, inner.top_n
        self.scorer = CachedScorer(inner.score)

    def score(self, query: str, texts: List[str]) -> List[float]:
        return self.scorer(query, texts)

    def rerank(self, query: str, ranked: Any, texts: Any) -> Any:
        from docscout.rerank.cross_encoder import rerank_candidates

        return rerank_candidates(query, ranked, texts, self.scorer, self.top_n)


class AppState:
    """Pipelines per store, document manifests, the feedback store and the per-query state."""

    def __init__(self, pipelines: Dict[str, Any], cfg: Dict[str, Any], store: Any,
                 manifests: Optional[Dict[str, Dict[str, Any]]] = None,
                 questions: Optional[Dict[str, List[str]]] = None) -> None:
        self.pipelines, self.cfg, self.store = pipelines, cfg, store
        self.manifests = manifests if manifests is not None else {}
        self.questions = questions or {}
        self.doc_store: Dict[str, str] = {}
        for name, pipe in pipelines.items():
            for d in getattr(pipe, "doc_sizes", {}):
                self.doc_store[d] = name
        self.queries: Dict[str, QueryState] = {}
        self.lock = threading.RLock()        # the pipelines are not thread-safe
        # PDF upload (one job at a time): runner(pdf_path, log) ingests + rebuilds the demo indexes;
        # reload_demo() swaps in a fresh demo-store pipeline. Both can be replaced (tests use fakes).
        self.ingest: Dict[str, Any] = {"status": "idle", "file": None, "doc_id": None, "log": [], "error": None,
                                       "started_at": None, "finished_at": None}
        self.ingest_runner: Callable[[Path, Callable[[str], None]], None] = run_ingest_subprocesses
        self.reload_demo: Callable[[], None] = lambda: reload_demo_store(self)
        self.upload_dir: Path = Path(cfg["paths"]["pdfs_dir"]) if "paths" in cfg else Path("data/raw/pdfs")

    def set_ingest(self, **kw: Any) -> None:
        with self.lock:
            self.ingest.update(kw)

    def ingest_log(self, line: str) -> None:
        log = self.ingest["log"]
        log.append(line.rstrip()[:300])
        del log[:-INGEST_LOG_LINES]

    def pipe_for(self, doc_id: str) -> Any:
        if doc_id not in self.doc_store:
            raise HTTPException(404, f"unknown document {doc_id!r}")
        return self.pipelines[self.doc_store[doc_id]]


INGEST_LOG_LINES = 40
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def run_ingest_subprocesses(pdf: Path, log: Callable[[str], None]) -> None:
    """The existing CLI steps for one uploaded PDF: ingest it into the demo store, then rebuild the demo indexes."""
    root = Path(__file__).resolve().parents[1]
    steps = [[sys.executable, "-m", "scripts.ingest", "--source", "pdf", "--store", "demo", "--docs", pdf.name],
             [sys.executable, "-m", "scripts.build_indexes", "--store", "demo", "--rebuild"]]
    for cmd in steps:
        log("$ " + " ".join(cmd[1:]))
        proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        assert proc.stdout is not None
        for line in proc.stdout:
            for part in line.replace("\r", "\n").split("\n"):  # tqdm redraws with \r
                if part.strip():
                    log(part)
        if proc.wait() != 0:
            raise RuntimeError(f"{cmd[2]} exited with code {proc.returncode}")


def reload_demo_store(state: "AppState") -> None:
    """Swap in a freshly loaded demo-store pipeline (indexes rebuilt); the eval store is untouched."""
    reranker = next(iter(state.pipelines.values())).reranker
    loaded = load_store_pipeline("demo", reranker)
    if loaded is None:
        raise RuntimeError("demo store has no components after ingest")
    pipe, manifests = loaded
    pipe.warm_up()
    with state.lock:
        state.pipelines["demo"] = pipe
        state.manifests.update(manifests)
        for d in pipe.doc_sizes:
            state.doc_store[d] = "demo"


_STORE_CFG_LOCK = threading.Lock()


def store_config(name: str) -> Dict[str, Any]:
    """Config of one store, without leaving the process switched to it (other requests keep the eval config)."""
    from docscout.config import use_store

    with _STORE_CFG_LOCK:
        cfg = use_store(name)
        use_store(None)
    return cfg


def load_store_pipeline(name: str, reranker: Any) -> Optional[tuple]:
    """(pipeline, manifests) for one store, or None if the store has no components yet."""
    from docscout.graph.builder import load_graph
    from docscout.indexing.build_all import load_index
    from docscout.pipeline import DocScoutPipeline
    from docscout.retrieval.retrievers import HybridRetriever
    from docscout.schema import Component, iter_jsonl

    cfg = store_config(name)
    if not Path(cfg["paths"]["components_file"]).exists():
        return None
    comps = {r["id"]: Component.from_dict(r) for r in iter_jsonl(cfg["paths"]["components_file"])}
    retr = HybridRetriever(load=lambda b, r, c=cfg: load_index(b, r, cfg=c, warn=False), comp_meta=comps, cfg=cfg)
    gdir = cfg["paths"]["graphs_dir"]
    pipe = DocScoutPipeline(cfg=cfg, retriever=retr, reranker=reranker, components=comps,
                            graph_loader=lambda d, g=gdir: load_graph(d, g))
    manifests: Dict[str, Dict[str, Any]] = {}
    for f in sorted(Path(cfg["paths"]["docs_dir"]).glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if m["doc_id"] in pipe.doc_sizes:
            manifests[m["doc_id"]] = {**m, "store": name}
    return pipe, manifests


def load_state() -> AppState:
    """Load both stores (eval + demo, if built), sharing the model weights (lru-cached loaders)."""
    from docscout.config import use_store
    from docscout.feedback.session_store import FeedbackStore
    from docscout.rerank.cross_encoder import CrossEncoderReranker
    from docscout.schema import iter_jsonl

    pipelines: Dict[str, Any] = {}
    manifests: Dict[str, Dict[str, Any]] = {}
    reranker = CachingReranker(CrossEncoderReranker(cfg=store_config("eval")))
    for name in ("eval", "demo"):
        loaded = load_store_pipeline(name, reranker)
        if loaded is None:
            continue
        pipelines[name], m = loaded
        manifests.update(m)
    cfg = use_store(None)
    questions: Dict[str, List[str]] = {}
    qfile = Path(cfg["paths"]["questions_file"])
    if qfile.exists():
        for q in iter_jsonl(qfile):
            questions.setdefault(q["doc_id"], []).append(q["question"])
    for d, qs in (cfg.get("api", {}).get("demo_questions") or {}).items():
        questions.setdefault(d, []).extend(qs)
    for p in pipelines.values():
        p.warm_up()
    db = Path(cfg["feedback"]["db_path"])
    store = FeedbackStore(db if db.is_absolute() else cfg["repo_root"] / db)
    return AppState(pipelines, cfg, store, manifests, questions)


# ---------------------------------------------------------------- evidence cards
def _cell_boxes(state: AppState, doc_id: str, comp: Any, cells: List[Any]) -> List[Dict[str, Any]]:
    m = state.manifests.get(doc_id, {})
    if not cells or m.get("source") != "pdf" or not m.get("pdf_path"):
        return []
    from docscout.evidence.cell_boxes import cell_boxes

    p = Path(m["pdf_path"])
    p = p if p.is_absolute() else state.cfg["repo_root"] / p
    if not p.exists():
        return []
    try:
        return cell_boxes(p, int(comp.page), comp.bbox, (comp.page_width, comp.page_height), cells)
    except Exception:  # noqa: BLE001 - highlighting is best-effort; the cells are still listed
        return []


def card_from_evidence(state: AppState, doc_id: str, e: Any) -> Dict[str, Any]:
    """JSON evidence card from a pipeline Evidence."""
    c = e.component
    cells = [[str(x) for x in cell] for cell in e.localized_cells]
    return {
        "id": c.id, "eid": e.meta.get("eid"), "rank": e.meta.get("rank"), "type": c.type, "page": int(c.page),
        "branch": e.meta.get("branch"), "bbox": list(c.bbox), "page_size": [c.page_width, c.page_height],
        "text": c.text_form[:SNIPPET_CHARS], "graph_path": e.meta.get("graph_path", ""), "why": e.why,
        "scores": e.scores, "feedback_tag": e.meta.get("feedback_tag", ""),
        "matched_terms": e.meta.get("matched_terms", []), "localized_cells": cells,
        "localize_flag": e.meta.get("localize_flag"), "cell_boxes": _cell_boxes(state, doc_id, c, cells),
        "table_preview": e.meta.get("table_preview"), "reserved_slot": e.meta.get("reserved_slot"),
        "context": [{"id": x.id, "type": x.type, "page": int(x.page), "text": x.text_form[:200]} for x in e.context],
    }


def card_from_candidate(state: AppState, qs: QueryState, cand: Any, rank: int) -> Dict[str, Any]:
    """Evidence card for an item past the top-k ("show next"), from the stored ranking: no new retrieval."""
    from docscout.context.graph_expander import expand_evidence
    from docscout.evidence.provenance import provenance_record
    from docscout.localization.table_locator import clean_table, df_markdown, locate
    from docscout.schema import Evidence

    pipe = state.pipe_for(qs.doc_id)
    comp = pipe.components[cand.id]
    e = Evidence(component=comp)
    ctx_cfg = pipe.cfg.get("context", {})
    try:
        G = pipe.graph(qs.doc_id)
        ctx_ids, path = expand_evidence([cand.id], G, int(ctx_cfg.get("max_siblings", 2)),
                                        int(ctx_cfg.get("sibling_window", 3))).get(cand.id, ([], ""))
    except Exception:  # noqa: BLE001 - no graph: no context
        ctx_ids, path = [], ""
    path = path or f"Page {comp.page} -> {comp.type.title()}"
    e.context = [pipe.components[x] for x in ctx_ids if x in pipe.components]
    if comp.type == "table":
        df = pipe.table(comp)
        if df is not None:
            if qs.qtype == "numerical":
                _, cells, flag = locate(qs.query, df)
                e.localized_cells = [tuple(x) for x in cells]
                e.meta["localize_flag"] = flag
            e.meta["table_preview"] = df_markdown(clean_table(df), int(pipe.cfg["answer"].get("max_table_rows", 6)))
    rec = provenance_record(cand, comp, qs.qtype, qs.bw, qs.query, path, qs.deltas.get(cand.id, 0.0),
                            qs.tags.get(cand.id, ""))
    e.scores = rec.pop("scores")
    e.why = rec["why"]
    e.meta.update({"eid": f"E{rank}", "rank": rank, **rec})
    return card_from_evidence(state, qs.doc_id, e)


# ---------------------------------------------------------------- core
def _answer_payload(res: Any) -> Dict[str, Any]:
    m = res.meta
    cited = m.get("citations") or []
    return {
        "text": res.answer, "not_found": bool(m.get("not_found")), "citations": cited,
        "confidence": {"cited_sources": len(cited), "not_found": bool(m.get("not_found")),
                       "slot_used": bool(m.get("slots", {}).get("reserved")),
                       "slot_variant": m.get("slots", {}).get("variant"),
                       "invalid_citations": len(m.get("answer_validation", {}).get("invalid_citations") or [])},
        "model": m.get("model"), "llm": m.get("llm"),
    }


def _run(state: AppState, qs: QueryState, generate: bool) -> Any:
    """Run the pipeline for a query state with the session's votes (Rocchio) and the memory prior."""
    from docscout.feedback.memory import memory_prior
    from docscout.feedback.rocchio import make_feedback_hook

    pipe = state.pipe_for(qs.doc_id)
    if not qs.qtype:  # the memory is gated on the query type, so classify before looking it up
        try:
            qs.classification = pipe._classify(qs.query, client=None if pipe.model else pipe._client)
        except Exception as e:  # noqa: BLE001 - LLM unreachable: rules / unsure fallback (as the pipeline)
            qs.classification = pipe._classify(qs.query, use_llm=False)
            qs.classification["llm_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        qs.qtype = qs.classification["type"]
    votes = state.store.session_votes(qs.session_id, qs.doc_id, qs.query)
    prior = memory_prior(state.store, qs.doc_id, qs.qtype, qs.plain_vec, pipe.cfg, exclude_session=qs.session_id) \
        if qs.memory and qs.plain_vec is not None else {}
    if votes and qs.q_vec is None:
        qs.q_vec = pipe.retriever._encode("dense_bge", qs.query)
    hook = make_feedback_hook(pipe, qs.query, votes, memory=prior, scorer=qs.scorer, q_vec=qs.q_vec)
    qs.scorer = hook.scorer
    res = pipe.answer(qs.query, qs.doc_id, session_id=qs.session_id, top_k=qs.top_k, feedback_adjust=hook,
                      generate=generate, qtype=qs.qtype)
    qs.ranked, qs.tags, qs.deltas = hook.ranked, dict(hook.tags), dict(hook.deltas)
    qs.bw = dict(res.meta["branch_weights"])
    qs.shown = [e.component.id for e in res.evidence]
    return res, votes, prior


def _response(state: AppState, qid: str, qs: QueryState, res: Any, votes: Dict[str, int],
              prior: Dict[str, float]) -> Dict[str, Any]:
    m = res.meta
    return {
        "query_id": qid, "session_id": qs.session_id, "doc_id": qs.doc_id, "store": qs.store, "query": qs.query,
        "query_type": qs.qtype, "classification": qs.classification, "branch_weights": m.get("branch_weights"),
        "answer": qs.answer, "evidence": [card_from_evidence(state, qs.doc_id, e) for e in res.evidence],
        "ranking_size": len(qs.ranked), "votes": votes, "memory": {"on": qs.memory, "prior_items": len(prior)},
        "feedback_active": bool(votes or prior), "timings_ms": m.get("timings_ms"),
    }


def create_app(state: Optional[AppState] = None) -> FastAPI:
    """App factory; tests pass a ready AppState (fake pipeline), the server loads the real one at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "docscout", None) is None:
            app.state.docscout = load_state()
        yield

    app = FastAPI(title="DocScout-X API", version="0.7", lifespan=lifespan)
    app.state.docscout = state
    from docscout.config import get_config

    origins = (get_config().get("api") or {}).get("cors_origins") or ["*"]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    def S() -> AppState:
        return app.state.docscout

    @app.get("/health")
    def health() -> Dict[str, Any]:
        st = S()
        return {"ok": True, "stores": {k: len(getattr(p, "doc_sizes", {})) for k, p in st.pipelines.items()},
                "votes": st.store.count()}

    @app.get("/documents")
    def documents() -> List[Dict[str, Any]]:
        st = S()
        out = []
        for d, name in sorted(st.doc_store.items(), key=lambda x: (x[1] != "demo", x[0])):
            m = st.manifests.get(d, {})
            out.append({"doc_id": d, "store": name, "source": m.get("source"),
                        "pages": [{"page": p["page"], "width": p["width"], "height": p["height"]}
                                  for p in m.get("pages", [])],
                        "components": st.pipelines[name].doc_sizes.get(d, 0),
                        "sample_questions": st.questions.get(d, [])[:6]})
        return out

    @app.get("/page-image/{doc_id}/{page}")
    def page_image(doc_id: str, page: int) -> FileResponse:
        st = S()
        m = st.manifests.get(doc_id)
        if m is None:
            raise HTTPException(404, f"unknown document {doc_id!r}")
        hit = next((p for p in m.get("pages", []) if int(p["page"]) == page), None)
        if hit is None:
            raise HTTPException(404, f"{doc_id} has no page {page}")
        p = Path(hit["image_path"])
        p = p if p.is_absolute() else st.cfg["repo_root"] / p
        if not p.exists():
            raise HTTPException(404, "page image missing on disk")
        return FileResponse(p, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.post("/ask")
    def ask(req: AskRequest) -> Dict[str, Any]:
        st = S()
        st.pipe_for(req.doc_id)
        sid = req.session_id or uuid.uuid4().hex
        memory = bool(st.cfg["feedback"].get("memory")) if req.memory is None else req.memory
        pipe = st.pipe_for(req.doc_id)
        with st.lock:
            plain = None
            if memory and hasattr(pipe.retriever, "indexes"):
                from docscout.feedback.memory import question_vector

                plain = question_vector(pipe.retriever, req.query)
            qs = QueryState(sid, req.doc_id, st.doc_store[req.doc_id], req.query, "", req.top_k, memory, {},
                            plain_vec=plain)
            res, votes, prior = _run(st, qs, req.generate)
            qs.answer = _answer_payload(res)
            qid = uuid.uuid4().hex
            st.queries[qid] = qs
            return _response(st, qid, qs, res, votes, prior)

    @app.post("/feedback")
    def feedback(req: FeedbackRequest) -> Dict[str, Any]:
        st = S()
        qs = st.queries.get(req.query_id)
        if qs is None or qs.session_id != req.session_id:
            raise HTTPException(404, "unknown query_id for this session")
        if req.target == "evidence" and not req.component_id:
            raise HTTPException(422, "an evidence vote needs component_id")
        with st.lock:
            pipe = st.pipe_for(qs.doc_id)
            if qs.plain_vec is None and hasattr(pipe.retriever, "indexes"):
                from docscout.feedback.memory import question_vector

                qs.plain_vec = question_vector(pipe.retriever, qs.query)
            st.store.record(qs.session_id, qs.doc_id, qs.query, qs.qtype, req.component_id, req.vote,
                            target=req.target, query_vec=qs.plain_vec)
            if req.target == "answer":  # logged only, never used for ranking
                return {"recorded": True, "target": "answer", "query_id": req.query_id}
            t0 = time.perf_counter()
            res, votes, prior = _run(st, qs, generate=False)  # re-rank only; the answer is not regenerated
            out = _response(st, req.query_id, qs, res, votes, prior)
            out.update({"recorded": True, "target": "evidence", "answer_stale": True,
                        "rerank_ms": round((time.perf_counter() - t0) * 1000, 1)})
            return out

    @app.get("/next")
    def next_item(session_id: str = Query(...), query_id: str = Query(...)) -> Dict[str, Any]:
        st = S()
        qs = st.queries.get(query_id)
        if qs is None or qs.session_id != session_id:
            raise HTTPException(404, "unknown query_id for this session")
        with st.lock:
            for i, c in enumerate(qs.ranked):
                if c.id not in qs.shown:
                    qs.shown.append(c.id)
                    return {"item": card_from_candidate(st, qs, c, len(qs.shown)), "position": i + 1,
                            "remaining": sum(1 for x in qs.ranked if x.id not in qs.shown)}
            return {"item": None, "message": "no further matches", "remaining": 0}

    # ------------------------------------------------------------ PDF upload (added alongside /documents)
    def _ingest_job(path: Path, doc_id: str) -> None:
        st = S()
        st.set_ingest(status="running")
        try:
            st.ingest_runner(path, st.ingest_log)
            st.ingest_log("reloading the demo store ...")
            st.reload_demo()
            if doc_id not in st.doc_store:
                raise RuntimeError(f"{doc_id} was not ingested (see the log)")
            st.ingest_log(f"done: {doc_id} is available")
            st.set_ingest(status="done", finished_at=time.time())
        except Exception as e:  # noqa: BLE001 - reported through /ingest/status
            st.ingest_log(f"error: {type(e).__name__}: {e}")
            st.set_ingest(status="error", error=f"{type(e).__name__}: {e}", finished_at=time.time())

    @app.post("/ingest", status_code=202)
    def ingest(file: UploadFile = File(...)) -> Dict[str, Any]:
        from docscout.ingestion.parser import safe_doc_id

        st = S()
        name = Path(file.filename or "").name
        if not name.lower().endswith(".pdf"):
            raise HTTPException(415, "only .pdf files are accepted")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if not data.startswith(b"%PDF"):
            raise HTTPException(415, "the file is not a PDF")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"PDF larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        doc_id = safe_doc_id(Path(name).stem)
        with st.lock:
            if st.ingest["status"] in ("queued", "running"):
                raise HTTPException(409, f"an upload is already being processed ({st.ingest['file']})")
            if doc_id in st.doc_store:
                raise HTTPException(409, f"document {doc_id!r} already exists")
            st.upload_dir.mkdir(parents=True, exist_ok=True)
            path = st.upload_dir / name
            if path.exists():
                raise HTTPException(409, f"{name} already exists in {st.upload_dir.name}/")
            path.write_bytes(data)
            st.ingest = {"status": "queued", "file": name, "doc_id": doc_id, "log": [], "error": None,
                         "started_at": time.time(), "finished_at": None}
        threading.Thread(target=_ingest_job, args=(path, doc_id), daemon=True).start()
        return {"status": "queued", "file": name, "doc_id": doc_id}

    @app.get("/ingest/status")
    def ingest_status() -> Dict[str, Any]:
        st = S()
        j = dict(st.ingest)
        j["log_tail"] = list(j.pop("log"))[-15:]
        j["elapsed_s"] = round((j["finished_at"] or time.time()) - j["started_at"], 1) if j["started_at"] else None
        return j

    return app


app = create_app()
