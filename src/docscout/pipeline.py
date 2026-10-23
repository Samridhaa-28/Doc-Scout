"""DocScoutPipeline: one query over one document, end to end.

Steps (timed separately, in this order):
  1. classify        query type (rules, LLM when unsure)
  2. retrieve        every component of the document, raw scores per retriever (BM25, BGE, CLIP for figures)
  3. normalize       min-max per (branch, retriever) list
  4. fuse            weighted sum with the query type's branch weights (soft routing)
  5. rerank          cross-encoder on the top retrieval.rerank_top_n
  6. feedback        feedback_adjust(ranked, qtype, session_id) may change c.final; the list is re-sorted
                     (identity by default; P7 adds the session preferences)
  7. localize        numerical queries only: table_locator on the first localization.top_k_tables tables
                     among the top-k evidence items
  8. expand          graph context (section heading, siblings, caption / linked table-figure), de-duplicated
  9. answer          LLM over the numbered evidence only (validated JSON, citations)
 10. provenance      scores per stage, matched terms, graph path, bbox, why-line

Models, indexes and components are loaded once in __init__; graphs and tables are cached per document.
QueryResult.meta has the full reranked list ("show next" in P7), timings per step, LLM stats and the model.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import pandas as pd

from docscout.config import get_config
from docscout.retrieval.retrievers import Candidate
from docscout.schema import Component, Evidence, QueryResult

FeedbackAdjust = Callable[[List[Candidate], str, Optional[str]], Optional[List[Candidate]]]


def identity_feedback(ranked: List[Candidate], qtype: str, session_id: Optional[str]) -> List[Candidate]:
    """Default feedback hook: no change."""
    return ranked


class DocScoutPipeline:
    """Loads everything once; answer() runs the 10 steps for one (query, document)."""

    def __init__(self, cfg: Optional[dict] = None, retriever: Any = None, reranker: Any = None,
                 client: Any = None, components: Optional[Mapping[str, Component]] = None,
                 graph_loader: Optional[Callable[[str], Any]] = None,
                 table_loader: Optional[Callable[[Component], pd.DataFrame]] = None,
                 classify_fn: Optional[Callable[..., Dict[str, Any]]] = None, model: Optional[str] = None) -> None:
        self.cfg = cfg or get_config()
        if components is None:
            from docscout.schema import iter_jsonl

            components = {r["id"]: Component.from_dict(r) for r in iter_jsonl(self.cfg["paths"]["components_file"])}
        self.components: Mapping[str, Component] = components
        if retriever is None:
            from docscout.retrieval.retrievers import HybridRetriever

            retriever = HybridRetriever(comp_meta=self.components, cfg=self.cfg)
        self.retriever = retriever
        if reranker is None:
            from docscout.rerank.cross_encoder import CrossEncoderReranker

            reranker = CrossEncoderReranker(cfg=self.cfg)
        self.reranker = reranker
        self._client = client
        self.model = model
        if graph_loader is None:
            from docscout.graph.builder import load_graph

            graph_loader = load_graph
        self._graph_loader = graph_loader
        self._table_loader = table_loader or self._read_table
        if classify_fn is None:
            from docscout.query.classifier import classify

            classify_fn = classify
        self._classify = classify_fn
        self._graphs: Dict[str, Any] = {}
        self._tables: Dict[str, pd.DataFrame] = {}
        self.texts = {cid: c.text_form for cid, c in self.components.items()}
        self.doc_sizes: Dict[str, int] = {}
        for c in self.components.values():
            self.doc_sizes[c.doc_id] = self.doc_sizes.get(c.doc_id, 0) + 1

    def warm_up(self) -> None:
        """Load the query encoders (BGE, CLIP) and the cross-encoder now, so the first query is not slowed down."""
        for r in ("dense_bge", "clip"):
            if hasattr(self.retriever, "_encode"):
                self.retriever._encode(r, "warm up")
        if hasattr(self.reranker, "score"):
            self.reranker.score("warm up", ["warm up"])

    # ------------------------------------------------------------------ loaders
    @property
    def client(self) -> Any:
        if self._client is None:
            from docscout.reasoning.llm_client import get_client

            self._client = get_client(model=self.model) if self.model else get_client()
        return self._client

    def graph(self, doc_id: str) -> Any:
        if doc_id not in self._graphs:
            self._graphs[doc_id] = self._graph_loader(doc_id)
        return self._graphs[doc_id]

    def _read_table(self, comp: Component) -> pd.DataFrame:
        p = Path(comp.table_path)
        return pd.read_parquet(p if p.is_absolute() else self.cfg["repo_root"] / p)

    def table(self, comp: Component) -> Optional[pd.DataFrame]:
        if not comp.table_path:
            return None
        if comp.id not in self._tables:
            try:
                self._tables[comp.id] = self._table_loader(comp)
            except Exception:  # noqa: BLE001 - missing / unreadable parquet: show text_form instead
                return None
        return self._tables[comp.id]

    # ------------------------------------------------------------------ main
    def answer(self, query: str, doc_id: str, session_id: Optional[str] = None, top_k: int = 5,
               feedback_adjust: Optional[FeedbackAdjust] = None, *, use_context: bool = True,
               use_localization: bool = True, generate: bool = True, qtype: Optional[str] = None,
               slots: Optional[str] = None) -> QueryResult:
        """Run the pipeline. `qtype` skips the classifier (evaluation with precomputed types);
        `use_context` / `use_localization` / `generate` switch steps 7-9 off for ablations.
        `slots`: modality slot reservation variant "off" | "conditional" | "both" (default answer.slots, "off");
        it only changes which top_k items become evidence, never meta["reranked"]."""
        from docscout.context.graph_expander import expand_evidence
        from docscout.evidence.provenance import provenance_record
        from docscout.localization.table_locator import clean_table, df_markdown, locate
        from docscout.reasoning.answerer import answer_question
        from docscout.retrieval.fusion import sort_candidates, weighted_sum
        from docscout.retrieval.normalize import normalize_candidates

        cfg = self.cfg
        fcfg = cfg["fusion"]
        t: Dict[str, float] = {}

        def tick(name: str, t0: float) -> float:
            now = time.perf_counter()
            t[name] = round((now - t0) * 1000, 1)
            return now

        t0 = time.perf_counter()
        if qtype is None:
            # the classifier creates the default client itself only when the rules are unsure
            # (with --demo it still uses the default model, so its cache is shared with evaluation)
            try:
                cls = self._classify(query, client=None if self.model else self._client)
            except Exception as e:  # noqa: BLE001 - LLM unreachable: rules / unsure fallback, never crash
                cls = self._classify(query, use_llm=False)
                cls["llm_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        else:
            cls = {"type": qtype, "confidence": 1.0, "source": "given"}
        qtype = cls["type"]
        bw = {b: float(w) for b, w in fcfg["branch_weights"][qtype].items()}
        t0 = tick("classify", t0)

        cands = self.retriever.retrieve(query, doc_id)
        t0 = tick("retrieve", t0)
        normalize_candidates(cands, fcfg.get("normalization", "minmax"))
        t0 = tick("normalize", t0)
        ranked = weighted_sum(cands, fcfg["retriever_weights"], bw)
        t0 = tick("fuse", t0)
        ranked = self.reranker.rerank(query, ranked, self.texts)
        t0 = tick("rerank", t0)

        before = {c.id: c.final for c in ranked}
        adj = feedback_adjust or identity_feedback
        out = adj(ranked, qtype, session_id)
        ranked = sort_candidates(out if out is not None else ranked)
        fb_delta = {c.id: c.final - before.get(c.id, c.final) for c in ranked}
        t0 = tick("feedback", t0)

        from docscout.retrieval.slots import reserve_modality_slots, slot_modalities

        slot_variant = slots or cfg["answer"].get("slots", "off")
        top, reserved = reserve_modality_slots(ranked, top_k, slot_modalities(slot_variant, qtype))
        evidence = [Evidence(component=self.components[c.id]) for c in top]
        loc_info: Dict[str, str] = {}
        n_loc = int(cfg.get("localization", {}).get("top_k_tables", 3))
        max_rows = int(cfg["answer"].get("max_table_rows", 6))
        n_tables = 0
        for c, e in zip(top, evidence):
            if e.component.type != "table":
                continue
            df = self.table(e.component)
            if df is None:
                continue
            n_tables += 1  # localize the first n_loc tables among the top-k evidence items
            if use_localization and qtype == "numerical" and n_tables <= n_loc:
                sub, cells, flag = locate(query, df)
                e.localized_cells = [tuple(x) for x in cells]  # type: ignore[misc]
                e.meta["localize_flag"] = flag
                loc_info[c.id] = flag
            e.meta["table_preview"] = df_markdown(clean_table(df), max_rows)
        t0 = tick("localize", t0)

        try:
            G = self.graph(doc_id)
        except Exception:  # noqa: BLE001 - no graph file: no context, path from metadata
            G = None
        ctx_map = expand_evidence([c.id for c in top], G, int(cfg.get("context", {}).get("max_siblings", 2)),
                                  int(cfg.get("context", {}).get("sibling_window", 3))) if G is not None else {}
        for c, e in zip(top, evidence):
            ctx_ids, path = ctx_map.get(c.id, ([], ""))
            e.meta["graph_path"] = path or f"Page {e.component.page} -> {e.component.type.title()}"
            if use_context:
                e.context = [self.components[x] for x in ctx_ids if x in self.components]
        t0 = tick("expand", t0)

        stats_before = dict(self.client.stats.as_dict()) if generate else {}
        if generate:
            ans = answer_question(query, qtype, evidence, self.client, cfg)
        else:
            from docscout.reasoning.answerer import AnswerResult

            ans = AnswerResult(answer="", not_found=True, error="generation off")
        stats_after = dict(self.client.stats.as_dict()) if generate else {}
        t0 = tick("answer", t0)

        for i, (c, e) in enumerate(zip(top, evidence), start=1):
            rec = provenance_record(c, e.component, qtype, bw, query, e.meta["graph_path"], fb_delta.get(c.id, 0.0))
            e.scores = rec.pop("scores")
            e.why = rec["why"]
            e.meta.update({"eid": f"E{i}", "rank": i, **rec})
            e.meta["context_ids"] = [x.id for x in e.context]
            if c.id in reserved:
                e.meta["reserved_slot"] = reserved[c.id]
                e.why += f" | slot: best fused {reserved[c.id]}"
        t0 = tick("provenance", t0)

        llm_stats = {k: stats_after.get(k, 0) - stats_before.get(k, 0)
                     for k in ("calls", "cache_hits", "retries", "prompt_tokens", "completion_tokens", "total_tokens")}
        meta = {
            "doc_id": doc_id, "session_id": session_id, "classification": cls,
            "branch_weights": bw, "citations": ans.citations, "not_found": ans.not_found,
            "answer_validation": {"raw_answer": ans.raw_answer, "invalid_citations": ans.invalid_citations,
                                  "no_valid_citation": ans.no_valid_citation, "error": ans.error},
            "prompt": ans.prompt_info, "localization": loc_info,
            "reranked": [{"id": c.id, "page": c.page, "type": c.type, "branch": c.branch, "final": c.final,
                          "rerank": c.rerank, "fused": c.fused} for c in ranked],
            "timings_ms": {**t, "total": round(sum(t.values()), 1)},
            "llm": llm_stats,
            "model": getattr(self._client, "model", None) if generate else None,
            "slots": {"variant": slot_variant, "reserved": reserved},
            "doc_components": self.doc_sizes.get(doc_id, 0),
        }
        return QueryResult(query=query, query_type=qtype, answer=ans.answer, evidence=evidence, meta=meta)
