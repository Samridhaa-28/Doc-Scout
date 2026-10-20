"""Normalization, weighted-sum / RRF fusion (orders worked out by hand), retriever scoring and rerank cut-off."""
from types import SimpleNamespace

import numpy as np
import pytest

from docscout.rerank.cross_encoder import rerank_candidates
from docscout.retrieval.fusion import effective_retriever_weights, hard_route, rank_candidates, rrf, weighted_sum
from docscout.retrieval.normalize import minmax, normalize_candidates, zscore
from docscout.retrieval.retrievers import Candidate, HybridRetriever


# ---------- normalize
def test_minmax_basic():
    assert minmax({"a": 2.0, "b": 4.0, "c": 3.0}) == {"a": 0.0, "b": 1.0, "c": 0.5}


def test_minmax_all_equal_and_single_and_empty():
    assert minmax({"a": 2.0, "b": 2.0, "c": 2.0}) == {"a": 0.0, "b": 0.0, "c": 0.0}
    assert minmax({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}  # e.g. no BM25 term match anywhere
    assert minmax({"only": 7.3}) == {"only": 1.0}
    assert minmax({}) == {}


def test_zscore():
    z = zscore({"a": 1.0, "b": 3.0})
    assert z == pytest.approx({"a": -1.0, "b": 1.0})
    assert zscore({"a": 5.0, "b": 5.0}) == {"a": 0.0, "b": 0.0}
    assert zscore({"a": 5.0}) == {"a": 1.0}


# ---------- hand-made candidates (one document)
def make_cands():
    def c(cid, branch, typ, **raw):
        return Candidate(cid, "d", 1, typ, branch, raw=dict(raw))
    return [
        c("t1", "text", "paragraph", bm25=4.0, dense_bge=0.75),
        c("t2", "text", "paragraph", bm25=0.0, dense_bge=0.5),
        c("t3", "text", "heading", bm25=2.0, dense_bge=0.25),
        c("tb1", "table", "table", bm25=1.0, dense_bge=0.5),  # the only table -> norm 1.0 from both retrievers
        c("f1", "figure", "figure", bm25=0.0, dense_bge=0.3, clip=0.25),
        c("f2", "figure", "figure", bm25=3.0, dense_bge=0.2, clip=0.30),
    ]


RW = {"bm25": 0.5, "dense": 0.5, "clip": 0.3}
BW = {"textual": {"text": 0.6, "table": 0.25, "figure": 0.15},
      "numerical": {"text": 0.3, "table": 0.6, "figure": 0.1},
      "visual": {"text": 0.25, "table": 0.15, "figure": 0.6}}


def test_normalize_candidates_per_branch_retriever_list():
    cs = {c.id: c for c in normalize_candidates(make_cands())}
    assert cs["t1"].norm == {"bm25": 1.0, "dense_bge": 1.0}
    assert cs["t2"].norm == {"bm25": 0.0, "dense_bge": 0.5}
    assert cs["t3"].norm == pytest.approx({"bm25": 0.5, "dense_bge": 0.0})
    assert cs["tb1"].norm == {"bm25": 1.0, "dense_bge": 1.0}
    assert cs["f1"].norm == pytest.approx({"bm25": 0.0, "dense_bge": 1.0, "clip": 0.0})
    assert cs["f2"].norm == pytest.approx({"bm25": 1.0, "dense_bge": 0.0, "clip": 1.0})


def test_retriever_weights_renormalized_per_branch():
    assert effective_retriever_weights(["bm25", "dense_bge"], RW) == {"bm25": 0.5, "dense_bge": 0.5}
    fig = effective_retriever_weights(["bm25", "clip", "dense_bge"], RW)
    assert fig == pytest.approx({"bm25": 0.5 / 1.3, "dense_bge": 0.5 / 1.3, "clip": 0.3 / 1.3})


def test_weighted_sum_textual_order_by_hand():
    # t1 = .6*(.5*1+.5*1) = .600 | tb1 = .25*1 = .250 | t2 = .6*(.5*0+.5*.5) = .150 | t3 = .6*(.5*.5+0) = .150
    # f2 = .15*(1/1.3*.5 + .3/1.3*1) = .0923 | f1 = .15*(.5/1.3*1) = .0577 ; t2/t3 tie -> id order
    ranked = weighted_sum(normalize_candidates(make_cands()), RW, BW["textual"])
    assert [c.id for c in ranked] == ["t1", "tb1", "t2", "t3", "f2", "f1"]
    s = {c.id: c.fused for c in ranked}
    assert s["t1"] == pytest.approx(0.6) and s["tb1"] == pytest.approx(0.25)
    assert s["f2"] == pytest.approx(0.15 * (0.5 / 1.3 + 0.3 / 1.3)) and s["f1"] == pytest.approx(0.15 * 0.5 / 1.3)


def test_weighted_sum_numerical_lone_table_wins():
    # min-max gives the only table 1.0 -> tb1 = .6 beats the best paragraph t1 = .3
    ranked = weighted_sum(normalize_candidates(make_cands()), RW, BW["numerical"])
    assert [c.id for c in ranked][:2] == ["tb1", "t1"]


def test_rrf_order_by_hand_and_bm25_zero_not_ranked():
    # lists (rank by raw; text bge = .75,.5,.25): text/bm25: t1,t3 (t2 has bm25=0 -> unranked) | text/bge: t1,t2,t3
    # table/bm25: tb1 | table/bge: tb1 | figure/bm25: f2 | figure/bge: f1,f2 | figure/clip: f2,f1
    # t1 = .6/61+.6/61 = .01967 | t3 = .6/62+.6/63 = .01920 | t2 = .6/62 = .00968 | tb1 = 2*.25/61 = .00820
    # f2 = .15/61+.15/62+.15/61 = .00734 | f1 = .15/61+.15/62 = .00488
    ranked = rrf(make_cands(), BW["textual"], k=60)
    assert [c.id for c in ranked] == ["t1", "t3", "t2", "tb1", "f2", "f1"]
    s = {c.id: c.rrf for c in ranked}
    assert s["t3"] == pytest.approx(0.6 / 62 + 0.6 / 63)
    assert s["t2"] == pytest.approx(0.6 / 62)


def test_rrf_uses_ranks_only():
    a = rrf(make_cands(), BW["textual"])
    scaled = make_cands()
    for c in scaled:
        c.raw = {r: v * 100 for r, v in c.raw.items()}
    b = rrf(scaled, BW["textual"])
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.rrf for c in a] == pytest.approx([c.rrf for c in b])
    assert all(c.norm == {} for c in b)  # RRF never normalizes


def test_ties_broken_by_component_id():
    cs = [Candidate(cid, "d", 1, "paragraph", "text", raw={"bm25": 1.0, "dense_bge": 0.5}) for cid in ("b", "c", "a")]
    ranked = weighted_sum(normalize_candidates(cs), RW, BW["textual"])
    assert [c.id for c in ranked] == ["a", "b", "c"]


def test_hard_vs_soft_routing():
    soft = rank_candidates(make_cands(), "visual", "weighted_sum", "soft", RW, BW)
    hard = rank_candidates(make_cands(), "visual", "weighted_sum", "hard", RW, BW)
    assert len(soft) == 6
    assert [c.id for c in hard] == ["f2", "f1"]
    assert hard_route(make_cands(), "table")[0].id == "tb1"


# ---------- rerank cut-off
def test_rerank_keeps_tail_in_fused_order():
    fused = [Candidate(f"c{i:02d}", "d", 1, "paragraph", "text", fused=1 - i / 100, final=1 - i / 100) for i in range(35)]
    order_before = [c.id for c in fused]
    texts = {c.id: c.id for c in fused}
    scores = lambda q, ts: [float(int(t[1:])) for t in ts]  # noqa: E731  reverses the head: c29 best
    out = rerank_candidates("q", fused, texts, scores, top_n=30)
    ids = [c.id for c in out]
    assert ids[:30] == list(reversed(order_before[:30]))
    assert ids[30:] == order_before[30:]                     # below the top 30: fused order kept
    assert all(c.rerank is None for c in out[30:]) and out[0].rerank == 29.0
    finals = [c.final for c in out]
    assert all(a > b for a, b in zip(finals, finals[1:]))    # strictly decreasing (page max stays consistent)


# ---------- retriever with fake indexes
class FakeBM25:
    def get_batch_scores(self, tokens, rows):
        table = {0: 2.0, 1: 0.0, 2: 1.0}
        return [table[r] for r in rows]


class FakeIndex:
    def __init__(self, ids, docs, emb=None, bm25=None):
        self.ids, self.bm25, self.emb = ids, bm25, emb
        self.doc_rows = {}
        for i, d in enumerate(docs):
            self.doc_rows.setdefault(d, []).append(i)
        self.doc_rows = {d: np.asarray(r) for d, r in self.doc_rows.items()}
        self.encoded = 0

    def encode_query(self, q):
        self.encoded += 1
        return np.asarray([1.0, 0.0], dtype=np.float32)


def test_retriever_scores_every_component_and_keeps_bm25_zero():
    ids, docs = ["d_a", "d_b", "e_c"], ["d", "d", "e"]
    emb = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]], dtype=np.float32)
    bge = FakeIndex(ids, docs, emb=emb)
    idx = {("text", "bm25"): FakeIndex(ids, docs, bm25=FakeBM25()), ("text", "dense_bge"): bge}
    meta = {i: SimpleNamespace(doc_id=d, page=1, type="paragraph") for i, d in zip(ids, docs)}
    r = HybridRetriever(load=lambda b, ret: idx.get((b, ret)), comp_meta=meta,
                        branch_retrievers={"text": ["bm25", "dense_bge"]}, cfg={"branches": {"text": ["paragraph"]}})
    cands = {c.id: c for c in r.retrieve("some query", "d")}
    assert set(cands) == {"d_a", "d_b"}                       # only the searched document
    assert cands["d_b"].raw["bm25"] == 0.0                    # no term match -> kept with 0
    assert cands["d_a"].raw["dense_bge"] == pytest.approx(0.9)
    assert bge.encoded == 1                                   # query encoded once per model
