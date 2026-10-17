"""Tests for BM25 / dense / index bookkeeping (fast: fake encoder; one real-model test uses cached MiniLM)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from docscout.indexing.base import file_fingerprint
from docscout.indexing.bm25_index import BM25Index, tokenize
from docscout.indexing.build_all import index_status, stale_indexes
from docscout.indexing.dense_index import DenseIndex
from docscout.schema import Component


def _c(cid, doc, text, ctype="paragraph"):
    return Component(id=cid, doc_id=doc, page=1, type=ctype, text=text, text_form=text)


CORPUS = [
    _c("a_p1_paragraph_0", "a", "The cat sat on the mat near the window."),
    _c("a_p1_paragraph_1", "a", "Quarterly revenue increased by twelve percent compared with last year."),
    _c("a_p1_paragraph_2", "a", "The weather was rainy and cold all week."),
    _c("b_p1_paragraph_0", "b", "Revenue table for 2023: total 1,234 million."),
    _c("b_p1_paragraph_1", "b", "Our cat adoption programme expanded in 2023."),
]


class FakeEncoder:
    """Deterministic bag-of-words hashing encoder; records every string it encodes."""

    def __init__(self, dim=64):
        self.dim, self.seen = dim, []

    def encode(self, texts, **kw):
        self.seen.extend(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t) or ["empty"]:
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return out / np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-9)


# ------------------------------------------------------------------ tokenizer / BM25

def test_tokenizer_numbers_punctuation_stopwords():
    assert tokenize("The Revenue (2023) was 1,234.5 & 26.9%!") == ["revenue", "2023", "1234.5", "26.9"]
    assert tokenize("U.S.-based co-op") == ["u", "based", "co", "op"]


def test_bm25_keyword_match_first():
    idx = BM25Index().build(CORPUS)
    res = idx.search("weather rainy", top_k=3)
    assert res[0][0] == "a_p1_paragraph_2"
    assert all(score > 0 for _, score in res)


def test_bm25_doc_filter_exact():
    idx = BM25Index().build(CORPUS)
    res = idx.search("cat 2023 revenue", top_k=10, doc_id="b")
    assert res and all(cid.startswith("b_") for cid, _ in res)
    assert idx.search("cat", doc_id="zzz") == []


def test_bm25_save_load_roundtrip(tmp_path: Path):
    idx = BM25Index().build(CORPUS)
    idx.save(tmp_path / "bm25")
    again = BM25Index.load(tmp_path / "bm25")
    assert again.search("revenue 2023", top_k=5) == idx.search("revenue 2023", top_k=5)


# ------------------------------------------------------------------ dense (fake encoder)

def test_bge_instruction_only_on_queries():
    enc = FakeEncoder()
    idx = DenseIndex("BAAI/bge-base-en-v1.5", query_instruction="Represent this sentence: ", model=enc).build(CORPUS, progress=False)
    assert enc.seen == [c.text_form for c in CORPUS]  # documents: plain text_form
    enc.seen.clear()
    idx.search("revenue growth", top_k=2)
    assert enc.seen == ["Represent this sentence: revenue growth"]  # query: with instruction


def test_dense_doc_filter_matches_brute_force(tmp_path: Path):
    enc = FakeEncoder()
    idx = DenseIndex("fake", query_instruction="", model=enc).build(CORPUS, progress=False)
    res = idx.search("cat revenue 2023", top_k=10, doc_id="b")
    assert [cid for cid, _ in res] and all(cid.startswith("b_") for cid, _ in res)
    q = enc.encode(["cat revenue 2023"])[0]
    brute = sorted(((c.id, float(idx.emb[i] @ q)) for i, c in enumerate(CORPUS) if c.doc_id == "b"), key=lambda x: -x[1])
    assert [cid for cid, _ in res] == [cid for cid, _ in brute]
    np.testing.assert_allclose([s for _, s in res], [s for _, s in brute], rtol=1e-5)
    # unfiltered FAISS search agrees with brute force on the top hit
    top_all = max(((c.id, float(idx.emb[i] @ q)) for i, c in enumerate(CORPUS)), key=lambda x: x[1])
    assert idx.search("cat revenue 2023", top_k=1)[0][0] == top_all[0]
    idx.save(tmp_path / "dense")
    again = DenseIndex.load(tmp_path / "dense", model=enc)
    assert again.search("cat revenue 2023", top_k=3, doc_id="b") == res


def _local_minilm():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu", local_files_only=True)
    except Exception:  # noqa: BLE001
        return None


def test_dense_ranks_paraphrase_first_bm25_does_not():
    model = _local_minilm()
    if model is None:
        pytest.skip("all-MiniLM-L6-v2 not in the local HF cache")
    query = "sales went up this quarter"  # no content word shared with the revenue sentence
    target = "a_p1_paragraph_1"
    assert not set(tokenize(query)) & set(tokenize(CORPUS[1].text_form))
    dense = DenseIndex("sentence-transformers/all-MiniLM-L6-v2", query_instruction="", model=model).build(CORPUS[:3], progress=False)
    assert dense.search(query, top_k=3)[0][0] == target
    bm25 = BM25Index().build(CORPUS[:3])
    assert target not in [cid for cid, _ in bm25.search(query, top_k=3)]


# ------------------------------------------------------------------ fingerprint / staleness

def test_stale_detection(tmp_path: Path):
    comp = tmp_path / "components.jsonl"
    comp.write_text('{"id": "x"}\n', encoding="utf-8")
    cfg = {"paths": {"components_file": comp, "indexes_dir": tmp_path / "idx"}, "branches": {"text": ["paragraph"]}}
    fp = file_fingerprint(comp)
    assert fp["lines"] == 1 and len(fp["sha256"]) == 64
    assert index_status("text", "bm25", fp, cfg) == "missing"
    d = tmp_path / "idx" / "text" / "bm25"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"fingerprint": fp}), encoding="utf-8")
    assert index_status("text", "bm25", fp, cfg) == "ok" and stale_indexes(cfg) == []
    with comp.open("a", encoding="utf-8") as f:  # another document ingested later
        f.write('{"id": "y"}\n')
    assert stale_indexes(cfg) == ["text/bm25"]
