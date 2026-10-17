"""Build / load all branch indexes from components.jsonl.

Layout: data/indexes/{branch}/{retriever}/   (meta.json + index files)
  branches   : text (paragraph, heading, caption), table (table), figure (figure)   [config `branches`]
  retrievers : bm25, dense_bge, dense_minilm for every branch; + clip for the figure branch

Each index stores a fingerprint of components.jsonl (line count + sha256). If the file changes
(more documents ingested), the index is reported as STALE and must be rebuilt with --rebuild.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from docscout.config import get_config
from docscout.indexing.base import BaseIndex, Timer, file_fingerprint
from docscout.indexing.bm25_index import BM25Index
from docscout.indexing.dense_index import DenseIndex
from docscout.indexing.visual_index import ClipIndex
from docscout.schema import Component, read_components

log = logging.getLogger(__name__)

TEXT_RETRIEVERS = ("bm25", "dense_bge", "dense_minilm")


def retrievers_for(branch: str) -> Tuple[str, ...]:
    return TEXT_RETRIEVERS + (("clip",) if branch == "figure" else ())


def index_dir(branch: str, retriever: str, cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or get_config()
    return cfg["paths"]["indexes_dir"] / branch / retriever


def new_index(retriever: str, cfg: Optional[Dict[str, Any]] = None) -> BaseIndex:
    cfg = cfg or get_config()
    m = cfg["models"]
    bs = int(m.get("batch_size", 32))
    if retriever == "bm25":
        return BM25Index()
    if retriever == "dense_bge":
        return DenseIndex(m["bge"], batch_size=bs, device=m.get("device", "cpu"))
    if retriever == "dense_minilm":
        return DenseIndex(m["minilm"], batch_size=bs, device=m.get("device", "cpu"))
    if retriever == "clip":
        return ClipIndex(m["clip"], batch_size=16, device=m.get("device", "cpu"))
    raise ValueError(f"unknown retriever {retriever!r}")


def index_class(retriever: str):
    return {"bm25": BM25Index, "dense_bge": DenseIndex, "dense_minilm": DenseIndex, "clip": ClipIndex}[retriever]


def branch_components(components: Sequence[Component], branch: str, cfg: Optional[Dict[str, Any]] = None) -> List[Component]:
    cfg = cfg or get_config()
    types = set(cfg["branches"][branch])
    return [c for c in components if c.type in types]


def current_fingerprint(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or get_config()
    return file_fingerprint(cfg["paths"]["components_file"])


def index_status(branch: str, retriever: str, fp: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> str:
    """'missing' | 'stale' | 'ok' (fingerprint of components.jsonl matches)."""
    meta_path = index_dir(branch, retriever, cfg) / "meta.json"
    if not meta_path.exists():
        return "missing"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return "ok" if meta.get("fingerprint") == fp else "stale"


def stale_indexes(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """'branch/retriever' for every existing index whose fingerprint differs from components.jsonl."""
    cfg = cfg or get_config()
    fp = current_fingerprint(cfg)
    return [f"{b}/{r}" for b in cfg["branches"] for r in retrievers_for(b) if index_status(b, r, fp, cfg) == "stale"]


def stale_warning(stale: List[str]) -> str:
    return (f"WARNING: {len(stale)} index(es) are STALE (components.jsonl changed since they were built: "
            f"{', '.join(stale)}). Run: .venv/Scripts/python -m scripts.build_indexes --rebuild")


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def build_all(rebuild: bool = False, cfg: Optional[Dict[str, Any]] = None, progress: bool = True) -> Dict[str, Any]:
    """Build missing indexes (or all with rebuild=True). Refuses to touch stale ones without rebuild.

    Returns {"stale": [...], "rows": [per-index dicts], "fingerprint": fp}.
    """
    cfg = cfg or get_config()
    fp = current_fingerprint(cfg)
    stale = stale_indexes(cfg)
    if stale and not rebuild:
        return {"stale": stale, "rows": [], "fingerprint": fp}

    components = read_components(cfg["paths"]["components_file"])
    rows: List[Dict[str, Any]] = []
    for branch in cfg["branches"]:
        comps = branch_components(components, branch, cfg)
        for retriever in retrievers_for(branch):
            out = index_dir(branch, retriever, cfg)
            status = index_status(branch, retriever, fp, cfg)
            row: Dict[str, Any] = {"branch": branch, "retriever": retriever, "components": len(comps)}
            if not comps:
                row.update(action="skipped (no components)", secs=0.0, n=0)
                rows.append(row)
                continue
            if status == "ok" and not rebuild:
                with Timer() as t:
                    idx = index_class(retriever).load(out)
                row.update(action="up to date (loaded)", secs=round(t.secs, 3), n=len(idx),
                           build_secs=idx.info.get("build_secs"))
            else:
                idx = new_index(retriever, cfg)
                with Timer() as t:
                    idx.build(comps) if retriever == "bm25" else idx.build(comps, progress=progress)
                idx.info.update(fingerprint=fp, build_secs=round(t.secs, 3), built_at=time.strftime("%Y-%m-%d %H:%M:%S"))
                idx.save(out)  # overwrites meta.json and the index files
                row.update(action="built", secs=round(t.secs, 3), n=len(idx), build_secs=round(t.secs, 3))
            row["size_bytes"] = dir_size(out)
            rows.append(row)
    return {"stale": [], "rows": rows, "fingerprint": fp}


def load_index(branch: str, retriever: str, cfg: Optional[Dict[str, Any]] = None, warn: bool = True) -> Optional[BaseIndex]:
    """Load one saved index (None if missing). Logs a warning if it is stale."""
    cfg = cfg or get_config()
    out = index_dir(branch, retriever, cfg)
    if not (out / "meta.json").exists():
        return None
    if warn:
        stale = stale_indexes(cfg)
        if stale:
            log.warning(stale_warning(stale))
    return index_class(retriever).load(out)
