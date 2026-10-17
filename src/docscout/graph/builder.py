"""Document-structure graph (networkx DiGraph), one per document.

Hierarchy (rel="parent_child"):  Document -> Page -> Section -> Block
  * a Section node is (heading, page): a section that spans pages gets one node per page,
    so path_to_root(block) always passes through the block's own page;
  * blocks before the first heading hang directly under their Page.
Other edges:
  * rel="sibling": block -> next block in reading order (document-wide)
  * rel="caption_of": caption -> its table/figure

Node ids: "doc::{doc_id}", "page::{doc_id}::{page}", "sec::{heading_id}::p{page}", block = component id.
Every node has attribute "kind" (document/page/section/block).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Union

import networkx as nx

from docscout.config import get_config
from docscout.schema import Component

PARENT, SIBLING, CAPTION_OF = "parent_child", "sibling", "caption_of"


def doc_node(doc_id: str) -> str:
    return f"doc::{doc_id}"


def page_node(doc_id: str, page: int) -> str:
    return f"page::{doc_id}::{page}"


def section_node(heading_id: str, page: int) -> str:
    return f"sec::{heading_id}::p{page}"


def build_graph(components: List[Component]) -> nx.DiGraph:
    """Build the structure graph for ONE document's components (any order; sorted by `order`)."""
    comps = sorted(components, key=lambda c: c.order)
    G = nx.DiGraph()
    if not comps:
        return G
    doc_id = comps[0].doc_id
    by_id = {c.id: c for c in comps}
    d = doc_node(doc_id)
    G.add_node(d, kind="document", doc_id=doc_id)

    for c in comps:
        p = page_node(doc_id, c.page)
        if p not in G:
            G.add_node(p, kind="page", doc_id=doc_id, page=c.page)
            G.add_edge(d, p, rel=PARENT)
        parent = p
        if c.section_id and c.section_id in by_id:
            s = section_node(c.section_id, c.page)
            if s not in G:
                G.add_node(s, kind="section", heading_id=c.section_id, title=by_id[c.section_id].text[:200], page=c.page)
                G.add_edge(p, s, rel=PARENT)
            parent = s
        G.add_node(c.id, kind="block", type=c.type, page=c.page, order=c.order, bbox=tuple(c.bbox),
                   text=c.text[:300], section_id=c.section_id)
        G.add_edge(parent, c.id, rel=PARENT)

    for a, b in zip(comps, comps[1:]):
        G.add_edge(a.id, b.id, rel=SIBLING)
    for c in comps:
        if c.type == "caption" and c.meta.get("caption_of") in by_id:
            G.add_edge(c.id, c.meta["caption_of"], rel=CAPTION_OF)
    # sort pages for stable iteration
    return G


def graph_path(doc_id: str, graphs_dir: Optional[Path] = None) -> Path:
    return (graphs_dir or get_config()["paths"]["graphs_dir"]) / f"{doc_id}.pkl"


def save_graph(G: nx.DiGraph, doc_id: str, graphs_dir: Optional[Path] = None) -> Path:
    out = graph_path(doc_id, graphs_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def load_graph(doc_id_or_path: Union[str, Path], graphs_dir: Optional[Path] = None) -> nx.DiGraph:
    """Load a pickled graph by doc_id or by file path."""
    p = Path(doc_id_or_path)
    if p.suffix != ".pkl":
        p = graph_path(str(doc_id_or_path), graphs_dir)
    with p.open("rb") as f:
        return pickle.load(f)


def _edges(G: nx.DiGraph, n: str, rel: str, out: bool = True) -> List[str]:
    it = G.out_edges(n, data="rel") if out else G.in_edges(n, data="rel")
    return [(v if out else u) for u, v, r in it if r == rel]


def parent(G: nx.DiGraph, n: str) -> Optional[str]:
    ps = _edges(G, n, PARENT, out=False)
    return ps[0] if ps else None


def parent_section(G: nx.DiGraph, n: str) -> Optional[str]:
    """Heading component id of the section containing block n (None if the block has no section)."""
    p = parent(G, n)
    if p is not None and G.nodes[p].get("kind") == "section":
        return G.nodes[p]["heading_id"]
    return None


def siblings(G: nx.DiGraph, n: str, k: int = 1) -> List[str]:
    """Up to k previous and k next blocks in reading order, returned in reading order (n excluded)."""
    prev, cur = [], n
    for _ in range(k):
        ps = _edges(G, cur, SIBLING, out=False)
        if not ps:
            break
        cur = ps[0]
        prev.insert(0, cur)
    nxt, cur = [], n
    for _ in range(k):
        ns = _edges(G, cur, SIBLING, out=True)
        if not ns:
            break
        cur = ns[0]
        nxt.append(cur)
    return prev + nxt


def caption_of(G: nx.DiGraph, n: str) -> Optional[str]:
    """Caption id for a table/figure n (None if it has no caption)."""
    caps = _edges(G, n, CAPTION_OF, out=False)
    return caps[0] if caps else None


def linked_component(G: nx.DiGraph, n: str) -> Optional[str]:
    """For a caption: the table/figure it describes. For a table/figure: its caption."""
    targets = _edges(G, n, CAPTION_OF, out=True)
    if targets:
        return targets[0]
    return caption_of(G, n)


def path_to_root(G: nx.DiGraph, n: str) -> List[str]:
    """[n, (section), page, document] following parent_child edges upward."""
    path, cur = [n], n
    while True:
        p = parent(G, cur)
        if p is None:
            return path
        path.append(p)
        cur = p
