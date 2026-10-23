"""Graph context expansion around a retrieved component (document-structure graph from graph/builder.py).

expand(component_id, graph, max_siblings=2) -> (context component ids, path string)
  context, in this order (each at most once, never the component itself):
    1. parent section heading (if the block sits under a section)
    2. caption of a table/figure, or the table/figure a caption describes
    3. up to `max_siblings` reading-order neighbours within `window` blocks back/forward:
       same-page neighbours first, nearest first (ties: previous before next); then other pages
  path: "Page 4 -> Section 'Revenue' -> Table 7", or "Page 4 -> Table 7" when there is no section.

expand_evidence(ids, graph, ...) applies expand() to a ranked evidence list and de-duplicates: a context
component is dropped if it is itself an evidence item or already a context item of a higher-ranked one, so the
same component is never sent to the LLM twice. This is what puts a label (e.g. "Presiding:") and its value
(in the next block) side by side.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

from docscout.graph.builder import linked_component, parent_section

TYPE_LABEL = {"paragraph": "Paragraph", "heading": "Heading", "table": "Table", "figure": "Figure",
              "caption": "Caption"}


def block_label(component_id: str, graph: nx.DiGraph) -> str:
    """'Table 7' from node type + the trailing index of the id ('..._p4_table_7')."""
    typ = graph.nodes[component_id].get("type", "block")
    n = component_id.rsplit("_", 1)[-1]
    return f"{TYPE_LABEL.get(typ, typ.title())} {n}"


def path_string(component_id: str, graph: nx.DiGraph, max_title: int = 40) -> str:
    """'Page 4 -> Section 'Revenue' -> Table 7' (no section part when the block has none)."""
    node = graph.nodes[component_id]
    parts = [f"Page {node.get('page')}"]
    sec = parent_section(graph, component_id)
    if sec is not None and sec in graph and sec != component_id:  # a heading is not inside its own section
        title = " ".join(str(graph.nodes[sec].get("text", "")).split())
        if len(title) > max_title:
            title = title[: max_title - 3] + "..."
        parts.append(f"Section '{title}'")
    parts.append(block_label(component_id, graph))
    return " -> ".join(parts)


def _blocks_in_order(graph: nx.DiGraph) -> List[str]:
    """Block nodes sorted by reading order (cached on the graph object)."""
    cached = graph.graph.get("_blocks_in_order")
    if cached is None:
        cached = sorted((n for n, d in graph.nodes(data=True) if d.get("kind") == "block"),
                        key=lambda n: (graph.nodes[n].get("order", 0), n))
        graph.graph["_blocks_in_order"] = cached
    return cached


def _ordered_siblings(component_id: str, graph: nx.DiGraph, window: int) -> List[str]:
    """Blocks within `window` positions in reading order, sorted: same page first, nearest first, previous first.

    Uses the nodes' `order` attribute, not the sibling edges: a DiGraph keeps one edge per node pair, so the
    caption -> table "caption_of" edge replaces the "sibling" edge between them and would cut the chain."""
    page = graph.nodes[component_id].get("page")
    blocks = _blocks_in_order(graph)
    i = blocks.index(component_id)
    near = blocks[max(0, i - window):i] + blocks[i + 1:i + 1 + window]
    order = graph.nodes[component_id].get("order", 0)

    def key(s: str) -> Tuple[int, int, int]:
        d = graph.nodes[s].get("order", 0) - order
        return (0 if graph.nodes[s].get("page") == page else 1, abs(d), 0 if d < 0 else 1)
    return sorted(near, key=key)


def expand(component_id: str, graph: nx.DiGraph, max_siblings: int = 2, window: int = 3
           ) -> Tuple[List[str], str]:
    """(context component ids, path string) for one component. See the module docstring."""
    if component_id not in graph:
        return [], ""
    ctx: List[str] = []

    def add(x: Optional[str]) -> None:
        if x and x != component_id and x not in ctx and x in graph:
            ctx.append(x)
    add(parent_section(graph, component_id))
    add(linked_component(graph, component_id))
    sibs = [s for s in _ordered_siblings(component_id, graph, window) if s not in ctx]
    for s in sibs[:max_siblings]:  # the heading / caption already added does not use up a sibling slot
        add(s)
    return ctx, path_string(component_id, graph)


def expand_evidence(ids: Sequence[str], graph: nx.DiGraph, max_siblings: int = 2, window: int = 3
                    ) -> Dict[str, Tuple[List[str], str]]:
    """{evidence id: (de-duplicated context ids, path)} for a ranked evidence list."""
    used = set(ids)
    out: Dict[str, Tuple[List[str], str]] = {}
    for cid in ids:
        ctx, path = expand(cid, graph, max_siblings, window)
        keep = [c for c in ctx if c not in used]
        used.update(keep)
        out[cid] = (keep, path)
    return out
