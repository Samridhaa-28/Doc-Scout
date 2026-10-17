"""Tests for docscout.schema: ids, dict round trips and JSONL round trips."""
from pathlib import Path

import pytest

from docscout.schema import (
    Component,
    Evidence,
    QueryResult,
    make_component_id,
    read_components,
    read_jsonl,
    write_components,
    write_jsonl,
)


def _component(n: int = 0, ctype: str = "paragraph") -> Component:
    return Component(
        id=make_component_id("docA", 2, ctype, n),
        doc_id="docA",
        page=2,
        type=ctype,
        text=f"Revenue grew 12% in 2023 — item {n}",
        text_form=f"Revenue grew 12% in 2023 — item {n}",
        bbox=(10, 20.5, 300, 80),
        page_width=1224,
        page_height=1584,
        order=n,
        section_id="docA_p1_heading_0",
        table_path="data/processed/tables/x.parquet" if ctype == "table" else None,
        meta={"caption_id": "docA_p2_caption_1", "conf": 0.9},
    )


def test_component_id_format():
    assert make_component_id("ffbf0023", 3, "table", 7) == "ffbf0023_p3_table_7"


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        Component(id="x", doc_id="d", page=1, type="banana")


def test_component_dict_round_trip():
    c = _component()
    assert Component.from_dict(c.to_dict()) == c
    assert isinstance(Component.from_dict(c.to_dict()).bbox, tuple)


def test_components_jsonl_round_trip(tmp_path: Path):
    comps = [_component(0), _component(1, "table"), _component(2, "figure")]
    path = tmp_path / "sub" / "components.jsonl"
    assert write_components(path, comps) == 3
    write_components(path, [_component(3, "caption")], append=True)
    back = read_components(path)
    assert back == comps + [_component(3, "caption")]
    assert path.read_text(encoding="utf-8").count("\n") == 4
    assert "—" in path.read_text(encoding="utf-8")  # non-ASCII kept as UTF-8


def test_query_result_round_trip(tmp_path: Path):
    ev = Evidence(
        component=_component(0, "table"),
        scores={"bm25": 3.2, "dense": 0.71, "fused": 0.8, "rerank": 5.1},
        context=[_component(1, "heading"), _component(2, "caption")],
        localized_cells=[("Revenue", "2023", "1,234")],
        why="numerical query; table branch weight 0.6",
        meta={"graph_path": "Page 2 -> Section 'Results' -> Table 0"},
    )
    qr = QueryResult(query="What was revenue in 2023?", query_type="numerical", answer="1,234 [E1]",
                     evidence=[ev], meta={"query_id": "q1"})
    path = tmp_path / "results.jsonl"
    write_jsonl(path, [qr.to_dict(), qr.to_dict()])
    rows = read_jsonl(path)
    assert len(rows) == 2
    back = QueryResult.from_dict(rows[0])
    assert back == qr
    assert back.evidence[0].localized_cells[0] == ("Revenue", "2023", "1,234")
