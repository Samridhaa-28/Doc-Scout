"""Tests for bbox conversion, the serializer and the graph builder (no Docling run, no models)."""
from pathlib import Path

import pandas as pd
import pytest

from docscout.graph.builder import (
    CAPTION_OF,
    PARENT,
    SIBLING,
    build_graph,
    caption_of,
    linked_component,
    load_graph,
    parent_section,
    path_to_root,
    save_graph,
    siblings,
)
from docscout.ingestion.parser import docling_bbox_to_pixels, images_to_pdf, link_captions_nearest, safe_doc_id
from docscout.ingestion.serializer import serialize_components, table_text_form, truncate_tokens
from docscout.schema import Component


# ------------------------------------------------------------------ bbox

def test_bbox_bottom_left_points_to_rendered_pixels():
    # US-letter page (612 x 792 pt) rendered at 144 dpi -> 1224 x 1584 px (scale 2).
    # Docling bbox, bottom-left origin: 1 inch from the left and top, 6.5 in wide, 1 in tall.
    bbox = docling_bbox_to_pixels(72, 720, 540, 648, True, 612, 792, 1224, 1584)
    assert bbox == (144.0, 144.0, 1080.0, 288.0)


def test_bbox_top_left_origin_and_mpdocvqa_png_space():
    # MP-DocVQA page: PNG 2257 x 1764 px, PDF page built at 200 dpi -> 812.52 x 635.04 pt.
    w_pt, h_pt = 2257 * 72 / 200, 1764 * 72 / 200
    bbox = docling_bbox_to_pixels(36, 72, 360, 144, False, w_pt, h_pt, 2257, 1764)
    assert bbox == pytest.approx((100.0, 200.0, 1000.0, 400.0), abs=0.01)
    # same box expressed bottom-left gives the same pixels
    bl = docling_bbox_to_pixels(36, h_pt - 72, 360, h_pt - 144, True, w_pt, h_pt, 2257, 1764)
    assert bl == pytest.approx(bbox, abs=0.01)


def test_bbox_is_clamped_inside_page():
    x0, y0, x1, y1 = docling_bbox_to_pixels(-5, 800, 700, -10, True, 612, 792, 1224, 1584)
    assert 0 <= x0 < x1 <= 1224 and 0 <= y0 < y1 <= 1584


def test_images_to_pdf_page_size(tmp_path: Path):
    import pymupdf
    from PIL import Image

    png = tmp_path / "p.png"
    Image.new("RGB", (400, 200), "white").save(png)
    sizes = images_to_pdf([png, png], tmp_path / "out.pdf", dpi=200)
    assert sizes == [(144.0, 72.0)] * 2
    with pymupdf.open(str(tmp_path / "out.pdf")) as pdf:
        assert len(pdf) == 2 and pdf[0].rect.width == pytest.approx(144.0)


def test_page_marker_regex():
    from docscout.ingestion.parser import PAGE_MARKER_RE

    assert PAGE_MARKER_RE.match("Page 5 January, 1970")
    assert not PAGE_MARKER_RE.match("Table 2: Page counts by county")


def test_safe_doc_id():
    assert safe_doc_id("Annual Report_2023 (final)") == "annual-report-2023-final"


# ------------------------------------------------------------------ serializer

def _c(cid, ctype, page=1, order=0, text="", bbox=(0, 0, 10, 10), section=None, **kw):
    return Component(id=cid, doc_id="d", page=page, type=ctype, text=text, bbox=bbox,
                     page_width=1000, page_height=1000, order=order, section_id=section, **kw)


def test_table_text_form_hand_made_table():
    df = pd.DataFrame({"Item": ["Revenue", "Costs", "Profit", "Tax"],
                       "2022": ["100", "60", "40", "8"], "2023": ["120", "70", "50", "10"]})
    tf = table_text_form(df, title="Table 2: Income statement")
    lines = tf.splitlines()
    assert lines[0] == "Table: Table 2: Income statement"
    assert lines[1] == "Columns: Item | 2022 | 2023"
    assert lines[2] == "Rows: Revenue; Costs; Profit; Tax"  # all row headers
    assert lines[3:] == ["Revenue | 100 | 120", "Costs | 60 | 70", "Profit | 40 | 50"]  # first 3 rows only
    assert "Tax | 8 | 10" not in tf


def test_serialize_components_all_types():
    df = pd.DataFrame({"Year": ["2022", "2023"], "Sales": ["5", "7"]})
    comps = [
        _c("d_p1_heading_0", "heading", order=0, text="Results", section="d_p1_heading_0"),
        _c("d_p1_paragraph_0", "paragraph", order=1, text="Sales grew strongly.", bbox=(0, 600, 500, 650),
           section="d_p1_heading_0"),
        _c("d_p1_table_0", "table", order=2, text="Year | Sales", bbox=(0, 100, 500, 300), section="d_p1_heading_0",
           meta={"caption_id": "d_p1_caption_0"}),
        _c("d_p1_caption_0", "caption", order=3, text="Table 1: Sales by year", section="d_p1_heading_0",
           meta={"caption_of": "d_p1_table_0"}),
        _c("d_p1_figure_0", "figure", order=4, bbox=(0, 660, 500, 900), section="d_p1_heading_0"),
        _c("d_p2_figure_0", "figure", page=2, order=5),  # no caption, no paragraph on its page
    ]
    serialize_components(comps, tables={"d_p1_table_0": df}, max_tokens=512)
    by = {c.id: c for c in comps}
    assert by["d_p1_paragraph_0"].text_form == "Sales grew strongly."
    # title = nearest block directly above the table (the heading), then the linked caption
    assert by["d_p1_table_0"].text_form.startswith(
        "Table: Results\nCaption: Table 1: Sales by year\nColumns: Year | Sales")
    assert "Sales grew strongly." in by["d_p1_figure_0"].text_form  # nearest paragraph
    assert by["d_p2_figure_0"].text_form == "figure on page 2"  # fallback is never empty
    assert all(c.text_form.strip() for c in comps)


def test_table_title_skips_page_marker():
    from docscout.ingestion.serializer import title_above

    table = _c("d_p1_table_0", "table", bbox=(50, 200, 950, 800))
    title = _c("d_p1_paragraph_0", "paragraph", text="Missouri Food Donation Program", bbox=(50, 90, 400, 150))
    marker = _c("d_p1_paragraph_1", "paragraph", text="Page 5 January, 1970", bbox=(800, 90, 900, 160),
                meta={"docling_label": "page_marker"})
    assert title_above(table, [table, title, marker]).id == "d_p1_paragraph_0"


def test_truncate_tokens():
    text = " ".join(f"w{i}" for i in range(1000))
    out = truncate_tokens(text, 512)
    assert len(out.split()) == 512 and text.startswith(out)
    assert truncate_tokens("short text", 512) == "short text"


def test_nearest_caption_fallback():
    table = _c("d_p1_table_0", "table", bbox=(100, 200, 600, 500))
    cap_below = _c("d_p1_caption_0", "caption", text="Table 1", bbox=(100, 510, 400, 530))
    cap_far = _c("d_p1_caption_1", "caption", text="Figure 9", bbox=(100, 950, 400, 970))
    assert link_captions_nearest([table, cap_below, cap_far], max_gap_frac=0.15) == 1
    assert table.meta["caption_id"] == "d_p1_caption_0" and cap_below.meta["caption_of"] == table.id
    assert "caption_of" not in cap_far.meta


# ------------------------------------------------------------------ graph

def _fake_doc():
    return [
        _c("d_p1_paragraph_0", "paragraph", page=1, order=0, text="Cover note"),
        _c("d_p1_heading_0", "heading", page=1, order=1, text="Revenue", section="d_p1_heading_0"),
        _c("d_p1_paragraph_1", "paragraph", page=1, order=2, text="Intro", section="d_p1_heading_0"),
        _c("d_p2_table_0", "table", page=2, order=3, text="t", section="d_p1_heading_0",
           meta={"caption_id": "d_p2_caption_0"}),
        _c("d_p2_caption_0", "caption", page=2, order=4, text="Table 1", section="d_p1_heading_0",
           meta={"caption_of": "d_p2_table_0"}),
        _c("d_p2_paragraph_0", "paragraph", page=2, order=5, text="After", section="d_p1_heading_0"),
    ]


def test_graph_structure_and_rels(tmp_path: Path):
    G = build_graph(_fake_doc())
    kinds = {n: d["kind"] for n, d in G.nodes(data=True)}
    assert list(kinds.values()).count("document") == 1
    assert {"page::d::1", "page::d::2"} <= set(kinds)
    assert kinds["sec::d_p1_heading_0::p2"] == "section"  # section continues on page 2
    assert all("rel" in d for _, _, d in G.edges(data=True))
    assert {d["rel"] for _, _, d in G.edges(data=True)} == {PARENT, SIBLING, CAPTION_OF}

    assert path_to_root(G, "d_p2_table_0") == ["d_p2_table_0", "sec::d_p1_heading_0::p2", "page::d::2", "doc::d"]
    assert path_to_root(G, "d_p1_paragraph_0") == ["d_p1_paragraph_0", "page::d::1", "doc::d"]  # before 1st heading
    assert parent_section(G, "d_p2_table_0") == "d_p1_heading_0"
    assert parent_section(G, "d_p1_paragraph_0") is None
    assert siblings(G, "d_p2_table_0", k=1) == ["d_p1_paragraph_1", "d_p2_caption_0"]
    assert siblings(G, "d_p2_table_0", k=2) == ["d_p1_heading_0", "d_p1_paragraph_1", "d_p2_caption_0",
                                                "d_p2_paragraph_0"]
    assert caption_of(G, "d_p2_table_0") == "d_p2_caption_0"
    assert linked_component(G, "d_p2_caption_0") == "d_p2_table_0"
    assert linked_component(G, "d_p2_table_0") == "d_p2_caption_0"

    save_graph(G, "d", graphs_dir=tmp_path)
    G2 = load_graph("d", graphs_dir=tmp_path)
    assert set(G2.edges) == set(G.edges)


# ------------------------------------------------------------------ data check (runs only after ingest)

def test_duplicate_ids_detected():
    from scripts.check_components import duplicate_ids

    assert duplicate_ids(["a", "b", "a", "c", "a"]) == {"a": 3}
    assert duplicate_ids(["a", "b"]) == {}


def test_force_cleanup_removes_old_lines_and_files(tmp_path: Path):
    from docscout.schema import read_components, write_components
    from scripts.ingest import clean_doc_outputs, drop_docs

    paths = {k: tmp_path / k for k in ("tables_dir", "figures_dir", "pages_dir", "debug_dir", "graphs_dir", "docs_dir")}
    for d in paths.values():
        d.mkdir()
    for name in ("doca_p1_table_0.parquet", "docb_p1_table_0.parquet"):
        (paths["tables_dir"] / name).write_text("x")
    (paths["figures_dir"] / "doca_p2_figure_0.png").write_text("x")
    (paths["pages_dir"] / "doca").mkdir()
    (paths["pages_dir"] / "doca" / "page_1.png").write_text("x")
    (paths["graphs_dir"] / "doca.pkl").write_text("x")
    removed = clean_doc_outputs("doca", {"paths": paths})
    assert removed == {"tables": 1, "figures": 1, "pages": 1, "other": 1}
    assert [p.name for p in paths["tables_dir"].iterdir()] == ["docb_p1_table_0.parquet"]  # other doc untouched
    assert not (paths["pages_dir"] / "doca").exists()

    comp_file = tmp_path / "components.jsonl"
    a = Component(id="doca_p1_paragraph_0", doc_id="doca", page=1, type="paragraph", text_form="x")
    b = Component(id="docb_p1_paragraph_0", doc_id="docb", page=1, type="paragraph", text_form="y")
    write_components(comp_file, [a, b, a])
    assert drop_docs(comp_file, {"doca"}) == 2
    assert read_components(comp_file) == [b]


def test_processed_components_valid():
    """Data check over components.jsonl: every check in scripts.check_components must pass."""
    from docscout.config import get_config
    from scripts.check_components import validate

    path = get_config()["paths"]["components_file"]
    if not path.exists():
        pytest.skip("components.jsonl not built yet")
    r = validate(path)
    assert r["lines"] > 0
    assert r["lines"] == r["unique_ids"], r["duplicate_ids"]
    for key in ("bad_id_format", "empty_text_form", "bad_page", "bbox_outside", "missing_files",
                "size_mismatch", "page_mismatch", "answer_page_missing"):
        assert not r[key], (key, r[key][:5])
