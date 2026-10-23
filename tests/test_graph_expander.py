"""Graph context expansion on a small hand-made document graph."""
from docscout.context.graph_expander import expand, expand_evidence, path_string
from docscout.graph.builder import build_graph
from docscout.schema import Component


def comps():
    def c(cid, page, typ, order, text, section=None, **meta):
        return Component(id=cid, doc_id="d", page=page, type=typ, text=text, text_form=text, order=order,
                         section_id=section, meta=meta)
    return [
        c("d_p1_paragraph_0", 1, "paragraph", 0, "Intro text"),
        c("d_p2_heading_0", 2, "heading", 1, "Revenue"),
        c("d_p2_paragraph_0", 2, "paragraph", 2, "Presiding:", "d_p2_heading_0"),
        c("d_p2_paragraph_1", 2, "paragraph", 3, "Lee A. Waller", "d_p2_heading_0"),
        c("d_p2_caption_0", 2, "caption", 4, "Table 1. Revenue by year", "d_p2_heading_0", caption_of="d_p2_table_0"),
        c("d_p2_table_0", 2, "table", 5, "Table: revenue", "d_p2_heading_0"),
        c("d_p3_paragraph_0", 3, "paragraph", 6, "Next page", "d_p2_heading_0"),
    ]


def test_expand_heading_caption_and_same_page_siblings():
    G = build_graph(comps())
    ctx, path = expand("d_p2_table_0", G, max_siblings=2)
    # heading, then caption (linked), then 2 siblings: same page first, nearest first (the caption, 1 back, is
    # already in; the p3 paragraph is on another page). The caption sits between paragraph_1 and the table, where
    # the DiGraph lost the sibling edge, so this also checks that siblings come from reading order.
    assert ctx == ["d_p2_heading_0", "d_p2_caption_0", "d_p2_paragraph_1", "d_p2_paragraph_0"]
    assert path == "Page 2 -> Section 'Revenue' -> Table 0"


def test_caption_links_to_its_table():
    G = build_graph(comps())
    ctx, _ = expand("d_p2_caption_0", G, max_siblings=0)
    assert ctx == ["d_p2_heading_0", "d_p2_table_0"]


def test_path_without_section():
    G = build_graph(comps())
    assert path_string("d_p1_paragraph_0", G) == "Page 1 -> Paragraph 0"
    ctx, _ = expand("d_p1_paragraph_0", G)
    assert "d_p1_paragraph_0" not in ctx


def test_label_and_value_end_up_together_and_dedup():
    G = build_graph(comps())
    out = expand_evidence(["d_p2_paragraph_0", "d_p2_paragraph_1"], G, max_siblings=2)
    # the label's context does not repeat the other evidence item (its value) ...
    assert "d_p2_paragraph_1" not in out["d_p2_paragraph_0"][0]
    # ... and the shared heading is sent only once (with the higher-ranked item)
    assert "d_p2_heading_0" in out["d_p2_paragraph_0"][0]
    assert "d_p2_heading_0" not in out["d_p2_paragraph_1"][0]
    all_ctx = [x for ids, _ in out.values() for x in ids]
    assert len(all_ctx) == len(set(all_ctx))
    # alone, the label's context contains its value
    assert "d_p2_paragraph_1" in expand("d_p2_paragraph_0", G)[0]


def test_heading_path_does_not_repeat_itself():
    cs = comps()
    cs[1].section_id = "d_p2_heading_0"  # headings may point at their own section
    G = build_graph(cs)
    assert path_string("d_p2_heading_0", G) == "Page 2 -> Heading 0"
