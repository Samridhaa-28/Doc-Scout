"""Cell boxes on a synthetic digital PDF: unique values, row bands for repeated values, no guessed boxes."""
import pymupdf
import pytest

from docscout.evidence.cell_boxes import _open, cell_boxes, has_text_layer


@pytest.fixture
def pdf(tmp_path):
    p = tmp_path / "t.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    for y, row in ((50, ["Row A", "5", "7"]), (80, ["Row B", "5", "9"])):
        for x, t in zip((10, 100, 150), row):
            page.insert_text((x, y), t, fontsize=10)
    doc.save(p)
    _open.cache_clear()
    return p


def test_unique_and_row_band_matching(pdf):
    px = (400, 400)                      # page pixels = 2 x PDF points
    table = (0, 0, 400, 400)
    out = cell_boxes(pdf, 1, table, px, [("Row B", "c2", "9"), ("Row B", "c1", "5"), ("Row A", "c1", "5"),
                                         ("Row C", "c1", "5"), ("Row A", "c1", "missing")])
    by = {(b["row"], b["value"]): b["bbox"] for b in out}
    assert ("Row B", "9") in by and ("Row B", "5") in by and ("Row A", "5") in by
    assert 130 < by[("Row B", "5")][1] < 165 and by[("Row B", "5")][0] == pytest.approx(200, abs=2)  # baseline 80 pt, top ~69 pt, x 2
    assert 70 < by[("Row A", "5")][1] < 105                                                       # via the row label
    assert ("Row C", "5") not in by and ("Row A", "missing") not in by                            # no guessed box
    assert has_text_layer(pdf, 1) and not has_text_layer(pdf, 2)


def test_clip_to_table_bbox(pdf):
    out = cell_boxes(pdf, 1, (0, 0, 400, 120), (400, 400), [("Row B", "c2", "9")])   # table above row B
    assert out == []
