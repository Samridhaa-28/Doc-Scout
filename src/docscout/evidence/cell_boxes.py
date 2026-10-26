"""Page-pixel boxes of localized table cells, for highlighting them in place (digital PDFs only).

P2 did not keep cell boxes. For a PDF with a text layer, each localized cell value is searched with PyMuPDF inside the
table's bbox; when the value occurs more than once, the hit that lines up with the row label (same line) and the
column header (same column) wins. Image-only pages (MP-DocVQA) have no text layer: no boxes, the UI shows the table box
and lists the cells instead of guessing positions.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pymupdf


@lru_cache(maxsize=8)
def _open(path: str) -> Any:
    return pymupdf.open(path)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _search(page: Any, text: Any, clip: Any) -> List[Any]:
    """Hits of the first 4 words of `text` (dot leaders removed) inside clip."""
    t = " ".join(re.sub(r"(\s*\.){2,}", " ", str(text)).split()[:4]).strip()
    return page.search_for(t, clip=clip) if t else []


def cell_boxes(pdf_path: Path, page_no: int, table_bbox: Sequence[float], page_size: Sequence[float],
               cells: Sequence[Sequence[Any]], margin: float = 4.0) -> List[Dict[str, Any]]:
    """[{row, column, value, bbox (page pixels, top-left origin)}] for the cells found; others are left out.

    A value found once inside the table is used as is. A value found several times must line up with its row (band
    from single-hit values or the row label) and, when known, its column; otherwise it gets no box (never a guess)."""
    doc = _open(str(pdf_path))
    if page_no < 1 or page_no > doc.page_count:
        return []
    page = doc[page_no - 1]
    s = float(page_size[0]) / page.rect.width  # page pixels per PDF point
    x0, y0, x1, y1 = (float(v) / s for v in table_bbox)
    clip = pymupdf.Rect(x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    # pass 1: hits per cell; a value found exactly once fixes its row's y-band and its column's x-band
    found = []
    row_band: Dict[str, Any] = {}
    col_band: Dict[str, Any] = {}
    for row, col, value in cells:
        hits = _search(page, value, clip)
        found.append((row, col, value, hits))
        if len(hits) == 1:
            row_band.setdefault(str(row), (hits[0].y0, hits[0].y1))
            col_band.setdefault(str(col), (hits[0].x0, hits[0].x1))
    for row, col, value, hits in found:  # row labels found as text also give the row band
        if str(row) not in row_band:
            lab = _search(page, row, clip)
            if len(lab) == 1:
                row_band[str(row)] = (lab[0].y0, lab[0].y1)
    out = []
    for row, col, value, hits in found:
        if not hits:
            continue
        if len(hits) == 1:
            best = hits[0]
        else:  # ambiguous value: must line up with its row band (and column band when known), else no box
            rb, cb = row_band.get(str(row)), col_band.get(str(col))
            ok = [r for r in hits if rb is not None and _overlap(r.y0, r.y1, *rb) > 0.5 * r.height
                  and (cb is None or _overlap(r.x0, r.x1, *cb) > 0)]
            if len(ok) != 1:
                continue
            best = ok[0]
        out.append({"row": str(row), "column": str(col), "value": str(value),
                    "bbox": [best.x0 * s, best.y0 * s, best.x1 * s, best.y1 * s]})
    return out


def has_text_layer(pdf_path: Optional[Path], page_no: int) -> bool:
    if pdf_path is None or not Path(pdf_path).exists():
        return False
    doc = _open(str(pdf_path))
    return 1 <= page_no <= doc.page_count and bool(doc[page_no - 1].get_text("text").strip())
