"""Fill Component.text_form (the text that gets indexed), as defined in CLAUDE.md:

  paragraph / heading / caption : raw text
  table  : title/caption + column headers + row headers + first 3 rows
  figure : caption + nearest paragraph (+ text Docling found inside the picture)

Every text_form is non-empty and truncated to ~max_text_form_tokens tokens.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from docscout.config import get_config
from docscout.schema import Component

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
N_TABLE_ROWS = 3
MAX_ROW_HEADERS = 30
# Not usable as a table title. page_header is allowed: on scans Docling often labels the real
# title block at the top of the page as page_header.
FURNITURE_LABELS = {"page_footer", "page_marker"}


def truncate_tokens(text: str, max_tokens: int) -> str:
    """Cut text after `max_tokens` word/punctuation tokens (~ subword tokens for BERT-style models)."""
    count = 0
    for m in _TOKEN_RE.finditer(text):
        count += 1
        if count > max_tokens:
            return text[: m.start()].rstrip()
    return text


def table_text_form(df, title: str = "", caption: str = "", n_rows: int = N_TABLE_ROWS) -> str:
    """Serialize a table: title, caption, column headers, row headers (first column), first n_rows rows."""
    parts: List[str] = []
    if title:
        parts.append(f"Table: {title.strip()}")
    if caption and caption.strip() != title.strip():
        parts.append(f"Caption: {caption.strip()}")
    if df is not None and not df.empty:
        cols = [str(c) for c in df.columns]
        if not all(re.fullmatch(r"(col_)?\d+", c) for c in cols):  # skip auto-generated names
            parts.append("Columns: " + " | ".join(cols))
        row_headers = [str(v).strip() for v in df.iloc[:, 0].tolist() if str(v).strip()]
        if row_headers:
            parts.append("Rows: " + "; ".join(row_headers[:MAX_ROW_HEADERS]))
        for row in df.head(n_rows).itertuples(index=False):
            parts.append(" | ".join(str(v) for v in row))
    return "\n".join(parts)


def _center_dist(a: Component, b: Component) -> float:
    ax, ay = (a.bbox[0] + a.bbox[2]) / 2, (a.bbox[1] + a.bbox[3]) / 2
    bx, by = (b.bbox[0] + b.bbox[2]) / 2, (b.bbox[1] + b.bbox[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def nearest_paragraph(target: Component, components: List[Component]) -> Optional[Component]:
    """Closest paragraph (by bbox centre distance) on the same page."""
    cands = [c for c in components if c.type == "paragraph" and c.page == target.page]
    return min(cands, key=lambda c: _center_dist(c, target)) if cands else None


def title_above(target: Component, components: List[Component], max_gap_frac: float = 0.15) -> Optional[Component]:
    """Nearest heading/paragraph that ends just above `target` on the same page and overlaps it horizontally."""
    best, best_gap = None, None
    for c in components:
        if c.page != target.page or c.type not in ("heading", "paragraph") or c.id == target.id:
            continue
        if c.meta.get("docling_label") in FURNITURE_LABELS:  # running headers/footers are not titles
            continue
        gap = target.bbox[1] - c.bbox[3]
        overlap = min(c.bbox[2], target.bbox[2]) - max(c.bbox[0], target.bbox[0])
        if 0 <= gap <= max_gap_frac * target.page_height and overlap > 0 and (best_gap is None or gap < best_gap):
            best, best_gap = c, gap
    return best


def _load_table(c: Component):
    import pandas as pd

    if not c.table_path:
        return None
    p = Path(c.table_path)
    p = p if p.is_absolute() else get_config()["repo_root"] / p
    return pd.read_parquet(p) if p.exists() else None


def serialize_components(components: List[Component], tables: Optional[Dict[str, object]] = None,
                         max_tokens: Optional[int] = None) -> List[Component]:
    """Fill text_form in place (and return the list). `tables` maps component id -> DataFrame (else parquet is read)."""
    max_tokens = max_tokens or int(get_config()["parsing"]["max_text_form_tokens"])
    by_id = {c.id: c for c in components}
    for c in components:
        caption = by_id[c.meta["caption_id"]].text if c.meta.get("caption_id") in by_id else ""
        heading = by_id[c.section_id].text if c.section_id in by_id and c.section_id != c.id else ""
        if c.type == "table":
            df = (tables or {}).get(c.id)
            if df is None:
                df = _load_table(c)
            above = title_above(c, components)
            title = above.text if above is not None else ("" if caption else heading)
            tf = table_text_form(df, title=title, caption=caption)
            if not tf.strip():
                tf = c.text
        elif c.type == "figure":
            parts = [f"Figure: {caption}" if caption else ""]
            para = nearest_paragraph(c, components)
            if para is not None:
                parts.append(para.text)
            if c.meta.get("inner_text"):
                parts.append(c.meta["inner_text"])
            tf = "\n".join(p for p in parts if p)
        else:
            tf = c.text
        if not tf.strip():  # guarantee non-empty
            tf = f"{c.type} on page {c.page}" + (f" in section: {heading}" if heading else "")
        c.text_form = truncate_tokens(tf.strip(), max_tokens)
    return components
