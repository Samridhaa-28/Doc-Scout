"""Structured value localization: find the rows / columns / cells of a table that a numerical query asks about.

locate(query, table_df) -> (sub_df, cells, flag)
  cells: [(row_label, column, value)]; flag: "cell" | "row" | "column" | "fallback" | "no_match"

Steps:
  1. clean_table: Docling headers are messy.
     - placeholder column names ("0", "col_3", "Unnamed: 2") are replaced by leading body rows that look like
       headers (no digits, every value <= 40 chars) - at most 2 such rows are promoted;
     - multi-level headers arrive joined with "." ("Participants (4).No."): a prefix shared by EVERY column is
       stripped (it is a table title, not a column name); pandas duplicate suffixes ".1" are dropped;
     - body rows identical to the header are dropped; trailing dot leaders ("Total . . . .") are removed.
  2. Row-label column = the first column whose non-empty values are mostly (> 50%) non-numeric (else column 0).
  3. Query terms = BM25 tokens of the query. A term "matches" a header / row label word when
       - it contains a digit, is a unit or is shorter than localization.min_fuzzy_len: exact token equality;
       - otherwise: rapidfuzz ratio >= localization.fuzzy_threshold.
     Column score = number of distinct query terms matched by the column header (the row-label column's header
     is not counted). Row score = number of distinct query terms matched by the row label.
  4. Best-scoring rows R and columns C (all ties at the max score, score > 0). A matched group-header row
     (label, no values, e.g. "EXPENDITURES:") stands for the rows under it (up to the next group header, max 10).
     No row matched but the query has an ordinal (first/second/.../5th, last/final) that is not a word of a
     column header -> that data row.
       R and C -> intersecting cells ("cell");  only R -> those rows ("row");  only C -> that column ("column").
  5. Fallback: rows where any cell contains a query number exactly or a query keyword ("fallback").
  6. Nothing -> the whole table, no cells ("no_match").
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
from rapidfuzz import fuzz

from docscout.config import get_config
from docscout.indexing.bm25_index import tokenize
from docscout.schema import Cell

_PLACEHOLDER = re.compile(r"^(?:col_\d+|unnamed: ?\d+(?:_level_\d+)?|)$", re.IGNORECASE)
_DUP_SUFFIX = re.compile(r"\.\d+$")
_DOT_LEADER = re.compile(r"(?:\s*\.){2,}\s*$")  # "Medicaid 2 . . . . ." -> "Medicaid 2"
_NUMERIC = re.compile(r"^[\s$€£¥(+\-]*\d[\d\s,.\-/:%)]*(?:[ap]\.?m\.?)?$", re.IGNORECASE)
UNITS = frozenset("% mg kg g lb lbs oz ml l km m cm mm pct usd $ no yr yrs hr hrs".split())

FLAGS = ("cell", "row", "column", "fallback", "no_match")
ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
            "fifth": 5, "5th": 5, "last": -1, "final": -1}
MAX_GROUP_ROWS = 10


def _s(v: Any) -> str:
    """Cell value as a clean single-line string ('' for None / NaN)."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return " ".join(str(v).split())


def is_numeric(v: Any) -> bool:
    return bool(_NUMERIC.match(_s(v)))


def _norm_token(t: str) -> str:
    return t.replace(",", "").strip("$%()")


def _tokens(text: str) -> List[str]:
    """Tokens for matching: BM25 tokenizer (numbers keep their dots, commas removed) plus unit symbols."""
    toks = [_norm_token(t) for t in tokenize(text)]
    if "%" in text:
        toks.append("%")
    return [t for t in toks if t]


def is_placeholder(name: str, i: int) -> bool:
    """'' / 'col_3' / 'Unnamed: 2' / the bare position index ('0', '1', ...) - but not a year such as '2022'."""
    return bool(_PLACEHOLDER.match(name)) or name == str(i)


def _common_prefix(names: Sequence[str]) -> str:
    """Longest '.'-terminated prefix shared by all names (a multi-level header's top level), if any."""
    if len(names) < 2:
        return ""
    first = names[0]
    cut = 0
    for i, ch in enumerate(first):
        if all(len(n) > i and n[i] == ch for n in names):
            if ch == ".":
                cut = i + 1
        else:
            break
    return first[:cut] if all(len(n) > cut for n in names) else ""


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with clean, unique string headers and string cells (see module docstring)."""
    df = df.copy()
    df.columns = [_s(c) for c in df.columns]
    df = df.apply(lambda col: col.map(lambda v: _DOT_LEADER.sub("", _s(v))))
    # promote header-like leading rows when the column names are placeholders
    placeholders = [is_placeholder(c, i) for i, c in enumerate(df.columns)]
    if df.shape[0] > 1 and sum(placeholders) >= max(1, len(placeholders) // 2 + 1):
        promoted = 0
        while promoted < 2 and len(df) > 1 and any(placeholders):
            row = df.iloc[0].tolist()
            filled = [v for v in row if v]
            # header-like: no digits at all and short values (data rows such as "1:30 pm | Talk" stay)
            if not filled or any(ch.isdigit() for v in filled for ch in v) or max(len(v) for v in filled) > 40:
                break
            new = []
            for c, ph, v in zip(df.columns, placeholders, row):
                base = "" if ph else c
                new.append((base + " " + v).strip() if v else base)
            df.columns = new
            placeholders = [not c for c in new]
            df = df.iloc[1:].reset_index(drop=True)
            promoted += 1
    names = ["" if is_placeholder(c, i) else _DUP_SUFFIX.sub("", c) for i, c in enumerate(df.columns)]
    prefix = _common_prefix(names)
    if prefix:
        names = [n[len(prefix):] for n in names]
    # unique, non-empty names
    seen: Dict[str, int] = {}
    out = []
    for i, n in enumerate(names):
        n = n.strip(" .") or f"col_{i}"
        if n in seen:
            seen[n] += 1
            n = f"{n} ({seen[n]})"
        else:
            seen[n] = 0
        out.append(n)
    df.columns = out
    hdr = [c.lower() for c in out]
    keep = [i for i in range(len(df)) if [v.lower() for v in df.iloc[i].tolist()] != hdr]
    return df.iloc[keep].reset_index(drop=True)


def row_label_column(df: pd.DataFrame) -> Optional[str]:
    """First column whose non-empty values are mostly non-numeric; else the first column (None if no columns)."""
    for c in df.columns:
        vals = [v for v in df[c].tolist() if v]
        if vals and sum(not is_numeric(v) for v in vals) > len(vals) / 2:
            return c
    return df.columns[0] if len(df.columns) else None


def _exact_only(term: str, min_fuzzy_len: int) -> bool:
    return any(ch.isdigit() for ch in term) or term in UNITS or len(term) < min_fuzzy_len


def matched_terms(terms: Sequence[str], text: str, threshold: float, min_fuzzy_len: int) -> Set[str]:
    """Query terms that match some token of `text` (exact for numbers/units/short words, fuzzy otherwise)."""
    toks = set(_tokens(text))
    if not toks:
        return set()
    out = set()
    for t in terms:
        if t in toks:
            out.add(t)
        elif not _exact_only(t, min_fuzzy_len):
            if any(not _exact_only(w, min_fuzzy_len) and fuzz.ratio(t, w) >= threshold for w in toks):
                out.add(t)
    return out


def _cells(df: pd.DataFrame, label_col: Optional[str], rows: Sequence[int], cols: Sequence[str]) -> List[Cell]:
    out: List[Cell] = []
    for r in rows:
        label = df.at[r, label_col] if label_col is not None else str(r)
        for c in cols:
            if c == label_col:
                continue
            v = df.at[r, c]
            if v:
                out.append((label, c, v))
    return out


def _is_group_row(df: pd.DataFrame, i: int, label_col: Optional[str]) -> bool:
    """A row with a label but no values ('EXPENDITURES:'): a group header for the rows below it."""
    return all(not df.at[i, c] for c in df.columns if c != label_col)


def _expand_groups(df: pd.DataFrame, rows: List[int], label_col: Optional[str]) -> List[int]:
    """Matched group-header rows -> the rows under them (until the next group header, at most MAX_GROUP_ROWS)."""
    out: List[int] = []
    for r in rows:
        if not _is_group_row(df, r, label_col):
            out.append(r)
            continue
        j = r + 1
        while j < len(df) and j - r <= MAX_GROUP_ROWS and not _is_group_row(df, j, label_col):
            out.append(j)
            j += 1
    return sorted(dict.fromkeys(out)) or rows


def _ordinal_row(query: str, df: pd.DataFrame, label_col: Optional[str]) -> Optional[int]:
    """'first' / 'second' / ... / 'last' / 'final' in the query -> index of that data row (group headers skipped)."""
    words = re.findall(r"[a-z0-9]+", query.lower())
    header_words = {w for c in df.columns for w in re.findall(r"[a-z0-9]+", c.lower())}
    # an ordinal that is part of a column name ("Final Wt") names the column, not a row
    n = next((ORDINALS[w] for w in words if w in ORDINALS and w not in header_words), None)
    if n is None:
        return None
    data_rows = [i for i in range(len(df)) if not _is_group_row(df, i, label_col)]
    if not data_rows or abs(n) > len(data_rows):
        return None
    return data_rows[n - 1] if n > 0 else data_rows[n]


def locate(query: str, table_df: pd.DataFrame, threshold: Optional[float] = None,
           min_fuzzy_len: Optional[int] = None, max_cells: Optional[int] = None
           ) -> Tuple[pd.DataFrame, List[Cell], str]:
    """(sub_df, cells [(row_label, column, value)], flag). See the module docstring for the rules."""
    cfg = get_config().get("localization", {})
    threshold = float(cfg.get("fuzzy_threshold", 85) if threshold is None else threshold)
    min_fuzzy_len = int(cfg.get("min_fuzzy_len", 4) if min_fuzzy_len is None else min_fuzzy_len)
    max_cells = int(cfg.get("max_cells", 12) if max_cells is None else max_cells)

    df = clean_table(table_df)
    if df.empty or not len(df.columns):
        return df, [], "no_match"
    terms = list(dict.fromkeys(_tokens(query)))
    if not terms:
        return df, [], "no_match"
    label_col = row_label_column(df)

    col_score = {c: len(matched_terms(terms, c, threshold, min_fuzzy_len)) for c in df.columns if c != label_col}
    row_score = {i: len(matched_terms(terms, df.at[i, label_col], threshold, min_fuzzy_len)) if label_col else 0
                 for i in range(len(df))}
    best_c = max(col_score.values(), default=0)
    best_r = max(row_score.values(), default=0)
    cols = [c for c, s in col_score.items() if best_c > 0 and s == best_c]
    rows = [i for i, s in row_score.items() if best_r > 0 and s == best_r]
    rows = _expand_groups(df, rows, label_col)
    if not rows:
        o = _ordinal_row(query, df, label_col)
        rows = [o] if o is not None else []

    if rows and cols:
        flag = "cell"
    elif rows:
        flag, cols = "row", [c for c in df.columns if c != label_col]
    elif cols:
        flag, rows = "column", list(range(len(df)))
    else:
        flag = "fallback"
        exact = [t for t in terms if _exact_only(t, min_fuzzy_len) and any(ch.isdigit() for ch in t)]
        keywords = [t for t in terms if not _exact_only(t, min_fuzzy_len)]
        hits: Dict[int, List[str]] = {}
        for i in range(len(df)):
            for c in df.columns:
                if c == label_col:
                    continue
                m = matched_terms(exact + keywords, df.at[i, c], threshold, min_fuzzy_len)
                if m:
                    hits.setdefault(i, []).append(c)
        if not hits:
            return df, [], "no_match"
        rows = sorted(hits)
        cols = [c for c in df.columns if c != label_col and any(c in hits[i] for i in rows)]

    keep_cols = ([label_col] if label_col else []) + [c for c in cols if c != label_col]
    sub = df.loc[rows, keep_cols].reset_index(drop=True)
    cells = _cells(df, label_col, rows, cols)[:max_cells]
    return sub, cells, flag


def cells_markdown(cells: Sequence[Cell]) -> str:
    """Localized cells as a small markdown table: | row | column | value |."""
    lines = ["| row | column | value |", "|---|---|---|"]
    for r, c, v in cells:
        lines.append(f"| {_s(r).replace('|', '/')} | {_s(c).replace('|', '/')} | {_s(v).replace('|', '/')} |")
    return "\n".join(lines)


def df_markdown(df: pd.DataFrame, max_rows: int, max_chars: int = 1200) -> str:
    """First `max_rows` rows of a (cleaned) table as markdown, cut at `max_chars`."""
    head = df.head(max_rows)
    lines = ["| " + " | ".join(_s(c).replace("|", "/") for c in head.columns) + " |",
             "|" + "---|" * len(head.columns)]
    for _, row in head.iterrows():
        lines.append("| " + " | ".join(_s(v).replace("|", "/") for v in row.tolist()) + " |")
    if len(df) > max_rows:
        lines.append(f"(... {len(df) - max_rows} more rows)")
    out = "\n".join(lines)
    return out if len(out) <= max_chars else out[: max_chars - 3] + "..."
