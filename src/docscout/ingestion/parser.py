"""Parse documents with Docling (OCR + layout) into Components.

Two inputs, one code path:
  * MP-DocVQA: the page PNGs of a document are joined into one PDF (PyMuPDF) and parsed.
    PDF page j (0-based) is page `meta.json["page_numbers"][j]`; bboxes are in original PNG pixels.
  * Demo PDF: parsed directly; bboxes are in pixels of the page rendered at parsing.image_dpi.

Coordinate contract: Component.bbox = (x0, y0, x1, y1) with TOP-LEFT origin, in the pixel
space of the page image we display (manifest `pages[i].image_path`); page_width/page_height
are that image's size.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from docscout.config import get_config
from docscout.schema import BBox, Component, make_component_id

log = logging.getLogger(__name__)

HEADING_LABELS = {"section_header", "title"}
CAPTION_LABELS = {"caption"}
TABLE_LABELS = {"table", "document_index"}
FIGURE_LABELS = {"picture", "chart"}
# Every other text-bearing label (text, paragraph, list_item, page_header, page_footer,
# footnote, formula, code, checkbox_*, handwritten_text, field_* ...) becomes a paragraph.
# Page headers/footers are kept: DocVQA answers are often in them.
PAGE_MARKER_RE = re.compile(r"^\s*page\s+\d+\b", re.IGNORECASE)


@dataclass
class PageInfo:
    """One displayable page: our page number, the image's pixel size and its path."""

    page: int  # 1-based page number used in Component.page / ids
    width: int  # pixels of the displayed image
    height: int
    image_path: str  # repo-relative posix path of the displayed page image


# ------------------------------------------------------------------ helpers

def rel_path(p: Path) -> str:
    """Repo-relative posix path (we never store absolute paths)."""
    root = get_config()["repo_root"]
    try:
        return Path(p).resolve().relative_to(root).as_posix()
    except ValueError:
        return Path(p).as_posix()


def abs_path(p: str) -> Path:
    """Inverse of rel_path."""
    q = Path(p)
    return q if q.is_absolute() else get_config()["repo_root"] / q


def safe_doc_id(name: str) -> str:
    """Doc id from a file stem: lowercase, [a-z0-9-] only (no '_' so ids stay parseable)."""
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "doc"


def docling_bbox_to_pixels(
    l: float, t: float, r: float, b: float, bottom_left: bool,
    page_w_pt: float, page_h_pt: float, img_w: float, img_h: float,
) -> BBox:
    """Convert a Docling bbox (PDF points) to top-left pixel coords of an img_w x img_h page image.

    With a bottom-left origin, t/b are measured from the page bottom (t > b). The result is
    sorted (x0 < x1, y0 < y1) and clamped to the image.
    """
    sx, sy = img_w / page_w_pt, img_h / page_h_pt
    if bottom_left:
        y_top, y_bot = page_h_pt - t, page_h_pt - b
    else:
        y_top, y_bot = t, b
    x0, x1 = sorted((l * sx, r * sx))
    y0, y1 = sorted((y_top * sy, y_bot * sy))
    clamp = lambda v, hi: max(0.0, min(float(hi), v))  # noqa: E731
    return (round(clamp(x0, img_w), 2), round(clamp(y0, img_h), 2),
            round(clamp(x1, img_w), 2), round(clamp(y1, img_h), 2))


def images_to_pdf(image_paths: Sequence[Path], out_pdf: Path, dpi: float) -> List[Tuple[float, float]]:
    """Join page images into one PDF; each page is sized img_px * 72 / dpi points. Returns page sizes (pt)."""
    import pymupdf
    from PIL import Image

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf = pymupdf.open()
    sizes = []
    for img_path in image_paths:
        with Image.open(img_path) as im:
            w_px, h_px = im.size
        w_pt, h_pt = w_px * 72.0 / dpi, h_px * 72.0 / dpi
        page = pdf.new_page(width=w_pt, height=h_pt)
        page.insert_image(page.rect, filename=str(img_path))
        sizes.append((w_pt, h_pt))
    pdf.save(str(out_pdf), deflate=True)
    pdf.close()
    return sizes


def render_pdf_pages(pdf_path: Path, out_dir: Path, dpi: int) -> List[Tuple[int, int, Path]]:
    """Render every PDF page to PNG at `dpi` (cached). Returns (width_px, height_px, path) per page."""
    import pymupdf

    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    with pymupdf.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf, start=1):
            png = out_dir / f"page_{i}.png"
            if png.exists():
                from PIL import Image

                with Image.open(png) as im:
                    out.append((im.size[0], im.size[1], png))
                continue
            pix = page.get_pixmap(dpi=dpi)
            pix.save(str(png))
            out.append((pix.width, pix.height, png))
    return out


def pdf_backend_for(source: str, cfg: Optional[Dict[str, Any]] = None) -> str:
    """Configured Docling PDF backend name for a source ('mpdocvqa' | 'pdf')."""
    cfg = cfg or get_config()
    return (cfg["parsing"].get("pdf_backend") or {}).get(source, "docling_parse")


def build_converter(cfg: Optional[Dict[str, Any]] = None, backend: str = "docling_parse"):
    """Docling DocumentConverter for PDFs with RapidOCR (backend = parsing.ocr_backend), table structure, CPU.

    `backend`: 'docling_parse' (Docling default) or 'pypdfium2' (PyPdfiumDocumentBackend).
    """
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    cfg = cfg or get_config()
    p = cfg["parsing"]
    opts = PdfPipelineOptions()
    opts.do_ocr = bool(p["ocr"])
    # Same settings Docling's auto picker uses for RapidOCR (lang=[] -> engine default model), backend from config.
    opts.ocr_options = RapidOcrOptions(backend=p.get("ocr_backend", "onnxruntime"), lang=[])
    opts.do_table_structure = True
    opts.generate_page_images = False
    opts.generate_picture_images = False
    opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=int(p.get("num_threads", 8)))
    fmt = PdfFormatOption(pipeline_options=opts)
    if backend == "pypdfium2":
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

        fmt = PdfFormatOption(pipeline_options=opts, backend=PyPdfiumDocumentBackend)
    elif backend != "docling_parse":
        raise ValueError(f"unknown pdf backend {backend!r}")
    conv = DocumentConverter(format_options={InputFormat.PDF: fmt})
    conv.docscout_backend = backend  # recorded in each manifest's parse_stats
    return conv


def _table_dataframe(item, doc):
    """Export a Docling table to a DataFrame with unique, non-empty string column names."""
    import pandas as pd

    df = item.export_to_dataframe(doc=doc)
    if df is None or df.empty:
        return pd.DataFrame()
    cols, seen = [], {}
    for j, c in enumerate(df.columns):
        name = str(c).strip() or f"col_{j}"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    df.columns = cols
    return df.astype(str)


def table_plain_text(df) -> str:
    """Plain-text version of a table: header line then one line per row, cells joined by ' | '."""
    if df is None or df.empty:
        return ""
    lines = [" | ".join(str(c) for c in df.columns)]
    lines += [" | ".join(str(v) for v in row) for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _vgap(a: BBox, b: BBox) -> float:
    """Vertical gap between two boxes (0 if they overlap vertically)."""
    return max(0.0, max(a[1], b[1]) - min(a[3], b[3]))


def _xoverlap(a: BBox, b: BBox) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


def link_captions_nearest(components: List[Component], max_gap_frac: float) -> int:
    """Fallback: link each uncaptioned table/figure to the nearest unlinked caption above/below on the same page."""
    linked = 0
    free = [c for c in components if c.type == "caption" and "caption_of" not in c.meta]
    for target in components:
        if target.type not in ("table", "figure") or "caption_id" in target.meta:
            continue
        best, best_gap = None, None
        for cap in free:
            if cap.page != target.page or "caption_of" in cap.meta:
                continue
            gap = _vgap(cap.bbox, target.bbox)
            if gap > max_gap_frac * target.page_height or _xoverlap(cap.bbox, target.bbox) <= 0:
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = cap, gap
        if best is not None:
            target.meta["caption_id"] = best.id
            target.meta["caption_link"] = "nearest"
            best.meta["caption_of"] = target.id
            best.meta["caption_link"] = "nearest"
            linked += 1
    return linked


# ------------------------------------------------------------------ core

def docling_to_components(
    doc, doc_id: str, pages: Dict[int, PageInfo], cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Component], Dict[str, Any]]:
    """Map a DoclingDocument to Components. `pages` maps Docling page_no (1-based PDF page) -> PageInfo.

    Never raises for a single bad item/page: problems are logged and counted in the returned stats.
    """
    import pandas as pd
    from docling_core.types.doc import CoordOrigin, PictureItem, TableItem, TextItem
    from docling_core.types.doc.document import ContentLayer

    cfg = cfg or get_config()
    tables_dir: Path = cfg["paths"]["tables_dir"]
    figures_dir: Path = cfg["paths"]["figures_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    comps: List[Component] = []
    ref_to_id: Dict[str, str] = {}
    counters: Dict[Tuple[int, str], int] = {}
    stats = {"items_skipped": 0, "item_errors": 0, "unmapped_pages": set()}
    pending_captions: List[Tuple[str, List[str]]] = []  # (target id, docling caption refs)
    current_section: Optional[str] = None
    page_images: Dict[int, Any] = {}

    def page_image(p: PageInfo):
        if p.page not in page_images:
            from PIL import Image

            page_images[p.page] = Image.open(abs_path(p.image_path)).convert("RGB")
        return page_images[p.page]

    def new_id(page: int, ctype: str) -> str:
        n = counters.get((page, ctype), 0)
        counters[(page, ctype)] = n + 1
        return make_component_id(doc_id, page, ctype, n)

    layers = {ContentLayer.BODY, ContentLayer.FURNITURE}
    seen_refs = set()

    def handle(item) -> None:
        nonlocal current_section
        label = str(getattr(item, "label", ""))
        if not getattr(item, "prov", None):
            stats["items_skipped"] += 1
            return
        prov = item.prov[0]
        pinfo = pages.get(prov.page_no)
        if pinfo is None:
            stats["unmapped_pages"].add(prov.page_no)
            stats["items_skipped"] += 1
            return
        def to_px(pv) -> Optional[Tuple[int, BBox]]:
            pi = pages.get(pv.page_no)
            if pi is None:
                return None
            dp, bb = doc.pages[pv.page_no], pv.bbox
            return pi.page, docling_bbox_to_pixels(
                bb.l, bb.t, bb.r, bb.b, bb.coord_origin == CoordOrigin.BOTTOMLEFT,
                dp.size.width, dp.size.height, pi.width, pi.height,
            )

        bbox = to_px(prov)[1]
        # A text item continued in another column/page has several provenance regions.
        regions = [r for r in (to_px(pv) for pv in item.prov) if r is not None]

        original_label = label
        if label in CAPTION_LABELS and PAGE_MARKER_RE.match((getattr(item, "text", "") or "")):
            label = "page_marker"  # Docling sometimes labels running headers ("Page 5 ...") as captions
        if isinstance(item, TableItem) or label in TABLE_LABELS:
            ctype = "table"
        elif isinstance(item, PictureItem) or label in FIGURE_LABELS:
            ctype = "figure"
        elif label in HEADING_LABELS:
            ctype = "heading"
        elif label in CAPTION_LABELS:
            ctype = "caption"
        elif isinstance(item, TextItem):
            ctype = "paragraph"
        else:
            stats["items_skipped"] += 1
            return

        cid = new_id(pinfo.page, ctype)
        meta: Dict[str, Any] = {"docling_label": label, "docling_ref": item.self_ref}
        if original_label != label:
            meta["docling_label_original"] = original_label
        if len(regions) > 1:
            meta["regions"] = [{"page": pg, "bbox": list(bx)} for pg, bx in regions]
        text, table_path, image_path = "", None, None

        if ctype == "table":
            df = _table_dataframe(item, doc)
            text = table_plain_text(df)
            if not df.empty:
                out = tables_dir / f"{cid}.parquet"
                df.to_parquet(out, index=False)
                table_path = rel_path(out)
            meta.update({"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])})
            pending_captions.append((cid, [c.cref for c in item.captions]))
        elif ctype == "figure":
            inner = [c.resolve(doc) for c in item.children]
            inner_text = " ".join(t.text for t in inner if isinstance(t, TextItem) and t.text and str(t.label) != "caption")
            text = inner_text.strip()
            if inner_text:
                meta["inner_text"] = inner_text.strip()
            x0, y0, x1, y1 = bbox
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                out = figures_dir / f"{cid}.png"
                page_image(pinfo).crop((int(x0), int(y0), int(round(x1)), int(round(y1)))).save(out)
                image_path = rel_path(out)
            pending_captions.append((cid, [c.cref for c in item.captions]))
        else:
            text = (item.text or "").strip()
            if not text:
                counters[(pinfo.page, ctype)] -= 1
                stats["items_skipped"] += 1
                return

        if ctype == "heading":
            current_section = cid
        comp = Component(
            id=cid, doc_id=doc_id, page=pinfo.page, type=ctype, text=text, bbox=bbox,
            page_width=pinfo.width, page_height=pinfo.height, order=len(comps),
            section_id=current_section, table_path=table_path, image_path=image_path, meta=meta,
        )
        comps.append(comp)
        ref_to_id[item.self_ref] = cid

    for item, _level in doc.iterate_items(included_content_layers=layers):
        if item.self_ref in seen_refs:
            continue
        seen_refs.add(item.self_ref)
        try:
            handle(item)
        except Exception as e:  # noqa: BLE001 - one bad item must not stop the document
            stats["item_errors"] += 1
            log.warning("%s: item %s failed: %s: %s", doc_id, getattr(item, "self_ref", "?"), type(e).__name__, e)

    # Captions referenced by tables/figures but not visited (e.g. children of pictures).
    by_id = {c.id: c for c in comps}
    for target_id, crefs in pending_captions:
        for cref in crefs:
            if cref not in ref_to_id:
                try:
                    cap_item = next(i for i in doc.texts if i.self_ref == cref)
                    if cap_item.self_ref not in seen_refs:
                        seen_refs.add(cap_item.self_ref)
                        handle(cap_item)
                        by_id = {c.id: c for c in comps}
                except Exception as e:  # noqa: BLE001
                    log.warning("%s: caption %s failed: %s", doc_id, cref, e)
            cap_id = ref_to_id.get(cref)
            if cap_id and by_id[cap_id].type == "caption" and "caption_id" not in by_id[target_id].meta:
                by_id[target_id].meta.update({"caption_id": cap_id, "caption_link": "docling"})
                by_id[cap_id].meta.update({"caption_of": target_id, "caption_link": "docling"})
                break

    nearest = link_captions_nearest(comps, float(cfg["parsing"].get("caption_max_gap", 0.15)))
    # reading order may have changed if late captions were appended; keep order = list position
    for i, c in enumerate(comps):
        c.order = i
    for im in page_images.values():
        im.close()

    pages_with_items = {c.page for c in comps}
    stats.update({
        "unmapped_pages": sorted(stats["unmapped_pages"]),
        "captions_linked_docling": sum(1 for c in comps if c.meta.get("caption_link") == "docling" and c.type != "caption"),
        "captions_linked_nearest": nearest,
        "pages_total": len(pages),
        "pages_with_components": len(pages_with_items),
        "pages_without_components": sorted(p.page for p in pages.values() if p.page not in pages_with_items),
    })
    return comps, stats


def convert_pdf(converter, pdf_path: Path):
    """Run Docling; returns (DoclingDocument | None, status str, error messages)."""
    res = converter.convert(str(pdf_path), raises_on_error=False)
    errors = [f"{getattr(e, 'component_type', '')}: {getattr(e, 'error_message', e)}" for e in (res.errors or [])]
    return (res.document if res.document is not None else None), str(res.status.value), errors


def write_manifest(doc_id: str, source: str, pages: Dict[int, PageInfo], extra: Dict[str, Any], cfg=None) -> Path:
    """Save data/processed/docs/{doc_id}.json describing the displayable pages."""
    cfg = cfg or get_config()
    out = cfg["paths"]["docs_dir"] / f"{doc_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {"doc_id": doc_id, "source": source,
            "pages": [asdict(p) for p in sorted(pages.values(), key=lambda p: p.page)], **extra}
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def load_manifest(doc_id: str, cfg=None) -> Dict[str, Any]:
    cfg = cfg or get_config()
    return json.loads((cfg["paths"]["docs_dir"] / f"{doc_id}.json").read_text(encoding="utf-8"))


def parse_mpdocvqa(doc_id: str, converter=None, cfg=None) -> Tuple[List[Component], Dict[str, Any]]:
    """Input B: join an MP-DocVQA document's PNGs into a PDF and parse it."""
    cfg = cfg or get_config()
    img_dir: Path = cfg["paths"]["mpdocvqa_images_dir"] / doc_id
    meta = json.loads((img_dir / "meta.json").read_text(encoding="utf-8"))
    pages: Dict[int, PageInfo] = {}
    pngs = []
    for j, (num, (w, h)) in enumerate(zip(meta["page_numbers"], meta["page_sizes"]), start=1):
        png = img_dir / f"page_{num}.png"
        pngs.append(png)
        pages[j] = PageInfo(page=int(num), width=int(w), height=int(h), image_path=rel_path(png))
    pdf_path = cfg["paths"]["combined_pdfs_dir"] / f"{doc_id}.pdf"
    if not pdf_path.exists():
        images_to_pdf(pngs, pdf_path, float(cfg["parsing"]["mpdocvqa_dpi"]))
    return _parse(pdf_path, doc_id, pages, "mpdocvqa", converter, cfg)


def parse_pdf(pdf_path: Path, doc_id: Optional[str] = None, converter=None, cfg=None) -> Tuple[List[Component], Dict[str, Any]]:
    """Input A: parse a digital PDF; display images are its pages rendered at parsing.image_dpi."""
    cfg = cfg or get_config()
    doc_id = doc_id or safe_doc_id(Path(pdf_path).stem)
    rendered = render_pdf_pages(Path(pdf_path), cfg["paths"]["pages_dir"] / doc_id, int(cfg["parsing"]["image_dpi"]))
    pages = {j: PageInfo(page=j, width=w, height=h, image_path=rel_path(p)) for j, (w, h, p) in enumerate(rendered, start=1)}
    return _parse(Path(pdf_path), doc_id, pages, "pdf", converter, cfg)


def _parse(pdf_path: Path, doc_id: str, pages: Dict[int, PageInfo], source: str, converter, cfg) -> Tuple[List[Component], Dict[str, Any]]:
    converter = converter or build_converter(cfg, backend=pdf_backend_for(source, cfg))
    try:
        doc, status, errors = convert_pdf(converter, pdf_path)
    except Exception as e:  # noqa: BLE001 - one bad document must not stop ingestion
        log.error("%s: conversion crashed: %s: %s", doc_id, type(e).__name__, e)
        doc, status, errors = None, "failure", [f"{type(e).__name__}: {e}"]
    stats: Dict[str, Any] = {"status": status, "errors": errors,
                             "pdf_backend": getattr(converter, "docscout_backend", "unknown")}
    comps: List[Component] = []
    if doc is not None:
        comps, s = docling_to_components(doc, doc_id, pages, cfg)
        stats.update(s)
        stats["docling_pages"] = len(doc.pages)
    else:
        stats.update({"pages_total": len(pages), "pages_with_components": 0,
                      "pages_without_components": sorted(p.page for p in pages.values())})
    write_manifest(doc_id, source, pages, {"pdf_path": rel_path(pdf_path), "parse_stats": stats}, cfg)
    return comps, stats
