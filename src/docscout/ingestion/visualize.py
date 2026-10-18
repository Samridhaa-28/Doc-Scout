"""Draw component bboxes (coloured by type, labelled with ids) on each page image of a document.

Draws on exactly the images whose pixel space the bboxes use (manifest pages[i].image_path).
Output: data/processed/debug/{doc_id}/page_{n}.png

Command line: .venv/Scripts/python -m scripts.visualize_components --doc snbx0223
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from docscout.config import get_config
from docscout.ingestion.parser import abs_path, load_manifest
from docscout.schema import Component, iter_jsonl

COLORS = {
    "paragraph": (30, 120, 255),
    "heading": (230, 40, 40),
    "table": (20, 170, 60),
    "figure": (200, 0, 200),
    "caption": (255, 140, 0),
}


def draw_page(img: Image.Image, comps: List[Component], extra: List = ()) -> Image.Image:
    """Draw main bboxes with id labels; `extra` = (component, bbox) continuation regions drawn with a '+' label."""
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    size = max(12, img.width // 110)
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    width = max(2, img.width // 500)
    items = [(c, c.bbox, c.id.split("_", 1)[1]) for c in comps]  # drop doc_id prefix to save space
    items += [(c, bb, "+" + c.id.split("_", 1)[1]) for c, bb in extra]
    for c, (x0, y0, x1, y1), label in items:
        col = COLORS[c.type]
        draw.rectangle([x0, y0, x1, y1], outline=col + (255,), width=width, fill=col + (28,))
        tb = draw.textbbox((x0, y0), label, font=font)
        ty = max(0, y0 - (tb[3] - tb[1]) - 2)
        draw.rectangle([x0, ty, x0 + (tb[2] - tb[0]) + 4, ty + (tb[3] - tb[1]) + 2], fill=col + (220,))
        draw.text((x0 + 2, ty), label, fill=(255, 255, 255), font=font)
    return img


def render_doc(doc_id: str, cfg: Optional[Dict] = None) -> List[Tuple[int, int, Path]]:
    """Draw every page of a document to data/processed/debug/{doc_id}/page_{n}.png.

    Returns [(page, n_components, png_path)]; empty if the document has no components.
    """
    cfg = cfg or get_config()
    comps = [Component.from_dict(r) for r in iter_jsonl(cfg["paths"]["components_file"]) if r["doc_id"] == doc_id]
    if not comps:
        return []
    manifest = load_manifest(doc_id, cfg)
    by_page = defaultdict(list)
    extra = defaultdict(list)  # continuation regions (meta["regions"] beyond the first)
    for c in comps:
        by_page[c.page].append(c)
        for r in c.meta.get("regions", [])[1:]:
            extra[r["page"]].append((c, tuple(r["bbox"])))
    out_dir: Path = cfg["paths"]["debug_dir"] / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for p in manifest["pages"]:
        with Image.open(abs_path(p["image_path"])) as im:
            assert im.size == (p["width"], p["height"]), f"page {p['page']}: image {im.size} != manifest"
            out = draw_page(im, by_page.get(p["page"], []), extra.get(p["page"], []))
        path = out_dir / f"page_{p['page']}.png"
        out.save(path)
        written.append((p["page"], len(by_page.get(p["page"], [])), path))
    return written
