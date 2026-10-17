"""CLIP (clip-ViT-B-32 via sentence-transformers) index over figure crops; text queries use the same model."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence

import numpy as np

from docscout.indexing.dense_index import DenseIndex
from docscout.schema import Component


class ClipIndex(DenseIndex):
    """Image embeddings of figure crops (Component.image_path); queries are encoded with CLIP's text tower.

    Components without an image file are skipped (they stay searchable in the text-based indexes).
    """

    kind = "clip"

    def __init__(self, model_name: str = "clip-ViT-B-32", model: Any = None, batch_size: int = 16,
                 device: str = "cpu", root: Path = None) -> None:
        super().__init__(model_name, query_instruction="", model=model, batch_size=batch_size, device=device)
        from docscout.config import get_config

        self.root = root or get_config()["repo_root"]

    def build(self, components: Sequence[Component], progress: bool = True) -> "ClipIndex":
        from PIL import Image

        usable = [c for c in components if c.image_path and (self.root / c.image_path).exists()]
        self._set_rows(usable)
        images: List[Image.Image] = [Image.open(self.root / c.image_path).convert("RGB") for c in usable]
        if images:
            emb = self.model.encode(images, batch_size=self.batch_size, show_progress_bar=progress,
                                    normalize_embeddings=True, convert_to_numpy=True)
            self.emb = np.ascontiguousarray(emb, dtype=np.float32)
        else:
            self.emb = np.zeros((0, 512), dtype=np.float32)
        for im in images:
            im.close()
        import faiss

        self.faiss_index = faiss.IndexFlatIP(self.emb.shape[1])
        self.faiss_index.add(self.emb)
        self.info = {"model": self.model_name, "query_instruction": "", "dim": int(self.emb.shape[1]),
                     "skipped_no_image": len(components) - len(usable)}
        return self

    @classmethod
    def load(cls, src_dir: Path, model: Any = None, **kwargs: Any) -> "ClipIndex":
        import json

        import faiss

        meta = json.loads((src_dir / "meta.json").read_text(encoding="utf-8"))
        idx = cls(meta["model"], model=model)
        idx._read_meta(src_dir)
        idx.emb = np.load(src_dir / "embeddings.npy")
        idx.faiss_index = faiss.read_index(str(src_dir / "faiss.index"))
        return idx

    def doc_text(self, c: Component) -> str:  # documents are images, not text
        return f"<image {c.image_path}>"
