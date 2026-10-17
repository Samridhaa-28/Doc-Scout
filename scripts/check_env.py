"""Import every project library, print versions and the torch device, then 'all good'.

Usage: .venv/Scripts/python -m scripts.check_env
"""
from __future__ import annotations

import importlib
import sys
from importlib import metadata
from typing import List, Tuple

# (import name, distribution name)
LIBRARIES: List[Tuple[str, str]] = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("docling", "docling"),
    ("pymupdf", "pymupdf"),
    ("PIL", "pillow"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("networkx", "networkx"),
    ("datasets", "datasets"),
    ("rapidfuzz", "rapidfuzz"),
    ("rank_bm25", "rank-bm25"),
    ("sentence_transformers", "sentence-transformers"),
    ("faiss", "faiss-cpu"),
    ("groq", "groq"),
    ("dotenv", "python-dotenv"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("multipart", "python-multipart"),
    ("pydantic", "pydantic"),
    ("requests", "requests"),
    ("streamlit", "streamlit"),
    ("ranx", "ranx"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
    ("yaml", "pyyaml"),
    ("pytest", "pytest"),
    ("docscout", "docscout"),
]


def main() -> int:
    print(f"python {sys.version.split()[0]} ({sys.executable})")
    failed: List[str] = []
    for mod, dist in LIBRARIES:
        try:
            importlib.import_module(mod)
            try:
                ver = metadata.version(dist)
            except metadata.PackageNotFoundError:
                ver = "?"
            print(f"  OK   {dist:<24} {ver}")
        except Exception as e:  # noqa: BLE001 - report any import failure
            print(f"  FAIL {dist:<24} {type(e).__name__}: {e}")
            failed.append(dist)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch device: {device} (cuda available: {torch.cuda.is_available()}, threads: {torch.get_num_threads()})")

    from docscout.config import get_config

    cfg = get_config()
    print(f"config loaded: repo_root={cfg['repo_root']}, {len(cfg)} top-level keys")

    if failed:
        print(f"FAILED imports: {', '.join(failed)}")
        return 1
    print("all good")
    return 0


if __name__ == "__main__":
    sys.exit(main())
