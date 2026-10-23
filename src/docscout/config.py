"""Load configs/config.yaml once, resolve paths against the repo root, and load .env."""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "configs" / "config.yaml"


def _resolve_paths(paths: Dict[str, str], root: Path) -> Dict[str, Path]:
    """Turn relative path strings into absolute Paths under `root`."""
    out: Dict[str, Path] = {}
    for key, value in paths.items():
        p = Path(value)
        out[key] = p if p.is_absolute() else (root / p).resolve()
    return out


def load_config(path: Optional[Path] = None, root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Read a YAML config file and return a dict with resolved paths (no caching)."""
    path = Path(path) if path else Path(os.environ.get("DOCSCOUT_CONFIG", DEFAULT_CONFIG_PATH))
    with path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}
    cfg["paths"] = _resolve_paths(cfg.get("paths", {}), root)
    store = os.environ.get("DOCSCOUT_STORE") or None
    if store:
        stores = cfg.get("stores") or {}
        if store not in stores:
            raise KeyError(f"unknown store {store!r} (DOCSCOUT_STORE); configured: {sorted(stores)}")
        cfg["paths"].update(_resolve_paths(stores[store], root))
    cfg["store"] = store or "eval"
    if "llm" in cfg and "cache_path" in cfg["llm"]:
        cfg["llm"]["cache_path"] = _resolve_paths({"c": cfg["llm"]["cache_path"]}, root)["c"]
    cfg["repo_root"] = root

    load_dotenv(root / ".env", override=False)
    llm = cfg.setdefault("llm", {})
    llm["provider"] = os.environ.get("LLM_PROVIDER") or llm.get("provider")
    llm["model"] = os.environ.get("LLM_MODEL") or llm.get("model")
    return cfg


@lru_cache(maxsize=1)
def _cached_config() -> Dict[str, Any]:
    return load_config()


def get_config() -> Dict[str, Any]:
    """Return the project config (loaded once). Returns a deep copy so callers can't mutate the cache."""
    return copy.deepcopy(_cached_config())


def use_store(name: Optional[str]) -> Dict[str, Any]:
    """Switch this process (and child processes) to a configured store ('demo'), or back to the eval store (None).

    Sets DOCSCOUT_STORE and reloads the cached config; returns the new config."""
    if name and name != "eval":
        os.environ["DOCSCOUT_STORE"] = name
    else:
        os.environ.pop("DOCSCOUT_STORE", None)
    _cached_config.cache_clear()
    return get_config()


def get_path(key: str) -> Path:
    """Shortcut for get_config()['paths'][key]."""
    return _cached_config()["paths"][key]


def get_secret(name: str) -> Optional[str]:
    """Read a secret (e.g. LLM_API_KEY) from the environment / .env. Never log the result."""
    _cached_config()  # ensures .env is loaded
    return os.environ.get(name) or None
