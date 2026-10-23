"""Tests for docscout.config."""
from pathlib import Path

from docscout.config import REPO_ROOT, get_config, get_path, load_config


def test_repo_root_is_project():
    assert (REPO_ROOT / "configs" / "config.yaml").exists()
    assert (REPO_ROOT / "pyproject.toml").exists()


def test_paths_resolved_relative_to_repo_root():
    cfg = get_config()
    for key, p in cfg["paths"].items():
        assert isinstance(p, Path) and p.is_absolute(), key
        assert REPO_ROOT in p.parents or p == REPO_ROOT, key
    assert get_path("questions_file") == REPO_ROOT / "data" / "raw" / "mpdocvqa" / "questions.jsonl"


def test_required_keys_present():
    cfg = get_config()
    assert cfg["parsing"]["ocr"] in (True, False)
    assert cfg["parsing"]["image_dpi"] > 0
    for m in ("bge", "minilm", "clip", "cross_encoder", "bge_query_instruction"):
        assert cfg["models"][m]
    assert cfg["retrieval"]["rerank_top_n"] == 30
    assert cfg["fusion"]["method"] in ("weighted_sum", "rrf")
    for qtype in ("textual", "numerical", "visual"):
        w = cfg["fusion"]["branch_weights"][qtype]
        assert set(w) == {"text", "table", "figure"}
        assert abs(sum(w.values()) - 1.0) < 1e-6
    assert cfg["feedback"]["step"] > 0 and cfg["feedback"]["cap"] >= cfg["feedback"]["step"]
    for k in ("provider", "model", "temperature", "cache_path"):
        assert k in cfg["llm"]
    assert cfg["eval"]["k_values"] == [1, 5, 10]
    assert cfg["eval"]["subset_size"] > 0


def test_get_config_returns_copy():
    cfg = get_config()
    cfg["retrieval"]["rerank_top_n"] = -1
    assert get_config()["retrieval"]["rerank_top_n"] == 30


def test_load_config_custom_file(tmp_path: Path):
    f = tmp_path / "c.yaml"
    f.write_text("paths:\n  foo: some/dir\nllm:\n  provider: groq\n  model: m\n", encoding="utf-8")
    cfg = load_config(f, root=tmp_path)
    assert cfg["paths"]["foo"] == (tmp_path / "some" / "dir").resolve()


def test_demo_store_replaces_store_paths_only(monkeypatch):
    from docscout.config import load_config

    monkeypatch.delenv("DOCSCOUT_STORE", raising=False)
    base = load_config()
    monkeypatch.setenv("DOCSCOUT_STORE", "demo")
    demo = load_config()
    assert demo["store"] == "demo" and base["store"] == "eval"
    for k in ("components_file", "graphs_dir", "tables_dir", "indexes_dir", "docs_dir"):
        assert demo["paths"][k] != base["paths"][k]
    for k in ("questions_file", "pdfs_dir", "results_dir"):
        assert demo["paths"][k] == base["paths"][k]
    assert demo["llm"]["cache_path"] == base["llm"]["cache_path"]  # LLM cache is shared
    monkeypatch.setenv("DOCSCOUT_STORE", "nope")
    import pytest
    with pytest.raises(KeyError):
        load_config()
