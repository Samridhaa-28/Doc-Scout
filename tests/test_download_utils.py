"""Tests for the pure helpers in docscout.dataset.mpdocvqa (no network)."""
import pytest

from docscout.dataset.mpdocvqa import page_number, parse_listish, resolve_fields


def test_parse_listish_string_and_native():
    assert parse_listish("['snbx0223_p3', 'snbx0223_p4']") == ["snbx0223_p3", "snbx0223_p4"]
    assert parse_listish('["0.28"]') == ["0.28"]
    assert parse_listish(["a"]) == ["a"]
    assert parse_listish(None) == []
    assert parse_listish("plain") == ["plain"]


def test_page_number_is_one_based():
    # dataset page ids are 0-based: '_p0' is the first page -> page_1.png
    assert page_number("ffbf0023_p0") == 1
    assert page_number("pybv0228_p80") == 81
    with pytest.raises(ValueError):
        page_number("no_page_suffix")


def test_resolve_fields_real_names():
    feats = ["questionId", "question", "doc_id", "page_ids", "answers", "answer_page_idx", "data_split", "image_1"]
    m = resolve_fields(feats)
    assert m["question_id"] == "questionId" and m["answer_page_idx"] == "answer_page_idx"
    assert m["question_types"] is None
    with pytest.raises(SystemExit):
        resolve_fields(["question", "doc_id"])
