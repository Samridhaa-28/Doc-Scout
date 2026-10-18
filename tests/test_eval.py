"""Tests for ANLS, page-from-component aggregation, qrels matching and the random baseline."""
import pytest

from docscout.schema import Component
from eval.build_qrels import is_numeric_answer, matches, norm, page_key, relevant_components
from eval.metrics import anls, anls_single, evaluate_rankings, pages_from_components
from eval.run_eval import merge_max, random_baseline


# ------------------------------------------------------------------ ANLS

def test_anls_exact_and_case_whitespace():
    assert anls_single("University of California", ["university of california"]) == 1.0
    assert anls_single("  0.28 ", ["0.28"]) == 1.0


def test_anls_partial_and_threshold():
    # 1 edit over 8 chars -> NL = 0.125 -> 0.875
    assert anls_single("Monsanto", ["Monsanta"]) == pytest.approx(0.875)
    # NL >= 0.5 -> 0
    assert anls_single("abc", ["xyz"]) == 0.0
    # best over several gold answers
    assert anls_single("1975", ["1976", "1975"]) == 1.0


def test_anls_mean():
    assert anls(["a", "b"], [["a"], ["zzzz"]]) == pytest.approx(0.5)
    assert anls([], []) == 0.0


# ------------------------------------------------------------------ page aggregation

def test_pages_from_components_uses_max():
    comp_page = {"d_p1_paragraph_0": "d_p1", "d_p1_table_0": "d_p1", "d_p2_paragraph_0": "d_p2", "d_p3_figure_0": "d_p3"}
    ranking = [("d_p2_paragraph_0", 0.9), ("d_p1_paragraph_0", 0.5), ("d_p1_table_0", 0.8), ("d_p3_figure_0", 0.1)]
    assert pages_from_components(ranking, comp_page) == [("d_p2", 0.9), ("d_p1", 0.8), ("d_p3", 0.1)]


def test_merge_max_keeps_highest_score():
    assert merge_max([[("a", 1.0), ("b", 0.2)], [("b", 0.7), ("c", 0.1)]]) == [("a", 1.0), ("b", 0.7), ("c", 0.1)]


def test_evaluate_rankings_with_ranx():
    qrels = {"q1": {"d_p2": 1}, "q2": {"d_p1": 1}, "q3": {"d_p5": 1}}
    runs = {"q1": [("d_p2", 0.9), ("d_p1", 0.1)], "q2": [("d_p3", 0.9), ("d_p1", 0.5)]}  # q3 has no run
    m = evaluate_rankings(qrels, runs, ["recall@1", "recall@5", "mrr"])
    assert m["recall@1"] == pytest.approx(1 / 3)
    assert m["recall@5"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)


def test_random_baseline():
    info = {"a": {"num_pages": 1}, "b": {"num_pages": 4}}
    b = random_baseline(["a", "b"], info)
    assert b["recall@1"] == pytest.approx((1 + 0.25) / 2)
    assert b["recall@5"] == pytest.approx(1.0)


# ------------------------------------------------------------------ qrels matching

def test_norm_and_numeric_kind():
    assert norm("University of California, San Diego!") == "universityofcaliforniasandiego"
    assert is_numeric_answer("0.28") and is_numeric_answer("$1,234") and is_numeric_answer("26.9%")
    assert is_numeric_answer("12/03/1998") and not is_numeric_answer("University of California")
    assert page_key("pybv0228", 81) == "pybv0228_p81"


def test_matching_long_fuzzy_and_short_exact():
    text = ["The University of Califomia, San Diego hosted the meeting."]  # OCR typo
    assert matches("University of California", text, text[0].split())
    # short answers need an exact word / cell match, not a substring
    assert matches("5", ["Page 5 of 10"], ["Page", "5", "of", "10"])
    assert not matches("5", ["Total 1,500"], ["Total", "1,500"])


def test_short_text_does_not_match_long_answer():
    # regression: partial_ratio('(910) 741-5327', '0') == 100 because it aligns the shorter string
    assert not matches("(910) 741-5327", ["0"], ["0"])
    assert not matches("Joyce Bagby", ["A"], ["A"])
    assert not matches("11:14 to 11:39 a.m.", ["11:39 a.m."], ["11:39", "a.m."])  # only part of the answer
    assert matches("KOOL KS", ["KOOL KS."], ["KOOL", "KS."])  # whole-text near match still allowed


def test_relevant_components_uses_table_cells(tmp_path, monkeypatch):
    import pandas as pd

    import eval.build_qrels as bq

    df = pd.DataFrame({"County": ["Osage", "Butler"], "Pop.": ["10,700", "35,500"]})
    p = tmp_path / "t.parquet"
    df.to_parquet(p, index=False)
    bq.table_cells.cache_clear()
    table = Component(id="d_p1_table_0", doc_id="d", page=1, type="table", text="", text_form="Table: counties",
                      table_path=str(p))
    para = Component(id="d_p1_paragraph_0", doc_id="d", page=1, type="paragraph", text="unrelated", text_form="unrelated")
    q = {"answers": ["35,500"]}
    assert relevant_components(q, [table, para]) == ["d_p1_table_0"]  # found only via the parquet cells
