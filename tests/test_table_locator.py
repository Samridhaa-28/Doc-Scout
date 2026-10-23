"""Table value localization on hand-made tables: row+column -> one cell, row only, column only, fallback, no match,
plus header cleaning (multi-level prefix, placeholder headers, header row repeated in the body)."""
import pandas as pd

from docscout.localization.table_locator import cells_markdown, clean_table, locate, row_label_column


def revenue_table():
    return pd.DataFrame({
        "Segment": ["Revenue", "Operating income", "Net income"],
        "2022": ["1,200", "300", "210"],
        "2023": ["1,450", "380", "260"],
    })


def test_row_and_column_give_single_cell():
    sub, cells, flag = locate("What was the revenue in 2023?", revenue_table())
    assert flag == "cell"
    assert cells == [("Revenue", "2023", "1,450")]
    assert list(sub.columns) == ["Segment", "2023"] and len(sub) == 1


def test_year_is_matched_exactly_not_fuzzily():
    # "2022" must not match column "2023" (numbers are exact tokens)
    _, cells, flag = locate("net income 2022", revenue_table())
    assert flag == "cell" and cells == [("Net income", "2022", "210")]


def test_row_only_returns_the_row():
    _, cells, flag = locate("operating income?", revenue_table())
    assert flag == "row"
    assert cells == [("Operating income", "2022", "300"), ("Operating income", "2023", "380")]


def test_fuzzy_row_label_typo():
    _, cells, flag = locate("operatng incme 2022", revenue_table())
    assert flag == "cell" and cells == [("Operating income", "2022", "300")]


def test_column_only_returns_the_column():
    df = pd.DataFrame({"County": ["Carter", "Douglas"], "Population": ["5,200", "10,500"], "Poverty %": ["39.1", "35.4"]})
    _, cells, flag = locate("population", df)
    assert flag == "column"
    assert cells == [("Carter", "Population", "5,200"), ("Douglas", "Population", "10,500")]


def test_fallback_rows_containing_query_number():
    df = pd.DataFrame({"Item": ["Quaser", "Grelanter"], "Material": ["60-1120", "60-0280"], "Pounds": ["183.4", "235.3"]})
    _, cells, flag = locate("which ingredient weighs 235.3?", df)
    assert flag == "fallback"
    assert cells == [("Grelanter", "Pounds", "235.3")]


def test_no_match_returns_whole_table():
    sub, cells, flag = locate("Who signed the letter?", revenue_table())
    assert flag == "no_match" and cells == []
    assert sub.shape == revenue_table().shape


def test_clean_table_multilevel_prefix_and_placeholders():
    df = pd.DataFrame([["Dept", "Lab"], ["Science", "Monks Wood"], ["Dept", "Lab"]], columns=["0", "1"])
    c = clean_table(df)
    assert list(c.columns) == ["Dept", "Lab"]          # header row promoted
    assert c.values.tolist() == [["Science", "Monks Wood"]]  # repeated header row dropped
    df2 = pd.DataFrame({"Departments.DEPARTMENT": ["A"], "Departments.LABORATORY": ["B"]})
    assert list(clean_table(df2).columns) == ["DEPARTMENT", "LABORATORY"]  # shared top level stripped


def test_data_row_is_not_promoted_and_label_column_is_first_text_column():
    df = pd.DataFrame([["1:30 pm", "Nutrition talk"], ["2:10 pm", "Coffee"]], columns=["0", "1"])
    c = clean_table(df)
    assert list(c.columns) == ["col_0", "col_1"] and len(c) == 2
    assert row_label_column(c) == "col_1"  # times are numeric-like
    num_first = pd.DataFrame({"No": ["1", "2"], "Name": ["a", "b"]})
    assert row_label_column(num_first) == "Name"


def test_cells_markdown():
    md = cells_markdown([("Revenue", "2023", "1,450")])
    assert md.splitlines() == ["| row | column | value |", "|---|---|---|", "| Revenue | 2023 | 1,450 |"]


def test_group_header_row_expands_to_rows_below():
    df = pd.DataFrame({"Item": ["INCOME:", "Trusts", "Investments", "EXPENDITURES:", "Projects"],
                       "Amount": ["", "7,265,516", "1,634,431", "", "11,369,878"]})
    _, cells, flag = locate("What is under income?", df)
    assert flag == "row"
    assert cells == [("Trusts", "Amount", "7,265,516"), ("Investments", "Amount", "1,634,431")]


def test_ordinal_picks_nth_data_row():
    df = pd.DataFrame({"Treatment": ["MPA", "Megestrol acetate", "Transdermal"], "Effective": ["Yes", "Yes", "No"]})
    _, cells, flag = locate("What was the second treatment?", df)
    assert flag == "row" and cells == [("Megestrol acetate", "Effective", "Yes")]
    _, cells, _ = locate("final treatment effective?", df)
    assert cells == [("Transdermal", "Effective", "No")]


def test_ordinal_inside_a_column_name_is_not_a_row():
    df = pd.DataFrame({"Child": ["A.B.", "J.W."], "Initial Wt": ["61.5", "22.0"], "Final Wt": ["58.5", "23.5"]})
    _, cells, flag = locate("Final Wt of the child with initial Wt 61.5?", df)
    assert flag != "no_match" and ("A.B.", "Final Wt", "58.5") in cells  # not the last row (J.W.) only


def test_dot_leaders_are_removed():
    df = pd.DataFrame({"Coverage type": ["Medicaid 2 . . . . . .", "Total. . . . ."], "2022.Percent": ["18.8", "X"]})
    c = clean_table(df)
    assert c["Coverage type"].tolist() == ["Medicaid 2", "Total"]
    assert c["2022.Percent"].tolist() == ["18.8", "X"]  # a single decimal point is not a leader
