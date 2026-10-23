"""Cell hit rate of the table locator on TEST.

Questions: TEST, gold type numerical (eval/results/qtypes_gold.json), with >= 1 relevant TABLE component (qrels).
Hit: some localized cell VALUE contains a gold answer (same matcher as the qrels: eval.build_qrels.matches).

Two settings:
  oracle   - locate() on each relevant table (isolates the locator from retrieval)
  pipeline - the system as it runs: predicted query type, rerank, locate() only when predicted numerical and
             only on tables in the top localization.top_k_tables; a hit needs the relevant table there too
Also reported: flag counts, and the upper bound "the whole table contains the answer" (true by qrels construction).

Usage: .venv/Scripts/python -m eval.eval_locator [--limit N] [--no-pipeline] [--show 3]
Output: eval/results/locator.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from docscout.config import get_config
from docscout.localization.table_locator import locate
from docscout.schema import Component, iter_jsonl
from eval.build_qrels import matches
from eval.split import load_split


def cell_hit(cells, answers, threshold: int, with_label: bool = False) -> bool:
    """A gold answer in a localized cell VALUE (strict), or also in its row label (with_label)."""
    vals = [str(v) for (_, _, v) in cells] + ([str(r) for (r, _, _) in cells] if with_label else [])
    return any(matches(a, [v], [v] + v.split(), threshold) for v in vals for a in answers)


def classify_miss(question: str, answers: List[str], df: pd.DataFrame, threshold: int) -> str:
    """Why the locator missed (analysis only; never used to change rules).

    Finds the cells of the CLEANED table whose value contains a gold answer and compares them with locate():
      parse_issue      answer not in any parsed cell value or row label (garbled / split / outside the table)
      row_label        answer only appears as a row label (counts as a hit "with row label")
      no_match         locate() found nothing
      cell_cap         the gold cell is in the selected rows x columns but cut by localization.max_cells
      wrong_row        the gold column was selected, its row was not
      wrong_column     the gold row was selected, its column was not
      wrong_row_and_column
    """
    from docscout.localization.table_locator import clean_table, row_label_column

    sub, cells, flag = locate(question, df)
    c = clean_table(df)
    lab = row_label_column(c)
    hit = lambda v: any(matches(a, [str(v)], [str(v)] + str(v).split(), threshold) for a in answers)  # noqa: E731
    gold = [(i, col) for i in range(len(c)) for col in c.columns if col != lab and c.at[i, col] and hit(c.at[i, col])]
    if not gold:
        if lab is not None and any(hit(c.at[i, lab]) for i in range(len(c))):
            return "row_label"
        return "parse_issue"
    if flag == "no_match":
        return "no_match"
    sel_rows = {r for r, _, _ in cells} | set(sub[lab].tolist() if lab in sub.columns else [])
    sel_cols = set(sub.columns) - {lab}
    gold_rows = {c.at[i, lab] if lab else str(i) for i, _ in gold}
    gold_cols = {col for _, col in gold}
    if any((c.at[i, lab] if lab else str(i)) in sel_rows and col in sel_cols for i, col in gold):
        return "cell_cap"
    if gold_cols & sel_cols:
        return "wrong_row"
    if gold_rows & sel_rows:
        return "wrong_column"
    return "wrong_row_and_column"


def main(argv: Optional[List[str]] = None) -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pipeline", action="store_true", help="oracle setting only (no models)")
    ap.add_argument("--show", type=int, default=3, help="print N oracle misses")
    ap.add_argument("--split", choices=["test", "dev"], default="test", help="dev: for inspecting misses")
    ap.add_argument("--slots", choices=["off", "conditional", "both"], default="off",
                    help="slot reservation in the pipeline setting (oracle setting unaffected)")
    args = ap.parse_args(argv)
    res_dir: Path = cfg["paths"]["results_dir"]
    thr = int(cfg["eval"]["fuzzy_match_threshold"])
    split = load_split(cfg)
    test = set(split[args.split]["question_ids"])
    qrels = json.loads((res_dir / "qrels_component.json").read_text(encoding="utf-8"))
    gold = json.loads((res_dir / "qtypes_gold.json").read_text(encoding="utf-8"))["gold"]
    comps = {r["id"]: Component.from_dict(r) for r in iter_jsonl(cfg["paths"]["components_file"])}
    questions = {q["question_id"]: q for q in iter_jsonl(cfg["paths"]["questions_file"]) if q["question_id"] in test}
    qids = [q for q in questions if gold.get(q) == "numerical"
            and any(comps[c].type == "table" for c in qrels.get(q, {}) if c in comps)]
    if args.limit:
        qids = qids[: args.limit]
    print(f"{args.split.upper()} numerical questions with a relevant table: {len(qids)}")

    def read(c: Component) -> pd.DataFrame:
        p = Path(c.table_path)
        return pd.read_parquet(p if p.is_absolute() else cfg["repo_root"] / p)

    # ---------------- oracle
    rows, flags, misses = [], Counter(), []
    for qid in qids:
        q = questions[qid]
        tabs = [c for c in qrels[qid] if comps[c].type == "table"]
        hit, hit_label, whole, fl = False, False, False, []
        for cid in tabs:
            df = read(comps[cid])
            _, cells, flag = locate(q["question"], df)
            fl.append(flag)
            all_cells = [("", "", v) for v in df.astype(str).values.ravel().tolist()] + \
                        [("", "", str(c)) for c in df.columns]
            whole = whole or cell_hit(all_cells, q["answers"], thr)
            hit = hit or cell_hit(cells, q["answers"], thr)
            hit_label = hit_label or cell_hit(cells, q["answers"], thr, with_label=True)
        flags.update(fl[:1])
        rows.append({"qid": qid, "hit": hit, "hit_with_label": hit_label, "whole_table_has_answer": whole, "flags": fl})
        if not hit:
            misses.append((qid, tabs[0], fl[0]))
            kinds = [classify_miss(q["question"], q["answers"], read(comps[cid]), thr) for cid in tabs]
            # several relevant tables and the answer table is not the first one -> wrong table is not possible in
            # the oracle (all relevant tables are tried); keep the first table's reason
            rows[-1]["miss_kind"] = kinds[0]
    n = len(rows)
    oracle = {"n": n, "cell_hit_rate": sum(r["hit"] for r in rows) / max(n, 1),
              "cell_hit_rate_with_label": sum(r["hit_with_label"] for r in rows) / max(n, 1),
              "whole_table_has_answer": sum(r["whole_table_has_answer"] for r in rows) / max(n, 1),
              "flags_first_table": dict(flags),
              "hit_rate_by_flag": {f: sum(r["hit"] for r in rows if r["flags"][0] == f) /
                                   max(1, sum(1 for r in rows if r["flags"][0] == f)) for f in flags}}
    oracle["miss_kinds"] = dict(Counter(r["miss_kind"] for r in rows if "miss_kind" in r))
    print(f"oracle  : cell hit rate {oracle['cell_hit_rate']:.3f} ({sum(r['hit'] for r in rows)}/{n}) | with row label "
          f"{oracle['cell_hit_rate_with_label']:.3f} ({sum(r['hit_with_label'] for r in rows)}/{n}) | "
          f"whole table contains answer {oracle['whole_table_has_answer']:.3f}")
    print(f"          flags (first relevant table): {dict(flags)}")
    print("          hit rate by flag: " + ", ".join(f"{k} {v:.2f}" for k, v in oracle["hit_rate_by_flag"].items()))
    print(f"          misses by kind: {oracle['miss_kinds']}")
    for qid, cid, flag in misses[: args.show]:
        q = questions[qid]
        _, cells, _ = locate(q["question"], read(comps[cid]))
        print(f"  miss {qid} [{flag}] Q: {q['question']} | gold {q['answers'][:2]} | cells {cells[:3]}")

    out = {"oracle": oracle, "oracle_rows": rows}
    # ---------------- pipeline
    if not args.no_pipeline:
        from docscout.pipeline import DocScoutPipeline

        pipe = DocScoutPipeline(cfg=cfg, components=comps)
        pipe.warm_up()
        prow = []
        for qid in tqdm(qids, desc="pipeline", leave=False):
            q = questions[qid]
            r = pipe.answer(q["question"], q["doc_id"], top_k=int(cfg["localization"]["top_k_tables"]),
                            use_context=False, generate=False, slots=args.slots)
            rel = set(qrels[qid])
            loc = [e for e in r.evidence if e.localized_cells or e.meta.get("localize_flag")]
            rel_in_top = any(e.component.id in rel and e.component.type == "table" for e in r.evidence)
            hit = any(cell_hit(e.localized_cells, q["answers"], thr) for e in loc)
            hit_l = any(cell_hit(e.localized_cells, q["answers"], thr, with_label=True) for e in loc)
            if hit:
                why = None
            elif r.query_type != "numerical":
                why = "not_predicted_numerical"
            elif not rel_in_top:
                why = "wrong_table"
            else:
                why = "locator"
            prow.append({"qid": qid, "miss": why, "pred_type": r.query_type, "rel_table_in_top": rel_in_top,
                         "localized": bool(loc), "hit": hit, "hit_with_label": hit_l})
        m = len(prow)
        pipeline = {"n": m, "pred_numerical": sum(p["pred_type"] == "numerical" for p in prow),
                    "rel_table_in_top_k": sum(p["rel_table_in_top"] for p in prow),
                    "localized_any": sum(p["localized"] for p in prow),
                    "miss_kinds": dict(Counter(p["miss"] for p in prow if p["miss"])),
                    "cell_hit_rate": sum(p["hit"] for p in prow) / max(m, 1),
                    "cell_hit_rate_with_label": sum(p["hit_with_label"] for p in prow) / max(m, 1)}
        print(f"pipeline (slots {args.slots}): cell hit rate {pipeline['cell_hit_rate']:.3f} ({sum(p['hit'] for p in prow)}/{m}) | with row "
              f"label {pipeline['cell_hit_rate_with_label']:.3f} | "
              f"predicted numerical {pipeline['pred_numerical']}/{m} | relevant table in top-"
              f"{cfg['localization']['top_k_tables']} {pipeline['rel_table_in_top_k']}/{m} | "
              f"localized something {pipeline['localized_any']}/{m} | misses {pipeline['miss_kinds']}")
        out.update({"pipeline": pipeline, "pipeline_rows": prow})
    name = "locator.json" if args.split == "test" else f"locator_{args.split}.json"
    if args.slots != "off":
        name = name.replace(".json", f"_slots-{args.slots}.json")
    (res_dir / name).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {(res_dir / name).relative_to(cfg['repo_root']).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
