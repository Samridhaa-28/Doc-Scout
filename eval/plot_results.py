"""Ablation chart from TEST results: grouped bars per system for page R@1, page MRR (documents with >= 2 pages),
component R@1 and component MRR. One colour per metric (validated categorical palette), value labels on bars.

R@1 / MRR are plotted instead of the R@5 in PHASES.md: documents are short, so page R@5 is near ceiling
(random R@5 = 0.77) and hides the differences between systems.

Usage: .venv/Scripts/python -m eval.plot_results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from docscout.config import get_config
from eval.run_eval import ANALYSIS_SYSTEMS, _load_results

METRICS = [  # (level, metric, label, colour)
    ("page_multi", "recall@1", "Page R@1", "#2a78d6"),
    ("page_multi", "mrr", "Page MRR", "#eb6834"),
    ("component", "recall@1", "Component R@1", "#1baf7a"),
    ("component", "mrr", "Component MRR", "#eda100"),
]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot(results_dir: Path, out: Path, split: str = "test") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_results(results_dir)
    names = [r["system"] + (" *" if r["system"] in ANALYSIS_SYSTEMS else "") for r in rows]
    n, m = len(rows), len(METRICS)
    bar_h, gap = 0.19, 0.02
    fig, ax = plt.subplots(figsize=(10, 0.95 * n + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for j, (level, metric, label, colour) in enumerate(METRICS):
        ys = [i + (j - (m - 1) / 2) * (bar_h + gap) for i in range(n)]
        vals = [r["splits"][split][level][metric] for r in rows]
        ax.barh(ys, vals, height=bar_h, color=colour, label=label, edgecolor=SURFACE, linewidth=1)
        for y, v in zip(ys, vals):
            ax.text(v + 0.006, y, f"{v:.3f}", va="center", ha="left", fontsize=7.5, color=INK2)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(f"score ({split.upper()} split)", color=INK2, fontsize=9)
    ax.tick_params(axis="x", colors=INK2, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    c = rows[0]["counts"][split] if rows else {}
    ax.set_title(f"Retrieval ablation, {split.upper()} split\n"
                 f"page: documents with >= 2 pages (n={c.get('multi')})  |  component: n={c.get('comp')}",
                 fontsize=10.5, color=INK, loc="left", pad=28)
    ax.legend(ncol=m, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9, handlelength=1.2)
    if any(r["system"] in ANALYSIS_SYSTEMS for r in rows):
        fig.text(0.01, 0.005, "* analysis row: true query types instead of the classifier", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "all", "dev"])
    args = ap.parse_args(argv)
    cfg = get_config()
    results_dir: Path = cfg["paths"]["results_dir"]
    name = "ablation.png" if args.split == "test" else f"ablation_{args.split}.png"
    out = plot(results_dir, results_dir / name, args.split)
    print(f"wrote {out.relative_to(cfg['repo_root'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
