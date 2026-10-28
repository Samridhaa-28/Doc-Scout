"""Build the P8 final package: DocScout_Project_Report.pdf, Abstract.pdf and CONTRIBUTIONS.md in docs/final/.

No evaluation is run and no LLM is called: every number below is copied from a repo file, cited next to it
(P1-P7 reports, eval/results/summary.md, eval/results/*.json, configs/config.yaml, split.json, README).
Fill in TEAM / SUBTITLE and rerun:  .venv/Scripts/python docs/final/build_final_docs.py
(Abstract.pdf is the team's edited version and is kept; add --abstract to regenerate it from this script.)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)
from reportlab.platypus.doctemplate import BaseDocTemplate, Frame, PageTemplate
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "final"

# ------------------------------------------------------------------ fill these in
TITLE = "DocScout"
SUBTITLE = "Query-Aware Fine-Grained Evidence Retrieval from Long Documents"
TEAM = [("Aarthi NJ", "22PD01"), ("Anuvarsana K", "22PD06"), ("Samridhaa R", "22PD29")]
ABSTRACT_TITLE = "DocScout-X"   # the abstract follows the team's IR package format (title as in their draft)

# ------------------------------------------------------------------ styles
ss = getSampleStyleSheet()
INK, ACC = colors.HexColor("#1f1d19"), colors.HexColor("#b4432a")
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=12.6, spaceAfter=4)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.6, leading=9.4, spaceAfter=0)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.8, leading=9.6, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=INK,
                    spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=ACC,
                    spaceBefore=8, spaceAfter=4)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=12, bulletIndent=2)
CODE = ParagraphStyle("code", parent=BODY, fontName="Courier", fontSize=7.4, leading=9, backColor=colors.HexColor("#f3efe6"),
                      borderPadding=4, spaceBefore=2, spaceAfter=6)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8, textColor=colors.HexColor("#555555"), alignment=TA_CENTER)
SRC = ParagraphStyle("src", parent=SMALL, textColor=colors.HexColor("#6f685c"), spaceAfter=5)

W = A4[0] - 3.2 * cm   # usable width


def P(t: str, st: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(t, st)


def bullets(items, st=BUL):
    return [Paragraph(i, st, bulletText="•") for i in items]


def src(t: str) -> Paragraph:
    return Paragraph("Source: " + t, SRC)


def table(rows, widths=None, header=True, font=CELL):
    """Wrapped-cell table (Paragraph cells never overflow the page)."""
    data = [[c if not isinstance(c, str) else Paragraph(c, CELLB if (header and i == 0) else font) for c in r]
            for i, r in enumerate(rows)]
    n = len(rows[0])
    widths = widths or [W / n] * n
    scale = W / sum(widths)
    t = Table(data, colWidths=[w * scale for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bdb3a0")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9e1d0")))
    t.setStyle(TableStyle(style))
    return t


def img(path: Path, width: float, caption: str):
    from PIL import Image as PILImage

    w, h = PILImage.open(path).size
    return KeepTogether([Image(str(path), width=width, height=width * h / w), P(caption, CAP), Spacer(1, 6)])


# ------------------------------------------------------------------ flow diagram (vector)
def flow_diagram() -> Drawing:
    d = Drawing(W, 330)
    off = ["PDF / page images", "Docling parse + OCR (RapidOCR)", "Components: paragraph / heading /\ntable / figure / caption",
           "text_form serialization", "Structure graph (networkx)", "Indexes: BM25 + BGE (+ CLIP figures)\nper branch text / table / figure"]
    on = ["Question + document", "Query-type classifier (rules + LLM)", "Retrieve all components (BM25, BGE, CLIP)",
          "Min-max normalize + weighted fusion (soft)", "Cross-encoder rerank (top 30)", "Feedback hook: session Rocchio + memory",
          "Modality slots (type-conditional)", "Table cell locator (numerical)", "Graph context expansion",
          "Grounded LLM answer + citations", "Provenance why-line", "FastAPI  ->  React UI"]

    def col(x, w, labels, title, fill, top=318, bh=19, gap=7.2):
        d.add(String(x, top + 4, title, fontName="Helvetica-Bold", fontSize=8.5, fillColor=INK))
        y = top - bh
        prev = None
        for lab in labels:
            d.add(Rect(x, y, w, bh, fillColor=fill, strokeColor=colors.HexColor("#8a7f6a"), strokeWidth=0.6, rx=3, ry=3))
            lines = lab.split("\n")
            for k, ln in enumerate(lines):
                d.add(String(x + w / 2, y + bh / 2 - 3 + (len(lines) - 1) * 4 - k * 8, ln, fontName="Helvetica",
                             fontSize=6.9 if len(lines) > 1 else 7.4, textAnchor="middle", fillColor=INK))
            if prev is not None:
                d.add(Line(x + w / 2, prev, x + w / 2, y + bh + 1, strokeColor=INK, strokeWidth=0.7))
                d.add(Polygon([x + w / 2 - 2.5, y + bh + 4, x + w / 2 + 2.5, y + bh + 4, x + w / 2, y + bh + 0.5],
                              fillColor=INK, strokeColor=INK))
            prev = y
            y -= bh + gap
        return y + bh + gap

    lw, rw = W * 0.40, W * 0.47
    col(0, lw, off, "OFFLINE (once per document)", colors.HexColor("#e7eef8"), bh=26, gap=13)
    col(W - rw, rw, on, "ONLINE (per question)", colors.HexColor("#f6e7df"))
    # index -> retrieve link
    d.add(Line(lw, 318 - 6 * 39 + 13, W - rw - 2, 318 - 3 * 26.2 + 9, strokeColor=ACC, strokeWidth=0.9, strokeDashArray=[3, 2]))
    d.add(String(lw + 6, 318 - 6 * 39 + 20, "indexes searched", fontName="Helvetica-Oblique", fontSize=6.8, fillColor=ACC))
    return d


# ------------------------------------------------------------------ data from repo files
def load(p: str):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def file_phase():
    """(phase, path, lines) for every tracked file, phase = first commit that added it (git history)."""
    log = subprocess.run(["git", "log", "--reverse", "--diff-filter=A", "--name-only", "--format=@@%s"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8").stdout
    first, cur = {}, None
    for line in log.splitlines():
        if line.startswith("@@"):
            cur = line[2:].split(":")[0].replace("phase-", "P")
        elif line.strip() and line not in first:
            first[line] = cur
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    out = []
    for f in files:
        try:
            n = sum(1 for _ in open(ROOT / f, encoding="utf-8", errors="ignore"))
        except OSError:
            n = 0
        out.append((first.get(f, "P8"), f, n))
    return out


DESC = {
    ".claude/launch.json": "dev-server entry for the UI preview (tooling)",
    ".env.example": "template for LLM_PROVIDER / LLM_API_KEY / LLM_MODEL (real .env is gitignored)",
    ".gitignore": "ignores data/, indexes, eval JSON outputs (except split.json), ui/node_modules, ui/dist",
    "CLAUDE.md": "project brief, working rules and architecture",
    "README.md": "setup on Windows, data, ingest, indexes, evaluation, API + UI, final deliverables",
    "pyproject.toml": "package metadata (src layout, package docscout)",
    "requirements.txt": "Python dependencies", "requirements.lock.txt": "pinned versions (CPU torch index on line 1)",
    "configs/config.yaml": "every setting: paths, stores, models, fusion weights, classifier, localization, answer, feedback, llm, eval, api",
    "docs/PHASES.md": "phase plan P1-P8 with acceptance criteria", "docs/PROGRESS.md": "approved phase summaries",
    "src/docscout/config.py": "loads config.yaml, resolves paths, store switching (use_store), secrets from .env",
    "src/docscout/schema.py": "Component / Evidence / QueryResult dataclasses, JSONL helpers",
    "src/docscout/env_check.py": "library/version check (scripts.check_env)",
    "src/docscout/pipeline.py": "DocScoutPipeline.answer(): the 10 online steps, timings, meta.reranked",
    "src/docscout/dataset/mpdocvqa.py": "download_subset(): MP-DocVQA val subset, page images, questions.jsonl",
    "src/docscout/dataset/explore.py": "dataset statistics",
    "src/docscout/ingestion/parser.py": "Docling parse (layout, TableFormer, RapidOCR), type mapping, tables to parquet, figure crops, captions",
    "src/docscout/ingestion/serializer.py": "text_form per component type (table/figure serialization)",
    "src/docscout/ingestion/runner.py": "isolated ingest worker, crash/timeout recovery, cleanup, resumable run",
    "src/docscout/ingestion/validate.py": "component checks (ids, bboxes, text)",
    "src/docscout/ingestion/visualize.py": "debug overlays of component boxes",
    "src/docscout/graph/builder.py": "Document -> Page -> Section -> Block graph, caption links, helpers",
    "src/docscout/indexing/base.py": "BaseIndex interface, per-document rows, fingerprints",
    "src/docscout/indexing/bm25_index.py": "tokenizer + BM25Okapi",
    "src/docscout/indexing/dense_index.py": "sentence-transformers + FAISS IndexFlatIP (BGE, MiniLM)",
    "src/docscout/indexing/visual_index.py": "CLIP clip-ViT-B-32 index over figure crops",
    "src/docscout/indexing/build_all.py": "build/load all branch indexes, staleness check",
    "src/docscout/indexing/search.py": "interactive search report (scripts.search_cli)",
    "src/docscout/query/classifier.py": "query type textual / numerical / visual: rules first, LLM when unsure",
    "src/docscout/retrieval/retrievers.py": "HybridRetriever: scores every component of the document; Candidate",
    "src/docscout/retrieval/normalize.py": "min-max (default) and z-score per (branch, retriever)",
    "src/docscout/retrieval/fusion.py": "weighted sum (main), RRF k=60, hard routing",
    "src/docscout/retrieval/slots.py": "modality slot reservation (off / conditional / both)",
    "src/docscout/rerank/cross_encoder.py": "ms-marco-MiniLM-L-6-v2 rerank of the top 30",
    "src/docscout/localization/table_locator.py": "row/column/cell selection in the top tables (numerical queries)",
    "src/docscout/context/graph_expander.py": "section heading, caption/linked item, reading-order siblings",
    "src/docscout/reasoning/llm_client.py": "provider-agnostic LLM client (Groq), JSON mode, disk cache, retries, rate limits",
    "src/docscout/reasoning/prompts.py": "numbered-evidence prompt, size cap",
    "src/docscout/reasoning/answerer.py": "grounded answer, citation validation, not-found handling",
    "src/docscout/evidence/provenance.py": "per-stage scores, matched terms, graph path, why-line",
    "src/docscout/evidence/cell_boxes.py": "page boxes of localized cells (PyMuPDF, digital PDFs)",
    "src/docscout/feedback/session_store.py": "SQLite votes table (evidence + answer votes)",
    "src/docscout/feedback/rocchio.py": "session Rocchio FeedbackHook, pin / exclude, CE score cache",
    "src/docscout/feedback/memory.py": "cross-session memory prior (one function)",
    "api/main.py": "FastAPI: /ask, /feedback, /next, /documents, /page-image; /ingest + /ingest/status (P8b)",
    "ui/src/components/UploadButton.tsx": "Upload PDF button: POST /ingest, polls status, refreshes the document list (P8b)",
    "scripts/check_env.py": "entry: library check", "scripts/download_data.py": "entry: download MP-DocVQA subset",
    "scripts/explore_data.py": "entry: dataset stats", "scripts/ingest.py": "entry: parse + serialize + graph (--store, --limit)",
    "scripts/check_components.py": "entry: validate components", "scripts/visualize_components.py": "entry: draw overlays",
    "scripts/build_indexes.py": "entry: build indexes (--store, --rebuild)", "scripts/search_cli.py": "entry: query one index",
    "scripts/ask.py": "entry: one question end to end (--demo, --k, --slots, --store, --json)",
    "eval/select_eval_docs.py": "choose the 49-document evaluation subset", "eval/build_qrels.py": "page + component qrels",
    "eval/metrics.py": "ranx wrapper, page-from-component, ANLS", "eval/run_eval.py": "8 retrieval systems, summary.md, t-tests",
    "eval/split.py": "document-level DEV/TEST split", "eval/tune_fusion.py": "DEV-only fusion weight grid",
    "eval/eval_classifier.py": "classifier accuracy / macro-F1", "eval/plot_results.py": "ablation.png",
    "eval/eval_answers.py": "ANLS settings (a)/(b)/oracle, slot decision", "eval/eval_locator.py": "table-locator cell hit rate",
    "eval/eval_feedback.py": "simulated Rocchio feedback, pre-registered rule, memory test",
    "ui/index.html": "Vite entry page", "ui/package.json": "UI dependencies and scripts", "ui/package-lock.json": "pinned npm tree",
    "ui/tsconfig.json": "TypeScript settings", "ui/vite.config.ts": "dev server + /api proxy",
    "ui/src/main.tsx": "React root", "ui/src/App.tsx": "layout, ask/vote/next flow, keyboard, FLIP re-rank animation",
    "ui/src/api.ts": "typed API client", "ui/src/styles.css": "paper-and-ink theme, modality colours",
    "ui/src/components/AnswerPanel.tsx": "answer, citation chips, confidence strip, answer thumbs",
    "ui/src/components/EvidenceCard.tsx": "evidence card: breadcrumb, cells, score bars, votes",
    "ui/src/components/PageViewer.tsx": "page image with bbox and cell highlights",
    "ui/src/components/ScoreBars.tsx": "bm25 / bge / clip / CE / fb bars",
    "docs/final/build_final_docs.py": "builds this report, the abstract and CONTRIBUTIONS.md",
}
TEST_DESC = {
    "tests/test_schema.py": (5, "Component / Evidence round trips, ids"), "tests/test_config.py": (6, "config paths, stores, keys"),
    "tests/test_download_utils.py": (3, "download helpers"), "tests/test_ingestion.py": (15, "parser mapping, serializer, graph"),
    "tests/test_ingest_isolation.py": (1, "crashing worker is isolated"), "tests/test_indexing.py": (8, "BM25 / dense / CLIP indexes, staleness"),
    "tests/test_eval.py": (11, "qrels, metrics, page-from-component"), "tests/test_classifier.py": (17, "rules, LLM fallback"),
    "tests/test_llm_client.py": (9, "cache, retries, rate limit"), "tests/test_retrieval.py": (13, "normalization, fusion, RRF, rerank tail"),
    "tests/test_split.py": (2, "document-level split"), "tests/test_answerer.py": (12, "prompt cap, citations, not found, bootstrap"),
    "tests/test_graph_expander.py": (5, "context expansion"), "tests/test_pipeline_smoke.py": (6, "end to end with fakes, feedback hook"),
    "tests/test_slots.py": (6, "slot reservation variants"), "tests/test_table_locator.py": (14, "cell / row / column selection"),
    "tests/test_rocchio.py": (9, "Rocchio math, pin / exclude, lambda blend, identity"),
    "tests/test_memory.py": (9, "votes store, memory gates"), "tests/test_api.py": (9, "all endpoints incl. PDF upload (TestClient)"),
    "tests/test_cell_boxes.py": (2, "cell boxes on a synthetic PDF"),
}

M1 = ["src/docscout/config.py", "src/docscout/schema.py", "src/docscout/env_check.py", "src/docscout/dataset/mpdocvqa.py",
      "src/docscout/dataset/explore.py", "src/docscout/ingestion/parser.py", "src/docscout/ingestion/runner.py",
      "src/docscout/ingestion/serializer.py", "src/docscout/ingestion/validate.py", "src/docscout/ingestion/visualize.py",
      "src/docscout/graph/builder.py", "src/docscout/indexing/base.py", "src/docscout/indexing/bm25_index.py",
      "src/docscout/indexing/dense_index.py", "src/docscout/indexing/visual_index.py", "src/docscout/indexing/build_all.py",
      "src/docscout/indexing/search.py", "scripts/check_env.py", "scripts/download_data.py", "scripts/explore_data.py",
      "scripts/ingest.py", "scripts/check_components.py", "scripts/visualize_components.py", "scripts/build_indexes.py",
      "scripts/search_cli.py", "eval/build_qrels.py", "eval/select_eval_docs.py", "tests/test_schema.py",
      "tests/test_config.py", "tests/test_download_utils.py", "tests/test_ingestion.py", "tests/test_ingest_isolation.py",
      "tests/test_indexing.py"]
M2 = ["src/docscout/retrieval/retrievers.py", "src/docscout/retrieval/normalize.py", "src/docscout/retrieval/fusion.py",
      "src/docscout/retrieval/slots.py", "src/docscout/rerank/cross_encoder.py", "src/docscout/query/classifier.py",
      "src/docscout/localization/table_locator.py", "src/docscout/context/graph_expander.py", "eval/metrics.py",
      "eval/run_eval.py", "eval/split.py", "eval/tune_fusion.py", "eval/eval_classifier.py", "eval/plot_results.py",
      "eval/eval_locator.py", "eval/eval_answers.py", "eval/eval_feedback.py", "tests/test_retrieval.py",
      "tests/test_classifier.py", "tests/test_slots.py", "tests/test_table_locator.py", "tests/test_graph_expander.py",
      "tests/test_eval.py", "tests/test_split.py"]
M3 = ["src/docscout/reasoning/llm_client.py", "src/docscout/reasoning/prompts.py", "src/docscout/reasoning/answerer.py",
      "src/docscout/pipeline.py", "src/docscout/evidence/provenance.py", "src/docscout/evidence/cell_boxes.py",
      "src/docscout/feedback/session_store.py", "src/docscout/feedback/rocchio.py", "src/docscout/feedback/memory.py",
      "api/main.py", "ui/src/App.tsx", "ui/src/api.ts", "ui/src/styles.css", "ui/src/components/AnswerPanel.tsx",
      "ui/src/components/EvidenceCard.tsx", "ui/src/components/PageViewer.tsx", "ui/src/components/ScoreBars.tsx",
      "ui/src/components/UploadButton.tsx", "ui/src/main.tsx", "ui/index.html", "ui/package.json", "ui/tsconfig.json", "ui/vite.config.ts", "scripts/ask.py",
      "tests/test_llm_client.py", "tests/test_answerer.py", "tests/test_pipeline_smoke.py", "tests/test_rocchio.py",
      "tests/test_memory.py", "tests/test_api.py", "tests/test_cell_boxes.py"]
ROLES = [
    ("Ingestion, parsing, indexing, data and qrels", "P1 setup + schema, P2 parsing/serialization/graph, P3 indexes, P4 data subset, qrels, ingest isolation", M1),
    ("Retrieval, fusion, rerank, slots, table locator, graph context, evaluation", "P4 metrics + baselines, P5 classifier/fusion/rerank/tuning/ablation, P6 locator, graph context, slots, ANLS + locator eval, P7 feedback eval", M2),
    ("LLM answering and grounding, feedback, API, React UI, tests", "P5 LLM client, P6 prompts/answerer/provenance/pipeline/ask.py, P7 Rocchio, memory, cell boxes, API, UI", M3),
]


def contributions(lines):
    rows = []
    for i, (role, phases, files) in enumerate(ROLES):
        n = sum(lines.get(f, 0) for f in files)
        rows.append((i, role, phases, files, n))
    return rows


def write_contributions_md(rows) -> None:
    out = [f"# {TITLE}: contribution split", "",
           "Three equal parts. Ownership follows the real files in the repository (line counts from the files at commit time).",
           "Shared by all three: `README.md`, `configs/config.yaml`, `docs/PHASES.md`, `docs/PROGRESS.md`, the phase reports and this final package.", ""]
    for i, role, phases, files, n in rows:
        name, roll = TEAM[i]
        out += [f"## Member {i + 1}: {name} ({roll}): {role}", "", f"Phases: {phases}", "",
                f"Files owned ({len(files)} files, {n:,} lines):", ""]
        out += [f"- `{f}`" for f in files] + [""]
    (OUT / "CONTRIBUTIONS.md").write_text("\n".join(out), encoding="utf-8")


# ------------------------------------------------------------------ report
class ReportDoc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(str(path), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm,
                         bottomMargin=1.5 * cm, title=f"{TITLE}: {SUBTITLE}", author=", ".join(n for n, _ in TEAM), **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=self._footer)])

    @staticmethod
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#6f685c"))
        canvas.drawString(1.6 * cm, 0.9 * cm, f"{TITLE}: {SUBTITLE}")
        canvas.drawRightString(A4[0] - 1.6 * cm, 0.9 * cm, f"page {doc.page}")
        canvas.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            lvl = 0 if f.style.name == "h1" else 1
            key = f"h{id(f)}"
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (lvl, f.getPlainText(), self.page, key))


def build_report(fp, contrib) -> Path:
    S = []
    add = S.extend
    cfg_note = "configs/config.yaml"
    card = load("docs/reports/p7_evidence/api_session_top.json")["top"]
    clf = load("eval/results/classifier.json")
    split = load("eval/results/split.json")
    rr = load("eval/results/hybrid_weighted_soft_rerank.json")["splits"]["test"]

    # title page
    add([Spacer(1, 3.2 * cm), P(f"<font size=30><b>{TITLE}</b></font>", ParagraphStyle("t", parent=BODY, alignment=TA_CENTER, leading=36)),
         Spacer(1, 8), P(f"<font size=14 color='#b4432a'>{SUBTITLE}</font>", ParagraphStyle("st", parent=BODY, alignment=TA_CENTER, leading=18)),
         Spacer(1, 18), P("Project report: Information Retrieval course project", ParagraphStyle("s2", parent=BODY, alignment=TA_CENTER)),
         Spacer(1, 30)])
    add([table([["Team member", "Roll number", "Role"]] + [[n, r, ROLES[i][0]] for i, (n, r) in enumerate(TEAM)], [4, 3, 9]),
         Spacer(1, 26),
         P("Every number in this report is copied from a file in the repository; the file is cited next to it. Nothing was "
           "re-run for this report and no LLM was called. Values that were never measured are marked <b>not recorded</b>.",
           ParagraphStyle("n", parent=BODY, alignment=TA_CENTER, textColor=colors.HexColor("#555555"))),
         PageBreak()])
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("t0", fontName="Helvetica-Bold", fontSize=10, leading=14, leftIndent=0),
                       ParagraphStyle("t1", fontName="Helvetica", fontSize=8.8, leading=11.5, leftIndent=14)]
    add([P("Contents", ParagraphStyle("toch", parent=H1)), toc, PageBreak()])

    # 1
    add([P("1. Problem, objective and what DocScout does", H1)])
    add([P("<b>Problem.</b> Questions about long documents (reports, filings, forms) are usually answered at the level of a whole "
           "PDF or page. A reader still has to find the one paragraph, table cell or chart that holds the answer, and a "
           "generative answer without a pointer to that evidence cannot be checked."),
         P("<b>Objective.</b> Build a retrieval system that, for a question and a document, returns the exact evidence "
           "component (paragraph, table or table cells, figure) with provenance and a one-line \"why retrieved\" explanation, "
           "and a short answer that is grounded only in that evidence and cites it."),
         P("<b>What DocScout does, in plain words.</b> Each document is parsed once into components with page and box "
           "coordinates and indexed three ways (keyword BM25, dense BGE embeddings, CLIP image embeddings for figures). For a "
           "question it decides whether the question is textual, numerical or visual, searches every component of the "
           "document with all retrievers, fuses and reranks the results, finds the matching cells in tables, adds the "
           "surrounding context from the document's structure graph, and asks an LLM to answer only from the numbered "
           "evidence, citing it, or to say \"Not found in the retrieved evidence\". Every evidence item carries its page, "
           "bounding box, per-stage scores and graph path. A user can mark evidence relevant or not relevant; the list "
           "re-ranks with Rocchio feedback. A FastAPI service and a React UI show the answer next to the real page with "
           "the evidence highlighted."),
         src("CLAUDE.md (project context); docs/reports/P6_REPORT.md, P7_REPORT.md")])

    # 2
    add([P("2. End-to-end flow", H1), flow_diagram(), P("Figure 1. Offline indexing (left) and the online question path (right).", CAP)])
    add(bullets([
        "<b>Ingest (Docling).</b> PDF pages (MP-DocVQA page PNGs are joined into one image-only PDF) are parsed by Docling with "
        "layout analysis, TableFormer and RapidOCR (onnxruntime backend). Items map to paragraph / heading / table / figure / caption; "
        "tables are saved to parquet and figures cropped to PNG (P2).",
        "<b>Components.</b> Each has doc_id, 1-based page, bbox in page pixels (top-left origin), page size, reading order, section "
        "and a <i>text_form</i>: raw text for paragraphs; title + caption + column headers + row headers + first 3 rows for tables; "
        "caption + nearest paragraph + inner text for figures (P2).",
        "<b>Structure graph.</b> Document -> Page -> Section (from headings) -> Block with parent-child, sibling and caption-of edges (P2).",
        "<b>Indexes.</b> Per branch (text = paragraph/heading/caption, table, figure): BM25 and BGE dense (FAISS); the figure "
        "branch adds CLIP image embeddings. MiniLM was built for comparison (P3).",
        "<b>Query-type classifier.</b> Word-boundary rules decide when exactly one group matches; otherwise the LLM decides (P5).",
        "<b>Hybrid weighted fusion (soft routing).</b> Every component of the document is scored by every retriever of its branch, "
        "min-max normalized per (branch, retriever) list and combined with retriever weights inside the branch and branch weights "
        "set by the query type. All branches are always searched (P5).",
        "<b>Cross-encoder rerank</b> of the fused top 30; the rest keep their fused order below (P5).",
        "<b>Feedback hook.</b> Identity by default; with votes it applies session Rocchio and (if on) the memory prior (P7).",
        "<b>Modality slots.</b> Type-conditional: a numerical question reserves a top-k slot for the best fused table, a visual one "
        "for the best fused figure (P6).",
        "<b>Table cell locator.</b> For numerical questions, rows and columns of the first tables in the evidence are matched to "
        "the question and the matching cells are returned (P6).",
        "<b>Graph context.</b> Section heading, caption or linked table/figure, and nearby reading-order siblings (P6).",
        "<b>Grounded answer.</b> The LLM sees only the numbered evidence, answers in JSON and must cite evidence ids; invalid ids are "
        "dropped and an answer without a valid citation becomes \"not found\" (P6).",
        "<b>Provenance.</b> Page, type, bbox, scores per stage, matched terms, graph path and a one-line why-line (P6).",
        "<b>Feedback.</b> Session Rocchio (re-score dense with q', re-fuse, rerank; up-votes pinned, down-votes removed) and an "
        "optional cross-session memory prior (P7).",
        "<b>FastAPI -> React UI.</b> /ask, /feedback, /next, /documents, /page-image; split-view UI (P7)."]))
    add([src("docs/reports/P2_REPORT.md - P7_REPORT.md; src/docscout/pipeline.py docstring"), PageBreak()])

    # 3 repo map
    add([P("3. Repository map", H1),
         P("Every tracked file except generated evidence (docs/reports/p6_evidence, p7_evidence: saved command outputs and images) "
           "and eval/results (section 6). \"Phase\" is the first commit that added the file at its current path (git history; "
           "files moved in the post-P4 tidy-up show P4). Data folders are gitignored: data/raw (MP-DocVQA images and questions, "
           "demo PDF), data/processed (components, tables, figures, graphs, manifests, LLM cache, feedback.sqlite), "
           "data/indexes and data/indexes_demo.")])
    folders = [["Folder", "Contents"],
               ["src/docscout/", "the Python package (config, schema, pipeline and one subpackage per stage)"],
               ["src/docscout/dataset, ingestion, graph, indexing", "offline: data download, parsing, serialization, graph, indexes"],
               ["src/docscout/query, retrieval, rerank, localization, context", "online retrieval stages"],
               ["src/docscout/reasoning, evidence, feedback", "LLM answering, provenance, relevance feedback"],
               ["api/", "FastAPI service"], ["ui/", "React + Vite + TypeScript front end"],
               ["scripts/", "thin command-line entry points (argparse + one call)"], ["eval/", "evaluation code; eval/results/ outputs"],
               ["tests/", "pytest suite (162 tests)"], ["configs/", "config.yaml"],
               ["docs/", "PHASES, PROGRESS, phase reports + evidence, final package"]]
    add([table(folders, [5, 11]), Spacer(1, 8)])
    rows = [["Path", "What it contains", "Phase", "Lines"]]
    for ph, f, n in fp:
        if f.startswith(("docs/reports/p6_evidence/", "docs/reports/p7_evidence/", "eval/results/")) or f.endswith(".gitkeep"):
            continue
        d = DESC.get(f) or (TEST_DESC[f][1] + f" ({TEST_DESC[f][0]} tests)" if f in TEST_DESC else None)
        if d is None:
            d = "package marker" if f.endswith("__init__.py") else (
                "phase report" if "/reports/" in f else "final package" if f.startswith("docs/final/") else "-")
        rows.append([f.replace("/", "/<wbr/>") if False else f, d, ph, str(n)])
    add([table(rows, [6.3, 8.2, 1.1, 1.1]), src("git log --diff-filter=A; git ls-files"), PageBreak()])

    # 4 stages
    add([P("4. Each stage in detail", H1)])
    st = [["Stage", "Input -> output", "Key config values", "Why this choice"],
          ["Parse (Docling)", "PDF -> components with bbox, tables (parquet), figure crops",
           "parsing.ocr: true; ocr_backend: onnxruntime; pdf_backend mpdocvqa: pypdfium2, pdf: docling_parse; image_dpi 144; mpdocvqa_dpi 200",
           "Layout + TableFormer give typed components with boxes. onnxruntime OCR: 27.4 -> 20.0 s/page (-27%), identical output (P2). "
           "pypdfium2 because docling_parse crashed natively on image-only PDFs (P4)."],
          ["Serialize", "component -> text_form", "max_text_form_tokens 512; caption_max_gap 0.15",
           "Tables/figures need searchable text: headers, row labels and caption carry most query terms (P2)."],
          ["Graph", "components -> networkx DiGraph", "context.max_siblings 2; sibling_window 3",
           "Gives section paths for provenance and context for the LLM (P2, P6)."],
          ["Index", "text_form / crops -> BM25, FAISS, CLIP", "models.bge BAAI/bge-base-en-v1.5; minilm all-MiniLM-L6-v2; clip clip-ViT-B-32; bge_query_instruction on queries only",
           "BM25 for exact terms and numbers, BGE for paraphrase, CLIP for charts (P3)."],
          ["Classify", "question -> textual / numerical / visual", "classifier.use_llm true; rule_confidence 0.9; unsure_fallback textual",
           f"Rules alone 0.614 accuracy; rules + LLM 0.751 (P5, eval/results/classifier.json)."],
          ["Retrieve + fuse", "question -> every component scored",
           "fusion.method weighted_sum; normalization minmax; retriever_weights bm25 0.7 / dense 0.3 / clip 0.3; branch_weights textual {0.4, 0.375, 0.225}, numerical {0.3, 0.6, 0.1}, visual {0.25, 0.15, 0.6}",
           "Soft routing keeps all branches; weights tuned on DEV only (objective 0.4753 -> 0.5133, P5). Weighted beat RRF on component R@1 (0.432 vs 0.398 TEST)."],
          ["Rerank", "fused list -> top-30 reordered", "models.cross_encoder ms-marco-MiniLM-L-6-v2; retrieval.rerank_top_n 30",
           "Largest single gain: component MRR 0.599 -> 0.653 TEST (P5)."],
          ["Slots", "ranking -> top-k evidence", "answer.slots conditional",
           "Adopted by a pre-registered non-inferiority rule on DEV; no measurable benefit (P6)."],
          ["Locate cells", "top tables -> matching cells", "localization.fuzzy_threshold 85; min_fuzzy_len 4; max_cells 12; top_k_tables 3",
           "Numerical answers live in one cell; showing the cell is the finest evidence (P6)."],
          ["Answer", "numbered evidence -> JSON answer + citations",
           "llm.model openai/gpt-oss-20b; temperature 0; max_tokens 512; answer.max_prompt_tokens 2500; not_found text",
           "Grounding + citation validation; prompt cap keeps under Groq 8,000 TPM (P5, P6)."],
          ["Provenance", "candidate -> scores, path, why-line", "-", "Explains every item (P6)."],
          ["Feedback", "votes -> re-ranked list",
           "feedback.rocchio alpha 1.0, beta 0.75, gamma 0.15, lambda 0.3; memory off; step 0.05; cap 0.15; min_votes 2; min_similarity 0.75; half_life_days 30",
           "Rocchio is the classic relevance-feedback method; lambda term lets it reorder the CE top 30 (P7)."],
          ["API + UI", "HTTP -> JSON / page PNG", "api.host 127.0.0.1; port 8000; cors_origins :5173", "Models loaded once; UI shows page + evidence (P7)."]]
    add([table(st, [2.2, 3.4, 5.6, 5.4]), src(f"{cfg_note}; P2-P7 reports; eval/results/classifier.json, tuning.json, summary.md"), PageBreak()])

    # 5 datasets
    add([P("5. Datasets", H1), P("5.1 MP-DocVQA subset", H2),
         *bullets([
             "Source: Hugging Face <b>lmms-lab/MP-DocVQA</b>, val split (fallback rubentito/mp-docvqa); the first 150 documents in "
             "stream order were downloaded (configs/config.yaml dataset; P1_REPORT).",
             "Evaluation subset (eval/select_eval_docs.py): documents with more than 20 pages skipped, stopping at about 250 pages: "
             "<b>49 documents, 250 pages, 293 questions</b> (246 on documents with >= 2 pages), 12 single-page documents (PROGRESS.md P4).",
             "Ingested: <b>2,976 components</b>: 2,380 paragraphs, 339 headings, 96 tables, 120 figures, 41 captions (PROGRESS.md P4).",
             f"DEV/TEST split by document, seed 42, 20% of documents to DEV: DEV {len(split['dev']['doc_ids'])} documents / "
             f"{len(split['dev']['question_ids'])} questions; TEST {len(split['test']['doc_ids'])} documents / "
             f"{len(split['test']['question_ids'])} questions (176 on documents with >= 2 pages, 209 with a relevant component) "
             "(eval/results/split.json, PROGRESS.md P5). DEV is used for every tuning decision; TEST is looked at once per decision.",
             "Question types for analysis: gold labels from the DocVQA question_types field, joined on questionId (100% join, P5)."]),
         P("5.2 Demo PDF: Census P60-281", H2),
         *bullets(["\"Health Insurance Coverage in the United States: 2022\" (US Census Bureau, public domain): 31 pages, "
                   "1,892,540 bytes, text layer; Table 1, Figures 1-8, Tables A-1 to A-3 (P6_REPORT 4.1).",
                   "Ingested into a separate <b>demo store</b> (stores.demo) so BM25 statistics of the evaluation corpus stay "
                   "unchanged: 344 components (258 paragraphs, 58 headings, 9 tables, 13 figures, 6 captions), 281.8 s (P6_REPORT 4.2).",
                   "Used for demos only; it has no gold answers and is not part of any metric."]),
         P("5.3 How the qrels were built", H2),
         *bullets(["<b>Page relevance:</b> the question's answer_page.",
                   "<b>Component relevance:</b> a component on the answer page whose text, text_form or a table cell contains a gold "
                   "answer, after normalizing to [a-z0-9]. partial_ratio >= 90 only when the text is at least as long as the answer, "
                   "else ratio; answers of <= 3 characters need an exact word or cell.",
                   "<b>Coverage:</b> 278 of 293 questions (94.9%) have >= 1 relevant component; an audit removed a matching bug that had "
                   "inflated it to 96.2%. Of 389 audited pairs, 298 were exact substrings, 77 exact short words/cells, 14 fuzzy OCR variants.",
                   "Component metrics use those 278 questions (209 on TEST); page metrics use documents with >= 2 pages as the main number."]),
         src("PROGRESS.md P1/P4/P5; P4_REPORT.md; eval/results/split.json, qrels_info.json"), PageBreak()])

    # 6 outputs
    add([P("6. Outputs explained", H1), P("6.1 An evidence card from a real /ask -> /feedback session", H2),
         P("Recorded in P7 (docs/reports/p7_evidence/api_session_top.json): the Medicaid question on p60-281 after the user "
           "down-voted the wrong table and up-voted Table 1. Fields, trimmed:")])
    cells = card["localized_cells"]
    snippet = {k: card[k] for k in ("id", "eid", "rank", "type", "page", "branch", "bbox", "page_size", "graph_path", "feedback_tag")}
    snippet["scores"] = {k: round(v, 3) for k, v in card["scores"].items()}
    snippet["localized_cells"] = cells[5:6] + ["... 9 more"]
    snippet["bbox"] = [round(v) for v in snippet["bbox"]]
    snippet["cell_boxes"] = [{**b, "bbox": [round(v) for v in b["bbox"]]} for b in card["cell_boxes"] if b["value"] == "62,050"]
    import textwrap

    body = ",\n".join(textwrap.fill(f'  "{k}": {json.dumps(v, ensure_ascii=False)}', 100, subsequent_indent="      ")
                      for k, v in snippet.items())
    txt = "{\n" + body + "\n}"
    add([Preformatted(txt, CODE), P("<b>why:</b> " + card["why"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), SMALL), Spacer(1, 4)])
    add([table([["Field", "What it tells us"],
                ["id / type / page / bbox / page_size", "where the evidence is: component id, type, 1-based page, box in page pixels (top-left origin) and the page size the box refers to"],
                ["branch", "which index branch found it (text / table / figure)"],
                ["graph_path", "Page -> Section -> Block path in the structure graph (shown as a breadcrumb in the UI)"],
                ["scores.bm25_raw / bm25_norm", "BM25 score and its min-max value within the branch list (bar in the UI)"],
                ["scores.dense_bge_raw / _norm", "BGE cosine with the query and its normalized value"],
                ["scores.fused", "weighted-sum score: branch weight x (retriever-weighted normalized scores)"],
                ["scores.rerank (CE)", "cross-encoder logit for (question, text_form); decides the order of the top 30 when feedback is off"],
                ["scores.feedback (fb)", "change made by feedback on the normalized scale (Rocchio / memory), pin offset excluded"],
                ["scores.final", "the score the list is sorted by (raw CE when feedback is off; pinned items +10)"],
                ["feedback_tag", "rocchio / memory / pinned"],
                ["localized_cells / localize_flag", "(row label, column, value) cells chosen by the locator; flag cell / row / column / fallback / no_match"],
                ["cell_boxes", "page boxes of those cells (digital PDFs only); repeated values without a unique match get no box"],
                ["why", "the same information as one readable line: type and branch weight, raw->normalized scores, fused, CE, fb, tag, matched terms, path, bbox"]],
               [5, 11])])
    add([P("6.2 The /ask response and the confidence strip", H2),
         P("POST /ask returns query_id, session_id, query_type with the classification source and confidence, branch_weights, "
           "the answer object, the evidence cards, ranking_size (all components of the document), votes, memory state and "
           "feedback_active. answer.confidence drives the UI's confidence strip: <b>cited_sources</b> (number of valid citations), "
           "<b>not_found</b>, <b>slot_used</b> (a modality slot changed the evidence) and invalid_citations dropped. In the recorded "
           "session the first answer was \"Not found\" (0 cited sources, slot used) and after the two votes and Re-answer it was "
           "\"62,050\" with 1 cited source (P7_REPORT 4.6, 4.8)."),
         P("6.3 eval/results/summary.md", H2),
         P("One generated page with all headline tables: TEST page level on documents with >= 2 pages (main), TEST component level, "
           "TEST page level on all questions, paired t-tests vs dense BGE, the all-293 secondary tables, the ANLS table (P6) and the "
           "two relevance-feedback tables (P7). R@1 and MRR are the headline metrics; R@5/R@10 are near ceiling because documents are "
           "short (random pages already reach R@5 = 0.80 on TEST)."),
         P("6.4 Each file in eval/results/", H2)])
    files = [["File", "What it holds / what the numbers tell us"],
             ["split.json", "DEV/TEST document and question ids (the only committed result file)"],
             ["qrels_page.json, qrels_component.json, qrels_info.json", "relevance judgements in ranx format; per-question match details (which text matched which answer)"],
             ["qtypes_gold.json, docvqa_qtypes_cache.json", "gold question types from DocVQA question_types"],
             ["baseline.json", "random-page baseline per split and subset (expected R@k = mean(1/num_pages))"],
             ["bm25_all.json, dense_minilm_all.json, dense_bge_all.json, hybrid_*.json", "one retrieval system each: metrics per split (test/dev/all) x level (page_all, page_multi, component), latency median/p90, empty runs, full rankings"],
             ["significance.json", "ranx paired t-tests vs dense_bge_all"],
             ["classifier.json", "rules-only / LLM-only / combined accuracy and macro-F1, label counts, latency, predictions"],
             ["tuning.json", "DEV fusion grid (81 settings), objective before/after, chosen weights"],
             ["ablation.png", "bar chart of page/component R@1 and MRR per system"],
             ["answers.json, answers_*.jsonl", "ANLS per setting with bootstrap CIs, per-type scores, not-found rates; per-question answers"],
             ["answers_slots_dev.json / _test.json, answers_*_slots-*.jsonl", "slot-reservation comparison (DEV decision, TEST once)"],
             ["locator.json, locator_dev*.json", "table-locator cell hit rates in oracle and pipeline settings, miss categories"],
             ["feedback_dev.json, feedback_test.json (+ _brief)", "simulated Rocchio: residual metrics per setting, paired bootstrap CIs, rule outcome"],
             ["feedback_memory_dev.json", "memory test per ordered question pair (related / unrelated, fired, before/after metrics)"]]
    add([table(files, [6, 10]), src("eval/results/ (JSON outputs are gitignored and regenerated by the eval scripts; README 'Evaluation'); summary.md"), PageBreak()])

    # 7 models
    add([P("7. Models", H1)])
    mt = [["Role", "Model / method", "Compared against", "Result (TEST unless noted)", "Winner"],
          ["Lexical retriever", "BM25Okapi (rank-bm25), own tokenizer", "BGE, MiniLM",
           "component R@1 0.417 / MRR 0.552; page (>=2 p.) R@1 0.636 / MRR 0.718", "best single retriever at component level"],
          ["Dense embedder", "BAAI/bge-base-en-v1.5 (query instruction)", "all-MiniLM-L6-v2",
           "page MRR 0.756 vs 0.725; comp MRR 0.544 vs 0.465; comp R@1 0.362 vs 0.259; median 35.6 vs 8.9 ms", "BGE (+0.031 page MRR, +0.079 comp MRR)"],
          ["Image retriever", "CLIP clip-ViT-B-32 (figure crops)", "not compared with another image model",
           "qualitative only: 'bar chart' 3/3 correct on the demo PDF; scores nearly flat (0.22-0.25) (P6 4.4)", "-"],
          ["Fusion", "weighted sum, min-max, soft routing", "RRF (k=60) soft; weighted hard routing",
           "comp R@1 0.432 vs RRF 0.398 vs hard 0.360; page MRR 0.788 vs 0.786 vs 0.696", "weighted soft"],
          ["Normalization", "min-max", "z-score implemented but <b>not compared</b>", "-", "-"],
          ["Reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2 (top 30)", "no rerank (no other reranker compared)",
           "page MRR 0.788 -> 0.828 (p = 0.0002 vs BGE); comp MRR 0.599 -> 0.653; median latency 54 -> 763 ms", "with rerank"],
          ["Query classifier", "rules + LLM (gpt-oss-20b) when unsure", "rules only, LLM only",
           "accuracy 0.614 / 0.747 / 0.751; macro-F1 0.521 / 0.644 / 0.652 (293 q)", "rules + LLM"],
          ["Routing input", "predicted query type", "gold DocVQA types (analysis)", "comp MRR 0.599 predicted vs 0.607 gold (soft, no rerank)", "gold slightly better (+0.008)"],
          ["Answer LLM (evaluation)", "openai/gpt-oss-20b via Groq, temp 0, JSON, reasoning_effort low",
           "<b>no other LLM evaluated</b>; llama-3.1-8b-instant was requested but returns 404 on this key",
           "ANLS (b) 0.646 [0.585, 0.707]", "-"],
          ["Answer LLM (demo only)", "openai/gpt-oss-120b (scripts.ask --demo)", "not evaluated",
           "one demo run: same answer as 20b on the textual demo question (P6 4.5)", "-"],
          ["OCR backend", "RapidOCR onnxruntime", "RapidOCR torch", "27.4 -> 20.0 s/page, identical text and boxes (P2)", "onnxruntime"],
          ["PDF backend (image-only)", "pypdfium2", "docling_parse", "docling_parse crashed natively after 7 documents (P4)", "pypdfium2"],
          ["Feedback", "Rocchio + lambda blend (beta 0.75, gamma 0.15, lambda 0.3)", "11 other grid points; lambda = 0 literal; no feedback",
           "residual MRR 0.312 vs 0.289 (+0.023 [-0.048, +0.091], not significant)", "chosen by pre-registered rule on DEV"]]
    add([table(mt, [2.3, 3.4, 3.0, 4.9, 2.6]),
         P("<b>Not compared:</b> other dense embedders (e.g. E5, GTE), other cross-encoders or LLM rerankers, other image models "
           "than CLIP ViT-B/32, other OCR engines or parsers than Docling + RapidOCR, BM25 parameters (k1, b), z-score normalization, "
           "and any LLM other than gpt-oss-20b on ANLS. gpt-oss-120b was never used in evaluation."),
         src("eval/results/summary.md; classifier.json; P2, P4, P5 (2.1, 4.4-4.8), P6 (2.9, 4.4, 4.5), P7 reports"), PageBreak()])

    # 8 evaluations
    add([P("8. Evaluations by phase", H1), P("8.1 P4 baselines (DEV + TEST: page level on documents with >= 2 pages, component level on 278 questions)", H2),
         table([["System", "page R@1", "page R@5", "page MRR", "comp R@1", "comp R@10", "ms/query"],
                ["random", "0.246", "0.770", "0.473", "-", "-", "-"], ["bm25_all", "0.593", "0.793", "0.679", "0.380", "0.633", "0.2"],
                ["dense_minilm_all", "0.537", "0.902", "0.696", "0.244", "0.729", "9.4"],
                ["dense_bge_all", "0.606", "0.907", "0.738", "0.331", "0.726", "35.4"]], [3.2, 1.6, 1.6, 1.6, 1.6, 1.6, 1.5]),
         src("PROGRESS.md P4 (page level on documents with >= 2 pages; component level on 278 questions)"),
         P("8.2 P5 retrieval systems (TEST)", H2),
         table([["System", "page R@1", "page MRR", "page nDCG@10", "comp R@1", "comp MRR", "comp nDCG@10", "median / p90 ms"],
                ["random", "0.270", "0.501", "-", "-", "-", "-", "-"],
                ["bm25_all", "0.636", "0.718", "0.751", "0.417", "0.552", "0.556", "0.2 / 0.5"],
                ["dense_minilm_all", "0.568", "0.725", "0.788", "0.259", "0.465", "0.524", "8.9 / 11.8"],
                ["dense_bge_all", "0.631", "0.756", "0.810", "0.362", "0.544", "0.582", "35.6 / 40.4"],
                ["hybrid_weighted_soft", "0.659", "0.788", "0.839", "0.432", "0.599", "0.624", "54.3 / 60.2"],
                ["hybrid_rrf_soft", "0.659", "0.786", "0.837", "0.398", "0.581", "0.612", "54.8 / 60.5"],
                ["hybrid_weighted_hard", "0.602", "0.696", "0.740", "0.360", "0.473", "0.458", "36.1 / 43.3"],
                ["<b>hybrid_weighted_soft_rerank</b>", "<b>0.733</b>", "<b>0.828</b>", "0.869", "<b>0.488</b>", "<b>0.653</b>", "0.677", "762.7 / 1850.9"],
                ["hybrid_weighted_soft_goldtype (analysis)", "0.670", "0.795", "0.844", "0.449", "0.607", "0.628", "57.2 / 66.4"]],
               [4.2, 1.4, 1.4, 1.5, 1.4, 1.4, 1.5, 2.2]),
         src("eval/results/summary.md (TEST page level, documents with >= 2 pages; TEST component level)"),
         P("Paired t-tests vs dense_bge_all: hybrid soft page MRR p = 0.0765 (not significant), component MRR p = 0.0032; soft + rerank "
           "page MRR p = 0.0002, component MRR p < 0.0001. Fusion weights were tuned on DEV only: objective (mean of DEV page MRR and "
           "component MRR) 0.4753 -> 0.5133 over 81 grid settings; the best point lies on the grid edge for BM25 = 0.7 and was not "
           "extended. Known limitation: with min-max a lone table gets 1.0 from both retrievers, so on DEV the hybrid page MRR (0.586) "
           "was below BGE (0.693); rerank mostly repairs it. Hard routing returns an empty ranking for 34/217 TEST questions."),
         src("eval/results/summary.md; P5_REPORT 2.13, 4.4, 4.8; PROGRESS.md P5")])
    add([P("8.3 P6 answer quality: ANLS (TEST, gpt-oss-20b)", H2),
         table([["Setting", "n", "ANLS [95% CI]", "numerical", "textual", "visual", "not found"],
                ["(a) top-1 evidence only (baseline)", "217", "0.486 [0.424, 0.546]", "0.515", "0.466", "0.344", "38.2%"],
                ["(b) top-3 + graph context + localized cells (final)", "217", "<b>0.646</b> [0.585, 0.707]", "0.674", "0.632", "0.457", "15.7%"],
                ["(c) oracle: gold answer page as evidence", "40", "0.791 [0.666, 0.910]", "0.778", "0.798", "1.000", "15.0%"]],
               [5.4, 0.8, 2.6, 1.5, 1.5, 1.4, 1.5]),
         P("Paired (b) - (a) = +0.161 [+0.103, +0.218] on the same 217 questions; (b) better on 47, (a) better on 11. On the 40 oracle "
           "questions: (a) 0.545, (b) 0.740, oracle 0.791. 0 invalid citations in all settings. CIs: percentile bootstrap over "
           "questions, 2,000 resamples, seed 42."),
         src("eval/results/summary.md (ANLS section); answers.json"),
         P("8.4 P6 table cell locator (TEST, 62 numerical questions with a relevant table)", H2),
         table([["Version", "oracle strict", "oracle with row label", "pipeline strict", "pipeline with row label", "DEV oracle strict"],
                ["v1 base rules (clean estimate)", "0.613 (38/62)", "not measured", "<b>0.548 (34/62)</b>", "not measured", "0.480 (12/25)"],
                ["v3 shipped (two TEST-informed rules)", "0.613 (38/62)", "0.710 (44/62)", "0.516 (32/62)", "0.532", "0.480 (12/25)"]],
               [4.4, 2.2, 2.4, 2.2, 2.4, 2.2]),
         P("The whole table contains the answer in 95.2% of oracle cases; 13 of 24 oracle misses are answers that are row labels. "
           "In the pipeline 16 misses are questions not predicted numerical and 7 are wrong tables."),
         src("P6_REPORT 4.6; eval/results/locator.json"),
         P("8.5 P6 modality slots (pre-registered non-inferiority rule)", H2),
         P("DEV (76 q): conditional - off = -0.005 [-0.015, +0.000] ANLS, locator unchanged (4/25) -> rule satisfied, type-conditional "
           "adopted with no measurable benefit. TEST once: 0.637 vs 0.646 (-0.009 [-0.023, +0.000]), 0 better / 2 worse. The demo "
           "check in P7 (k = 5) showed the reserved table/figure was the wrong one for both demo questions; no answer changed."),
         src("P6_REPORT 4.10; P7_REPORT 4.1")])
    add([P("8.6 P7 relevance feedback (simulated, residual collection, component level)", H2),
         P("Pre-registered before any DEV run (docs/reports/P7_PREREGISTRATION.md): grid beta {0.5, 0.75} x gamma {0, 0.15, 0.25} x "
           "lambda {0.3, 0.5}; keep the default (0.75, 0.15, 0.3) unless another point gains >= 0.01 DEV residual MRR without losing "
           "R@1. Round 1 = final pipeline; the simulated user votes on the top 5 from the qrels; judged items are removed from both "
           "rankings (residual collection). The rule kept the default."),
         table([["Split / system", "n", "R@1 (diff [95% CI])", "MRR (diff [95% CI])", "nDCG@5 (diff [95% CI])"],
                ["DEV no feedback", "34", "0.059", "0.202", "0.226"],
                ["DEV lambda = 0 (literal)", "34", "0.074 (+0.015 [+0.000, +0.044])", "0.229 (+0.027 [-0.002, +0.083])", "0.244 (+0.018 [+0.000, +0.054])"],
                ["DEV chosen b0.75 g0.15 l0.3", "34", "0.191 (+0.132 [+0.029, +0.250])", "0.293 (+0.091 [+0.011, +0.180])", "0.270 (+0.044 [-0.027, +0.115])"],
                ["TEST no feedback", "68", "0.120", "0.289", "0.271"],
                ["TEST lambda = 0 (literal)", "68", "0.110 (-0.010 [-0.044, +0.015])", "0.294 (+0.004 [-0.020, +0.027])", "0.271 (+0.000 [-0.018, +0.018])"],
                ["<b>TEST chosen (run once)</b>", "68", "0.140 (+0.020 [-0.069, +0.101])", "<b>0.312 (+0.023 [-0.048, +0.091])</b>", "0.295 (+0.024 [-0.040, +0.089])"]],
               [4.0, 0.8, 3.7, 3.7, 3.7]),
         P("<b>The TEST gain is not significant</b> (every CI includes 0); only 16 of 68 TEST residual questions had an up-vote in the "
           "top 5. Round 1 reproduces P5 (TEST component R@1 0.488, MRR 0.653), so feedback-off numbers are unchanged."),
         src("eval/results/summary.md (feedback sections); P7_REPORT 4.2-4.4"),
         P("8.7 P7 cross-session memory test (DEV, 598 ordered question pairs on 10 documents)", H2),
         table([["Group", "pairs", "fired", "MRR diff [95% CI]", "R@1 diff", "better / worse"],
                ["related (share a gold component)", "172", "22", "+0.0070 [+0.0003, +0.0161]", "+0.0116", "5 / 0"],
                ["related, where the prior fired", "22", "22", "+0.0551 [+0.0021, +0.1220]", "+0.0909", "5 / 0"],
                ["unrelated", "426", "34", "+0.0003 [-0.0008, +0.0015]", "+0.0000", "4 / 5"],
                ["unrelated, where the prior fired", "34", "34", "+0.0033 [-0.0094, +0.0185]", "+0.0000", "4 / 5"]],
               [5.0, 1.3, 1.3, 4.0, 1.7, 2.0]),
         P("Related questions improve where the gates fire; unrelated questions are flat on average but 5 of 34 pairs moved down "
           "below rank 1 (worst rank 3 -> 4). The 0.75 similarity gate passes only 22 of 172 related pairs (median same-document cosine 0.51)."),
         src("P7_REPORT 4.5; eval/results/feedback_memory_dev.json")])
    add([P("8.8 Final headline results (TEST)", H2),
         table([["Metric", "Value", "Compared with", "Source"],
                ["Component R@1 / MRR (final pipeline)", "<b>0.488 / 0.653</b>", "dense BGE 0.362 / 0.544 (MRR p < 0.0001)", "summary.md"],
                ["Component nDCG@5 (final pipeline)", "<b>0.644</b>", "BGE: not recorded", "P7_REPORT 4.3 (no-fb full ranking)"],
                ["Component nDCG@10 (final pipeline)", "0.677", "BGE 0.582", "summary.md"],
                ["Page R@1 / MRR, documents >= 2 pages", "<b>0.733 / 0.828</b>", "BGE 0.631 / 0.756; random 0.270 / 0.501", "summary.md"],
                ["Page nDCG@5", "not recorded", "page nDCG@10 = 0.869", "summary.md"],
                ["ANLS: baseline (a) / final (b) / oracle", "<b>0.486 / 0.646 / 0.791</b>", "(b)-(a) +0.161 [+0.103, +0.218]", "summary.md"],
                ["ANLS with shipped default slots: conditional", "0.637", "slots off 0.646", "P6_REPORT 4.10"],
                ["Classifier accuracy (rules + LLM)", "0.751", "rules only 0.614", "classifier.json"],
                ["Table locator cell hit rate (pipeline, clean v1)", "0.548 (34/62)", "oracle 0.613", "P6_REPORT 4.6"],
                ["Rocchio residual MRR", "0.312", "no feedback 0.289; +0.023 [-0.048, +0.091], <b>not significant</b>", "summary.md"],
                ["Median latency, retrieval + rerank", "762.7 ms (p90 1,850.9)", "no rerank 54.3 ms", "summary.md"]],
               [5.0, 3.0, 5.0, 3.0]), PageBreak()])

    # 9 UI
    ev = ROOT / "docs/reports/p7_evidence"
    add([P("9. UI walkthrough", H1),
         P("Start the API and the UI (section 12), open http://localhost:5173, pick a document in the dropdown (the demo PDF is "
           "first, then the 49 evaluation documents), and ask a question or click a sample question. Left: the answer with citation "
           "chips and the confidence strip, then the evidence cards (rank, type, page, breadcrumb, localized cells, score bars, "
           "why-line on demand). Right: the real page image with every evidence box in its modality colour (text blue, table "
           "vermilion, figure green) and the cited cells highlighted. Vote with the buttons or the keyboard (J/K move, up/down "
           "vote, N next); the list re-ranks without a new LLM call, and \"Re-answer with current evidence\" asks again in the "
           "same session. The memory switch turns on the cross-session prior."),
         img(ev / "ui_1_feedback_rerank.png", W, "Figure 2. After down-voting the wrong table and up-voting Table 1: the table is pinned at #1 (moved up 4), "
             "other items show their rank change and the rocchio tag; the answer is marked as coming from the earlier evidence."),
         img(ev / "ui_2_reanswer.png", W, "Figure 3. Re-answer in the same session: \"62,050\", citing E1 on page 8; the six located cells are highlighted on the page."),
         img(ev / "ui_3_textual.png", W, "Figure 4. A textual question: \"employment-based insurance\", citing the paragraph on page 9."),
         src("docs/reports/p7_evidence/ui_1_feedback_rerank.png, ui_2_reanswer.png, ui_3_textual.png (headless Edge, 1440x1000, real API)"),
         PageBreak()])

    # 10 testing
    trows = [["Test file", "Tests", "What it covers"]] + [[f, str(n), d] for f, (n, d) in TEST_DESC.items()]
    trows.append(["<b>Total</b>", "<b>162</b>", "pytest -> 162 passed (160 in P7_REPORT 5 + 2 upload tests in P8b)"])
    add([P("10. Testing", H1),
         P("Unit and integration tests run on CPU without downloading large models: the LLM, the cross-encoder and the retrievers "
           "are faked where needed, and the API is tested with FastAPI's TestClient on a fake pipeline. The UI was checked with "
           "a TypeScript build (tsc + vite build), real HTTP calls and headless-browser screenshots."),
         table(trows, [5.0, 1.2, 9.8]), src("tests/; pytest --collect-only; P7_REPORT 5"), PageBreak()])

    # 11 limitations
    add([P("11. Limitations and future work", H1),
         *bullets([
             "<b>Rocchio gain not significant on TEST:</b> +0.023 residual MRR [-0.048, +0.091] on 68 questions; DEV (34 questions, 5 with "
             "an up-vote) was too small for a reliable choice. Real user feedback, not simulated qrels votes, is needed.",
             "<b>Feedback-aware slots:</b> slot reservation picks the best <i>fused</i> table/figure and ignores feedback, so a memory "
             "penalty cannot remove a reserved item (left as is by decision; session Rocchio is unaffected).",
             "<b>Follow-up questions:</b> each question is answered on its own; there is no conversational context or query rewriting.",
             "<b>General-knowledge fallback:</b> by design the LLM answers only from evidence and says \"Not found\" otherwise; there is "
             "no optional, clearly labelled fallback to general knowledge.",
             "<b>PDF upload (added in P8b):</b> an Upload PDF button next to the dropdown sends the file to POST /ingest, which runs "
             "the existing ingest and demo-index rebuild in the background (one upload at a time; 188 s for a 1-page PDF on CPU) and "
             "reloads only the demo store; every upload rebuilds all demo indexes.",
             "<b>Cross-encoder vs tables/figures:</b> the CE prefers prose; on the demo it pushes Figure 2 to about rank 25 and Table 1 to "
             "rank 11, so the visual demo question fails at k = 5.",
             "<b>Lone-table normalization:</b> min-max lifts a lone table to the maximum branch score (hurts DEV page MRR before rerank).",
             "<b>Locator:</b> 13 of 24 oracle misses are row-label answers; superlative/count questions need comparison, not lookup; "
             "two rules were TEST-informed (clean estimate v1 quoted).",
             "<b>Memory:</b> small by design (cap 0.15 moved the gold table from rank 12 to 8); 5 of 34 unrelated fired pairs got slightly worse.",
             "<b>Cell boxes</b> only on digital PDFs; <b>CLIP</b> separates chart types weakly; <b>Groq</b> free-tier rate and daily "
             "token limits slow full ANLS runs.",
             "<b>Scale:</b> evaluation covers 49 documents / 293 questions of MP-DocVQA val; the full 150-document ingest was not run."]),
         src("P5-P7 reports, sections 8"), PageBreak()])

    # 12 how to run
    add([P("12. How to run from a fresh clone (Windows)", H1),
         Preformatted("""python -m venv .venv          (Python 3.10)
.venv/Scripts/python -m pip install torch torchvision \n    --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
copy .env.example .env
    (set LLM_PROVIDER=groq, LLM_API_KEY, LLM_MODEL=openai/gpt-oss-20b)
.venv/Scripts/python -m scripts.check_env
# behind HTTPS-scanning antivirus only: $env:SSL_CERT_FILE = "C:\\path\\to\\ca-bundle.pem"

# data and evaluation corpus
.venv/Scripts/python -m scripts.download_data --limit 150
.venv/Scripts/python -m eval.select_eval_docs --target-pages 250 --max-pages 20
.venv/Scripts/python -m scripts.ingest --source mpdocvqa \n    --docs-file data/raw/mpdocvqa/eval_docs.txt          (repeat until 49/49)
.venv/Scripts/python -m scripts.build_indexes --rebuild
.venv/Scripts/python -m eval.build_qrels

# demo store (put the PDF in data/raw/pdfs/ first)
.venv/Scripts/python -m scripts.ingest --source pdf --store demo
.venv/Scripts/python -m scripts.build_indexes --store demo --rebuild

# evaluation (see README for all systems and options)
.venv/Scripts/python -m eval.run_eval
.venv/Scripts/python -m eval.eval_answers
.venv/Scripts/python -m eval.eval_feedback --split dev --grid

# one question, the API and the UI
.venv/Scripts/python -m scripts.ask --doc p60-281 \n    "Which subtype of health insurance was the most common?"
.venv/Scripts/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
npm --prefix ui install
npm --prefix ui run dev        (open http://localhost:5173)
.venv/Scripts/python -m pytest""", CODE),
         src("README.md; P1-P7 reports, section 3"), PageBreak()])

    # 13 contributions
    add([P("13. Team contribution split", H1),
         P("Three equal parts by ownership of the real files (line counts at commit time). Shared by all: README.md, "
           "configs/config.yaml, docs/PHASES.md, docs/PROGRESS.md, the phase reports and this final package. "
           "Also in docs/final/CONTRIBUTIONS.md.")])
    for i, role, phases, files, n in contrib:
        name, roll = TEAM[i]
        add([P(f"Member {i + 1}: {name} ({roll}): {role}", H2), P(f"<b>Phases:</b> {phases}. <b>Files owned:</b> {len(files)} files, {n:,} lines."),
             P(", ".join(f.replace("src/docscout/", "") for f in files), SMALL), Spacer(1, 4)])

    path = OUT / "DocScout_Project_Report.pdf"
    doc = ReportDoc(path)
    doc.multiBuild(S)
    return path


# ------------------------------------------------------------------ abstract (1 page)
def build_abstract(body_size: float = 11.0) -> Path:
    """One-page abstract in the team's IR package format (layout of their DocScout-X draft: Letter, Arial-like,
    bold centred title + subtitle, bold inline section labels). Every number is a final TEST number from the repo."""
    from reportlab.lib.pagesizes import letter

    path = OUT / "Abstract.pdf"
    lead = body_size * 1.27
    T1 = ParagraphStyle("a_t1", fontName="Helvetica-Bold", fontSize=16, leading=19, alignment=TA_CENTER, spaceAfter=2)
    T2 = ParagraphStyle("a_t2", fontName="Helvetica-Bold", fontSize=12.5, leading=15, alignment=TA_CENTER, spaceAfter=9)
    B = ParagraphStyle("a_b", fontName="Helvetica", fontSize=body_size, leading=lead, alignment=0, spaceAfter=5.5)
    BL = ParagraphStyle("a_bl", parent=B, alignment=0, leftIndent=16, bulletIndent=4, spaceAfter=1.2)
    names = ", ".join(n for n, _ in TEAM)
    TB = ParagraphStyle("a_tb", parent=B, leftIndent=34, bulletIndent=20, spaceAfter=0)
    S = [P(ABSTRACT_TITLE, T1), P(SUBTITLE, T2),
         P("<b>Team Members:</b>", ParagraphStyle("a_team", parent=B, spaceAfter=1))]
    S += [Paragraph(f"{r} – {n}", TB, bulletText="●") for n, r in TEAM]
    S += [Spacer(1, 8),
         P("<b>Problem Statement:</b> In long documents such as reports, filings and forms, the answer to a question usually sits "
           "in one paragraph, table cell or chart buried among many pages. Document- or page-level search returns whole files or "
           "pages, flat-text pipelines lose the layout and table structure, and a generated answer without a pointer to its "
           "evidence cannot be checked.", B),
         P("<b>Objective:</b> Build a query-aware retrieval system that decides whether a question is textual, numerical or "
           "visual, retrieves the exact evidence component (paragraph, table and its matching cells, or figure) inside the "
           "document, explains why each item was retrieved (page, box, scores, section path), answers only from that evidence "
           "with citations, and improves the ranking within a session from the user's relevance feedback.", B),
         P("<b>Dataset Used:</b> MP-DocVQA (Hugging Face lmms-lab/MP-DocVQA, validation split): 49 multi-page documents, "
           "250 pages, 293 questions, parsed into 2,976 components; split by document into DEV (10 documents, 76 questions, "
           "used for all tuning) and TEST (39 documents, 217 questions). Relevance labels: the answer page and every component "
           "that contains a gold answer (94.9% of questions covered). Demo document: US Census report P60-281 (31 pages).", B),
         P("<b>Methods Used / Algo Used:</b> Docling layout parsing with TableFormer and RapidOCR into typed components, each "
           "serialized to text and linked in a document structure graph (networkx). Three index branches (text, table, figure) "
           "with BM25 and BGE-base-en-v1.5 dense retrieval (FAISS), plus CLIP ViT-B/32 image embeddings for figures. A query-type "
           "classifier (rules, LLM when unsure) sets soft branch weights; scores are min-max normalized and combined by weighted-sum "
           "fusion (Reciprocal Rank Fusion as comparison), then the top 30 are reranked by a cross-encoder "
           "(ms-marco-MiniLM-L-6-v2). Numerical questions get table cell localization; graph context expansion adds the section "
           "heading, caption and neighbours; an LLM (gpt-oss-20b) answers only from the numbered evidence with validated "
           "citations or says \"Not found\". Relevance feedback uses session Rocchio query expansion and a capped cross-session "
           "memory prior. Served by FastAPI with a React UI that highlights the evidence on the page.", B),
         P("<b>Results (Metrics Used):</b> final pipeline on the TEST split (component = evidence-level retrieval):",
           ParagraphStyle("a_rh", parent=B, spaceAfter=2.5))]
    S += [Paragraph(t, BL, bulletText="•") for t in [
        "<b>Recall@1 (component):</b> 0.488 (dense BGE baseline 0.362)",
        "<b>MRR (component):</b> 0.653 (dense BGE baseline 0.544; paired t-test p &lt; 0.0001)",
        "<b>nDCG@5 (component):</b> 0.644",
        "<b>Recall@1 / MRR (page level):</b> 0.733 / 0.828 (random baseline 0.270 / 0.501)",
        "<b>ANLS (answer accuracy):</b> 0.646 final vs 0.486 top-1 baseline (+0.161) and 0.791 oracle",
        "<b>Query-type classifier accuracy:</b> 75.1%",
        "<b>Rocchio feedback, residual MRR:</b> 0.312 vs 0.289 without feedback (+0.023, not significant on TEST)"]]
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.95 * 72, rightMargin=0.95 * 72, topMargin=0.75 * 72,
                            bottomMargin=0.6 * 72, title=f"{ABSTRACT_TITLE}: abstract", author=names)
    doc.build(S)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fp = file_phase()
    lines = {f: n for _, f, n in fp}
    contrib = contributions(lines)
    write_contributions_md(contrib)
    rep = build_report(fp, contrib)
    # docs/final/Abstract.pdf is the team's own final version: regenerate it only on request
    ab = build_abstract() if "--abstract" in sys.argv else OUT / "Abstract.pdf"
    import pymupdf

    for p in (rep, ab):
        with pymupdf.open(p) as d:
            print(f"{p.relative_to(ROOT)}: {d.page_count} pages, {p.stat().st_size:,} bytes")
    for i, role, _, files, n in contrib:
        print(f"member {i + 1}: {len(files)} files, {n:,} lines - {role}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
