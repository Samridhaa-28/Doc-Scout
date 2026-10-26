# DocScout: contribution split

Three equal parts. Ownership follows the real files in the repository (line counts from the files at commit time).
Shared by all three: `README.md`, `configs/config.yaml`, `docs/PHASES.md`, `docs/PROGRESS.md`, the phase reports and this final package.

## Member 1: [Name 1] ([Roll 1]): Ingestion, parsing, indexing, data and qrels

Phases: P1 setup + schema, P2 parsing/serialization/graph, P3 indexes, P4 data subset, qrels, ingest isolation

Files owned (33 files, 3,622 lines):

- `src/docscout/config.py`
- `src/docscout/schema.py`
- `src/docscout/env_check.py`
- `src/docscout/dataset/mpdocvqa.py`
- `src/docscout/dataset/explore.py`
- `src/docscout/ingestion/parser.py`
- `src/docscout/ingestion/runner.py`
- `src/docscout/ingestion/serializer.py`
- `src/docscout/ingestion/validate.py`
- `src/docscout/ingestion/visualize.py`
- `src/docscout/graph/builder.py`
- `src/docscout/indexing/base.py`
- `src/docscout/indexing/bm25_index.py`
- `src/docscout/indexing/dense_index.py`
- `src/docscout/indexing/visual_index.py`
- `src/docscout/indexing/build_all.py`
- `src/docscout/indexing/search.py`
- `scripts/check_env.py`
- `scripts/download_data.py`
- `scripts/explore_data.py`
- `scripts/ingest.py`
- `scripts/check_components.py`
- `scripts/visualize_components.py`
- `scripts/build_indexes.py`
- `scripts/search_cli.py`
- `eval/build_qrels.py`
- `eval/select_eval_docs.py`
- `tests/test_schema.py`
- `tests/test_config.py`
- `tests/test_download_utils.py`
- `tests/test_ingestion.py`
- `tests/test_ingest_isolation.py`
- `tests/test_indexing.py`

## Member 2: [Name 2] ([Roll 2]): Retrieval, fusion, rerank, slots, table locator, graph context, evaluation

Phases: P4 metrics + baselines, P5 classifier/fusion/rerank/tuning/ablation, P6 locator, graph context, slots, ANLS + locator eval, P7 feedback eval

Files owned (24 files, 3,570 lines):

- `src/docscout/retrieval/retrievers.py`
- `src/docscout/retrieval/normalize.py`
- `src/docscout/retrieval/fusion.py`
- `src/docscout/retrieval/slots.py`
- `src/docscout/rerank/cross_encoder.py`
- `src/docscout/query/classifier.py`
- `src/docscout/localization/table_locator.py`
- `src/docscout/context/graph_expander.py`
- `eval/metrics.py`
- `eval/run_eval.py`
- `eval/split.py`
- `eval/tune_fusion.py`
- `eval/eval_classifier.py`
- `eval/plot_results.py`
- `eval/eval_locator.py`
- `eval/eval_answers.py`
- `eval/eval_feedback.py`
- `tests/test_retrieval.py`
- `tests/test_classifier.py`
- `tests/test_slots.py`
- `tests/test_table_locator.py`
- `tests/test_graph_expander.py`
- `tests/test_eval.py`
- `tests/test_split.py`

## Member 3: [Name 3] ([Roll 3]): LLM answering and grounding, feedback, API, React UI, tests

Phases: P5 LLM client, P6 prompts/answerer/provenance/pipeline/ask.py, P7 Rocchio, memory, cell boxes, API, UI

Files owned (30 files, 3,527 lines):

- `src/docscout/reasoning/llm_client.py`
- `src/docscout/reasoning/prompts.py`
- `src/docscout/reasoning/answerer.py`
- `src/docscout/pipeline.py`
- `src/docscout/evidence/provenance.py`
- `src/docscout/evidence/cell_boxes.py`
- `src/docscout/feedback/session_store.py`
- `src/docscout/feedback/rocchio.py`
- `src/docscout/feedback/memory.py`
- `api/main.py`
- `ui/src/App.tsx`
- `ui/src/api.ts`
- `ui/src/styles.css`
- `ui/src/components/AnswerPanel.tsx`
- `ui/src/components/EvidenceCard.tsx`
- `ui/src/components/PageViewer.tsx`
- `ui/src/components/ScoreBars.tsx`
- `ui/src/main.tsx`
- `ui/index.html`
- `ui/package.json`
- `ui/tsconfig.json`
- `ui/vite.config.ts`
- `scripts/ask.py`
- `tests/test_llm_client.py`
- `tests/test_answerer.py`
- `tests/test_pipeline_smoke.py`
- `tests/test_rocchio.py`
- `tests/test_memory.py`
- `tests/test_api.py`
- `tests/test_cell_boxes.py`
