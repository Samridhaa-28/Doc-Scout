# Retrieval results (within-document search)

Evaluation set: 49 documents from `eval_docs.txt`, 293 questions. Page metrics on documents with >= 2 pages: 246 questions (main page-level number). Component metrics: 278 questions with >= 1 relevant component.

## Page level: documents with >= 2 pages (main)

| system | R@1 | R@5 | R@10 | MRR | nDCG@10 | median ms/query |
|---|---|---|---|---|---|---|
| random (expected) | 0.246 | 0.770 | 0.938 | 0.473 | - | - |
| bm25_all | 0.593 | 0.793 | 0.833 | 0.679 | 0.716 | 0.3 |
| dense_bge_all | 0.606 | 0.907 | 0.976 | 0.738 | 0.795 | 35.4 |
| dense_minilm_all | 0.537 | 0.902 | 0.963 | 0.696 | 0.760 | 9.4 |

## Page level: all questions

| system | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| random (expected) | 0.367 | 0.807 | 0.948 | 0.557 | - |
| bm25_all | 0.625 | 0.792 | 0.826 | 0.696 | 0.727 |
| dense_bge_all | 0.669 | 0.922 | 0.980 | 0.780 | 0.828 |
| dense_minilm_all | 0.611 | 0.918 | 0.969 | 0.745 | 0.798 |

## Component level (questions with >= 1 relevant component)

| system | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| bm25_all | 0.380 | 0.588 | 0.633 | 0.522 | 0.530 |
| dense_bge_all | 0.331 | 0.633 | 0.726 | 0.514 | 0.551 |
| dense_minilm_all | 0.244 | 0.593 | 0.729 | 0.448 | 0.502 |
