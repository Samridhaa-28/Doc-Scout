# P7 real API session (uvicorn, 127.0.0.1:8000)

### GET /documents (first 2 of 50)
```json
[
 {
  "doc_id": "p60-281",
  "store": "demo",
  "source": "pdf",
  "components": 344,
  "sample_questions": [
   "How many people had Medicaid coverage in 2022, in thousands?",
   "Which subtype of health insurance was the most common?",
   "According to Figure 2, which age group had the highest uninsured rate in 2022?"
  ],
  "pages": 31
 },
 {
  "doc_id": "flpp0227",
  "store": "eval",
  "source": "mpdocvqa",
  "components": 114,
  "sample_questions": [
   "What is the service provided by 'Container Services' for the frozen shipments from west coast to Hawaii ?",
   "Who is the new manager for  Edinburg Ice and Cold Storage ?",
   "What is the title in the last rectangle ?",
   "What is the title in the first rectangle ?",
   "Which company has vacancies to the post of general manager and operating engineer?",
   "How many years of experience does the Refrigerated Warehouse Executive have ?"
  ],
  "pages": 7
 }
]
```

### POST /ask (2348 ms)
```json
{
 "session_id": "3a39b6af4d3f41d8bb5e8d6c2d63f45d",
 "query_id": "3b81d2ce44ba499eb84331045c770797",
 "query_type": "numerical",
 "answer": {
  "text": "Not found in the retrieved evidence",
  "not_found": true,
  "citations": [],
  "confidence": {
   "cited_sources": 0,
   "not_found": true,
   "slot_used": true,
   "slot_variant": "conditional",
   "invalid_citations": 0
  },
  "model": "openai/gpt-oss-20b",
  "llm": {
   "calls": 0,
   "cache_hits": 1,
   "retries": 0,
   "prompt_tokens": 0,
   "completion_tokens": 0,
   "total_tokens": 0
  }
 },
 "ranking_size": 344,
 "feedback_active": false,
 "evidence": [
  {
   "eid": "E1",
   "id": "p60-281_p8_paragraph_12",
   "type": "paragraph",
   "page": 8,
   "tag": "",
   "fb": 0.0,
   "why": "numerical (text 0.3) | bm25 3.47->0.43, bge 0.73->1.00, fused 0.180, CE 6.21 | terms: people, 2022 | Page 8 -> Section 'Highlights' -> Paragraph 12 | bbox 455,426,763,676"
  },
  {
   "eid": "E2",
   "id": "p60-281_p9_paragraph_9",
   "type": "paragraph",
   "page": 9,
   "tag": "",
   "fb": 0.0,
   "why": "numerical (text 0.3) | bm25 3.94->0.48, bge 0.70->0.93, fused 0.185, CE 5.59 | terms: people, coverage, 2022 | Page 9 -> Section 'ESTIMATES OF HEALTH INSURANCE COVERAG..."
  },
  {
   "eid": "E3",
   "id": "p60-281_p12_paragraph_2",
   "type": "paragraph",
   "page": 12,
   "tag": "",
   "fb": 0.0,
   "why": "numerical (text 0.3) | bm25 2.98->0.37, bge 0.65->0.81, fused 0.150, CE 4.15 | terms: medicaid, coverage, 2022 | Page 12 -> Section 'Public Coverage' -> Paragraph 2 | bbo"
  },
  {
   "eid": "E4",
   "id": "p60-281_p18_paragraph_6",
   "type": "paragraph",
   "page": 18,
   "tag": "",
   "fb": 0.0,
   "why": "numerical (text 0.3) | bm25 3.35->0.41, bge 0.63->0.76, fused 0.155, CE 3.91 | terms: people, coverage, 2022 | Page 18 -> Section 'Coverage Type by Demographic and Soci.."
  },
  {
   "eid": "E5",
   "id": "p60-281_p6_table_0",
   "type": "table",
   "page": 6,
   "tag": "",
   "fb": 0.0,
   "why": "numerical (table 0.6) | bm25 4.67->1.00, bge 0.55->0.24, fused 0.463, CE -2.81 | terms: people, coverage, 2022 | Page 6 -> Section 'TABLE' -> Table 0 | bbox 106,148,1115,"
  }
 ]
}
```

### POST /feedback down-vote p60-281_p6_table_0
```json
{
 "recorded": true,
 "votes": {
  "p60-281_p6_table_0": -1
 },
 "answer_stale": true,
 "rerank_ms": 221.9,
 "feedback_active": true,
 "answer": "Not found in the retrieved evidence",
 "evidence": [
  {
   "eid": "E1",
   "id": "p60-281_p8_paragraph_12",
   "type": "paragraph",
   "page": 8,
   "tag": "rocchio",
   "fb": 0.0,
   "why": "numerical (text 0.3) | bm25 3.47->0.43, bge 0.68->1.00, fused 0.180, CE 6.21, fb +0.00 | rocchio | terms: people, 2022 | Page 8 -> Section 'Highlights' -> Paragraph 12 | "
  },
  {
   "eid": "E2",
   "id": "p60-281_p9_paragraph_9",
   "type": "paragraph",
   "page": 9,
   "tag": "rocchio",
   "fb": -0.03,
   "why": "numerical (text 0.3) | bm25 3.94->0.48, bge 0.64->0.92, fused 0.185, CE 5.59, fb -0.03 | rocchio | terms: people, coverage, 2022 | Page 9 -> Section 'ESTIMATES OF HEALTH "
  },
  {
   "eid": "E3",
   "id": "p60-281_p14_paragraph_0",
   "type": "paragraph",
   "page": 14,
   "tag": "rocchio",
   "fb": 0.007,
   "why": "numerical (text 0.3) | bm25 2.96->0.36, bge 0.64->0.90, fused 0.158, CE 3.06, fb +0.01 | rocchio | terms: medicaid, 2022 | Page 14 -> Section 'Children Under the Age of 1"
  },
  {
   "eid": "E4",
   "id": "p60-281_p12_paragraph_2",
   "type": "paragraph",
   "page": 12,
   "tag": "rocchio",
   "fb": -0.061,
   "why": "numerical (text 0.3) | bm25 2.98->0.37, bge 0.60->0.81, fused 0.150, CE 4.15, fb -0.06 | rocchio | terms: medicaid, coverage, 2022 | Page 12 -> Section 'Public Coverage' "
  },
  {
   "eid": "E5",
   "id": "p60-281_p8_table_0",
   "type": "table",
   "page": 8,
   "tag": "rocchio",
   "fb": -0.06,
   "why": "numerical (table 0.6) | bm25 2.01->0.41, bge 0.54->1.00, fused 0.352, CE 0.43, fb -0.06 | rocchio | terms: medicaid, coverage, 2022, thousands | Page 8 -> Section 'Number"
  }
 ]
}
```

### GET /next until the p8 table
```json
[
 {
  "position": 5,
  "remaining": 337,
  "id": "p60-281_p22_paragraph_15",
  "type": "paragraph",
  "cells": []
 },
 {
  "position": 6,
  "remaining": 336,
  "id": "p60-281_p18_paragraph_6",
  "type": "paragraph",
  "cells": []
 },
 {
  "position": 7,
  "remaining": 335,
  "id": "p60-281_p22_paragraph_8",
  "type": "paragraph",
  "cells": []
 },
 {
  "position": 8,
  "remaining": 334,
  "id": "p60-281_p9_paragraph_11",
  "type": "paragraph",
  "cells": []
 },
 {
  "position": 9,
  "remaining": 333,
  "id": "p60-281_p12_paragraph_3",
  "type": "paragraph",
  "cells": []
 },
 {
  "position": 10,
  "remaining": 332,
  "id": "p60-281_p9_paragraph_13",
  "type": "paragraph",
  "cells": []
 },
 {
  "position": 11,
  "remaining": 331,
  "id": "p60-281_p11_heading_0",
  "type": "heading",
  "cells": []
 },
 {
  "position": 12,
  "remaining": 330,
  "id": "p60-281_p8_heading_1",
  "type": "heading",
  "cells": []
 }
]
```

### POST /feedback up-vote p60-281_p8_table_0
```json
{
 "votes": {
  "p60-281_p6_table_0": -1,
  "p60-281_p8_table_0": 1
 },
 "rerank_ms": 194.5,
 "evidence": [
  {
   "eid": "E1",
   "id": "p60-281_p8_table_0",
   "type": "table",
   "page": 8,
   "tag": "pinned",
   "fb": 0.105,
   "why": "numerical (table 0.6) | bm25 2.01->0.41, bge 0.85->1.00, fused 0.352, CE 0.43, fb +0.10 | pinned | terms: medicaid, coverage, 2022, thousands | Page 8 -> Section 'Number "
  },
  {
   "eid": "E2",
   "id": "p60-281_p8_paragraph_12",
   "type": "paragraph",
   "page": 8,
   "tag": "rocchio",
   "fb": -0.09,
   "why": "numerical (text 0.3) | bm25 3.47->0.43, bge 0.78->1.00, fused 0.180, CE 6.21, fb -0.09 | rocchio | terms: people, 2022 | Page 8 -> Section 'Highlights' -> Paragraph 12 | "
  },
  {
   "eid": "E3",
   "id": "p60-281_p9_paragraph_9",
   "type": "paragraph",
   "page": 9,
   "tag": "rocchio",
   "fb": -0.096,
   "why": "numerical (text 0.3) | bm25 3.94->0.48, bge 0.76->0.97, fused 0.189, CE 5.59, fb -0.10 | rocchio | terms: people, coverage, 2022 | Page 9 -> Section 'ESTIMATES OF HEALTH "
  },
  {
   "eid": "E4",
   "id": "p60-281_p14_paragraph_0",
   "type": "paragraph",
   "page": 14,
   "tag": "rocchio",
   "fb": -0.091,
   "why": "numerical (text 0.3) | bm25 2.96->0.36, bge 0.73->0.89, fused 0.157, CE 3.06, fb -0.09 | rocchio | terms: medicaid, 2022 | Page 14 -> Section 'Children Under the Age of 1"
  },
  {
   "eid": "E5",
   "id": "p60-281_p18_paragraph_6",
   "type": "paragraph",
   "page": 18,
   "tag": "rocchio",
   "fb": -0.145,
   "why": "numerical (text 0.3) | bm25 3.35->0.41, bge 0.70->0.82, fused 0.160, CE 3.91, fb -0.14 | rocchio | terms: people, coverage, 2022 | Page 18 -> Section 'Coverage Type by De"
  }
 ],
 "pinned_cells": [
  [
   "Marketplace coverage 2",
   "2022.Number",
   "11,840"
  ],
  [
   "Marketplace coverage 2",
   "2022.Margin of error 1 (±)",
   "461"
  ],
  [
   "Marketplace coverage 2",
   "2022.Percent",
   "3.6"
  ],
  [
   "Marketplace coverage 2",
   "2022.Margin of error 1 (±) (1)",
   "0.1"
  ],
  [
   "Marketplace coverage 2",
   "Change in.percent (2022 less 2021)",
   "0.1"
  ],
  [
   "Medicaid 2",
   "2022.Number",
   "62,050"
  ],
  [
   "Medicaid 2",
   "2022.Margin of error 1 (±)",
   "1,112"
  ],
  [
   "Medicaid 2",
   "2022.Percent",
   "18.8"
  ],
  [
   "Medicaid 2",
   "2022.Margin of error 1 (±) (1)",
   "0.3"
  ],
  [
   "Medicaid 2",
   "Change in.percent (2022 less 2021)",
   "-0.1"
  ]
 ],
 "cell_boxes": [
  {
   "row": "Marketplace coverage 2",
   "column": "2022.Number",
   "value": "11,840",
   "bbox": [
    720.930419921875,
    1048.095947265625,
    773.7302856445312,
    1067.904052734375
   ]
  },
  {
   "row": "Marketplace coverage 2",
   "column": "2022.Margin of error 1 (±)",
   "value": "461",
   "bbox": [
    829.3624267578125,
    1048.095947265625,
    858.162353515625,
    1067.904052734375
   ]
  },
  {
   "row": "Marketplace coverage 2",
   "column": "2022.Percent",
   "value": "3.6",
   "bbox": [
    916.1624145507812,
    1048.095947265625,
    940.162353515625,
    1067.904052734375
   ]
  },
  {
   "row": "Marketplace coverage 2",
   "column": "2022.Margin of error 1 (±) (1)",
   "value": "0.1",
   "bbox": [
    1089.986572265625,
    1010.0960083007812,
    1113.986572265625,
    1029.904052734375
   ]
  },
  {
   "row": "Marketplace coverage 2",
   "column": "Change in.percent (2022 less 2021)",
   "value": "0.1",
   "bbox": [
    1089.986572265625,
    1010.0960083007812,
    1113.986572265625,
    1029.904052734375
   ]
  },
  {
   "row": "Medicaid 2",
   "column": "2022.Number",
   "value": "62,050",
   "bbox": [
    720.930419921875,
    1129.095947265625,
    773.7302856445312,
    1148.904052734375
   ]
  },
  {
   "row": "Medicaid 2",
   "column": "2022.Margin of error 1 (±)",
   "value": "1,112",
   "bbox": [
    814.96240234375,
    1129.095947265625,
    858.1622924804688,
    1148.904052734375
   ]
  },
  {
   "row": "Medicaid 2",
   "column": "2022.Percent",
   "value": "18.8",
   "bbox": [
    906.5623779296875,
    1129.095947265625,
    940.1622924804688,
    1148.904052734375
   ]
  },
  {
   "row": "Medicaid 2",
   "column": "2022.Margin of error 1 (±) (1)",
   "value": "0.3",
   "bbox": [
    665.298583984375,
    991.0960083007812,
    689.2985229492188,
    1010.9039916992188
   ]
  }
 ],
 "table_bbox": [
  103.52,
  854.95,
  1120.14,
  1193.91
 ]
}
```

### POST /feedback answer vote (logged only)
```json
{
 "recorded": true,
 "target": "answer",
 "query_id": "3b81d2ce44ba499eb84331045c770797"
}
```

### GET /page-image/p60-281/8
```json
{
 "status": 200,
 "content-type": "image/png",
 "bytes": 361072
}
```

### GET /next wrong session
```json
{
 "status": 404,
 "body": {
  "detail": "unknown query_id for this session"
 }
}
```
