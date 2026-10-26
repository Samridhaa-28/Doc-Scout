// Typed client for the DocScout-X API (served through the Vite proxy at /api).

export type Modality = "text" | "table" | "figure";

export interface CellBox { row: string; column: string; value: string; bbox: number[] }
export interface ContextItem { id: string; type: string; page: number; text: string }

export interface Evidence {
  id: string;
  eid: string;
  rank: number;
  type: string;
  page: number;
  branch: Modality;
  bbox: number[];
  page_size: number[];
  text: string;
  graph_path: string;
  why: string;
  scores: Record<string, number>;
  feedback_tag: string;
  matched_terms: string[];
  localized_cells: string[][];
  localize_flag?: string | null;
  cell_boxes: CellBox[];
  table_preview?: string | null;
  reserved_slot?: string | null;
  context: ContextItem[];
}

export interface Answer {
  text: string;
  not_found: boolean;
  citations: string[];
  confidence: { cited_sources: number; not_found: boolean; slot_used: boolean; slot_variant?: string; invalid_citations: number };
  model?: string | null;
}

export interface AskResponse {
  query_id: string;
  session_id: string;
  doc_id: string;
  store: string;
  query: string;
  query_type: "textual" | "numerical" | "visual";
  classification?: { type: string; confidence: number; source: string };
  branch_weights: Record<string, number>;
  answer: Answer;
  evidence: Evidence[];
  ranking_size: number;
  votes: Record<string, number>;
  memory: { on: boolean; prior_items: number };
  feedback_active: boolean;
  answer_stale?: boolean;
  rerank_ms?: number;
}

export interface DocInfo {
  doc_id: string;
  store: string;
  source: string | null;
  pages: { page: number; width: number; height: number }[];
  components: number;
  sample_questions: string[];
}

export interface NextResponse { item: Evidence | null; position?: number; remaining: number; message?: string }

const BASE = (import.meta.env.VITE_API as string | undefined) ?? "/api";

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { headers: { "Content-Type": "application/json" }, ...init });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new Error(`${r.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }
  return r.json() as Promise<T>;
}

export const api = {
  documents: () => call<DocInfo[]>("/documents"),
  ask: (body: { doc_id: string; query: string; session_id?: string; top_k?: number; memory?: boolean; generate?: boolean }) =>
    call<AskResponse>("/ask", { method: "POST", body: JSON.stringify(body) }),
  vote: (body: { session_id: string; query_id: string; component_id: string; vote: 1 | -1 }) =>
    call<AskResponse>("/feedback", { method: "POST", body: JSON.stringify({ ...body, target: "evidence" }) }),
  voteAnswer: (body: { session_id: string; query_id: string; vote: 1 | -1 }) =>
    call<{ recorded: boolean }>("/feedback", { method: "POST", body: JSON.stringify({ ...body, target: "answer" }) }),
  next: (session_id: string, query_id: string) =>
    call<NextResponse>(`/next?session_id=${encodeURIComponent(session_id)}&query_id=${encodeURIComponent(query_id)}`),
  pageImage: (doc: string, page: number) => `${BASE}/page-image/${encodeURIComponent(doc)}/${page}`,
  upload: async (file: File): Promise<{ status: string; doc_id: string }> => {
    const body = new FormData();
    body.append("file", file);
    const r = await fetch(`${BASE}/ingest`, { method: "POST", body });   // no JSON content-type: multipart
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail ?? detail; } catch { /* not JSON */ }
      throw new Error(`${r.status}: ${detail}`);
    }
    return r.json();
  },
  ingestStatus: () => call<IngestStatus>("/ingest/status"),
};

export interface IngestStatus {
  status: "idle" | "queued" | "running" | "done" | "error";
  file: string | null;
  doc_id: string | null;
  error: string | null;
  log_tail: string[];
  elapsed_s: number | null;
}

export const modalityOf = (type: string): Modality =>
  type === "table" ? "table" : type === "figure" ? "figure" : "text";
