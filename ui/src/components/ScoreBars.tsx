import type { Evidence } from "../api";

// Small horizontal bars for the why-line stages: BM25 / BGE (/ CLIP) normalized, CE as sigmoid(logit),
// feedback as a diverging bar around 0 (±0.5 full scale).
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

function Bar({ label, value, shown, title }: { label: string; value: number; shown: string; title: string }) {
  return (
    <div className="bar" title={title}>
      <span className="bar-label">{label}</span>
      <span className="bar-track"><span className="bar-fill" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} /></span>
      <span className="bar-val">{shown}</span>
    </div>
  );
}

function FbBar({ value }: { value: number }) {
  const w = Math.min(Math.abs(value) / 0.5, 1) * 50;
  return (
    <div className="bar fb" title="feedback change on the normalized scale (Rocchio / memory)">
      <span className="bar-label">fb</span>
      <span className="bar-track diverging">
        <span className={`bar-fill ${value < 0 ? "neg" : "pos"}`}
              style={value < 0 ? { right: "50%", width: `${w}%` } : { left: "50%", width: `${w}%` }} />
        <span className="bar-zero" />
      </span>
      <span className="bar-val">{value >= 0 ? "+" : ""}{value.toFixed(2)}</span>
    </div>
  );
}

export default function ScoreBars({ ev }: { ev: Evidence }) {
  const s = ev.scores;
  const ce = s.rerank;
  return (
    <div className="bars">
      {s.bm25_norm !== undefined && (
        <Bar label="bm25" value={s.bm25_norm} shown={`${s.bm25_raw?.toFixed(1) ?? "-"}`} title="BM25 raw (bar: min-max within branch)" />
      )}
      {s.dense_bge_norm !== undefined && (
        <Bar label="bge" value={s.dense_bge_norm} shown={`${s.dense_bge_raw?.toFixed(2) ?? "-"}`} title="BGE cosine (bar: min-max within branch)" />
      )}
      {s.clip_norm !== undefined && (
        <Bar label="clip" value={s.clip_norm} shown={`${s.clip_raw?.toFixed(2) ?? "-"}`} title="CLIP cosine (bar: min-max within branch)" />
      )}
      {ce !== undefined ? (
        <Bar label="CE" value={sigmoid(ce)} shown={ce.toFixed(2)} title="cross-encoder logit (bar: sigmoid)" />
      ) : (
        <Bar label="CE" value={0} shown="-" title="outside the reranked top 30" />
      )}
      {s.feedback !== undefined && <FbBar value={s.feedback} />}
    </div>
  );
}
