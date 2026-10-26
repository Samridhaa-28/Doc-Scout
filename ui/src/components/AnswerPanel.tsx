import { modalityOf, type AskResponse } from "../api";

interface Props {
  res: AskResponse;
  stale: boolean;
  answerVote: number;
  busy: boolean;
  onCite: (eid: string) => void;
  onAnswerVote: (v: 1 | -1) => void;
  onReanswer: () => void;
}

/** Answer with citation chips, the confidence strip, and the answer thumbs (logged only). */
export default function AnswerPanel({ res, stale, answerVote, busy, onCite, onAnswerVote, onReanswer }: Props) {
  const a = res.answer;
  const c = a.confidence;
  const byEid = new Map(res.evidence.map((e) => [e.eid, e]));
  return (
    <section className={`answer${a.not_found ? " not-found" : ""}`} aria-label="answer">
      <div className="answer-meta">
        <span className={`qtype q-${res.query_type}`}>{res.query_type}</span>
        <span className="muted">
          {res.classification?.source ?? ""} {res.classification ? `· ${res.classification.confidence.toFixed(2)}` : ""}
        </span>
        <span className="spacer" />
        <span className="muted">{a.model ?? ""}</span>
      </div>
      <p className="answer-text">{a.not_found ? "Not found in the retrieved evidence" : a.text}</p>
      <div className="cites">
        {a.citations.map((eid) => {
          const e = byEid.get(eid);
          return (
            <button key={eid} className={`chip mod-${e ? modalityOf(e.type) : "text"}`} onClick={() => onCite(eid)}
                    title={e ? `${e.type}, page ${e.page}` : "cited item no longer in the list"}>
              {eid}{e ? ` · p.${e.page}` : ""}
            </button>
          );
        })}
      </div>
      <div className="strip" aria-label="confidence">
        <span className={c.cited_sources > 0 ? "ok" : "warn"}>{c.cited_sources} cited source{c.cited_sources === 1 ? "" : "s"}</span>
        <span className={c.not_found ? "warn" : "ok"}>{c.not_found ? "not found" : "answered"}</span>
        <span className={c.slot_used ? "info" : "dim"}>{c.slot_used ? "modality slot used" : "no slot"}</span>
        {res.memory.on && <span className="info">memory · {res.memory.prior_items} prior{res.memory.prior_items === 1 ? "" : "s"}</span>}
        {c.invalid_citations > 0 && <span className="warn">{c.invalid_citations} invalid citation(s) dropped</span>}
        <span className="spacer" />
        <span className="answer-votes" title="answer feedback is logged only, never used for ranking">
          <button className={answerVote > 0 ? "on" : ""} disabled={busy} onClick={() => onAnswerVote(1)} aria-label="good answer">👍</button>
          <button className={answerVote < 0 ? "on" : ""} disabled={busy} onClick={() => onAnswerVote(-1)} aria-label="bad answer">👎</button>
        </span>
      </div>
      {stale && (
        <div className="stale">
          Evidence re-ranked by your votes · the answer above is from the earlier evidence.
          <button onClick={onReanswer} disabled={busy}>Re-answer with current evidence</button>
        </div>
      )}
    </section>
  );
}
