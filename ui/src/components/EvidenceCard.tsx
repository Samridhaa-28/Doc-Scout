import { forwardRef, useState } from "react";
import { modalityOf, type Evidence } from "../api";
import ScoreBars from "./ScoreBars";

interface Props {
  ev: Evidence;
  index: number;
  selected: boolean;
  vote: number;               // +1 / -1 / 0 for this session and question
  moved?: number;             // rank change after the last vote (positive = moved up)
  cited: boolean;
  onSelect: () => void;
  onVote: (v: 1 | -1) => void;
  busy: boolean;
}

/** "Page 8 -> Section 'Number and percentage' -> Table 0" as breadcrumb parts. */
function crumbs(path: string): string[] {
  return path.split("->").map((s) => s.trim()).filter(Boolean);
}

const EvidenceCard = forwardRef<HTMLElement, Props>(function EvidenceCard(
  { ev, index, selected, vote, moved, cited, onSelect, onVote, busy }, ref,
) {
  const [open, setOpen] = useState(false);
  const mod = modalityOf(ev.type);
  const cells = ev.localized_cells;
  return (
    <article
      ref={ref}
      data-flip-id={ev.id}
      className={`card mod-${mod}${selected ? " selected" : ""}${vote > 0 ? " voted-up" : ""}`}
      onClick={onSelect}
      aria-selected={selected}
    >
      <header className="card-head">
        <span className={`eid mod-${mod}`}>{index + 1}</span>
        <span className="type">{ev.type}</span>
        <span className="page-no">p.{ev.page}</span>
        {cited && <span className="cited" title="cited by the answer">cited {ev.eid}</span>}
        {ev.feedback_tag && <span className={`tag tag-${ev.feedback_tag.split("+")[0]}`}>{ev.feedback_tag}</span>}
        {ev.reserved_slot && <span className="tag tag-slot" title="modality slot reservation">slot</span>}
        {moved ? <span className={`moved ${moved > 0 ? "up" : "down"}`}>{moved > 0 ? "▲" : "▼"}{Math.abs(moved)}</span> : null}
        <span className="spacer" />
        <div className="votes" onClick={(e) => e.stopPropagation()}>
          <button className={`vote up${vote > 0 ? " on" : ""}`} disabled={busy} onClick={() => onVote(1)}
                  title="relevant (↑)" aria-label="mark relevant">▲</button>
          <button className={`vote down${vote < 0 ? " on" : ""}`} disabled={busy} onClick={() => onVote(-1)}
                  title="not relevant (↓)" aria-label="mark not relevant">▼</button>
        </div>
      </header>

      <nav className="crumbs" aria-label="graph path">
        {crumbs(ev.graph_path).map((c, i) => (
          <span key={i} className="crumb">{i > 0 && <span className="sep">›</span>}{c}</span>
        ))}
      </nav>

      <p className={`snippet${open ? " open" : ""}`}>{ev.text}</p>

      {cells.length > 0 && (
        <table className="cells">
          <caption>localized cells · {ev.localize_flag}</caption>
          <tbody>
            {cells.slice(0, 6).map((c, i) => (
              <tr key={i}><th>{c[0]}</th><td className="col">{c[1]}</td><td className="val">{c[2]}</td></tr>
            ))}
          </tbody>
        </table>
      )}

      <ScoreBars ev={ev} />

      <button className="why-toggle" onClick={(e) => { e.stopPropagation(); setOpen(!open); }}>
        {open ? "less" : "why retrieved"}
      </button>
      {open && (
        <div className="why">
          <code>{ev.why}</code>
          {ev.context.length > 0 && (
            <ul className="ctx">
              {ev.context.map((c) => <li key={c.id}><b>{c.type} p.{c.page}</b> {c.text}</li>)}
            </ul>
          )}
        </div>
      )}
    </article>
  );
});

export default EvidenceCard;
