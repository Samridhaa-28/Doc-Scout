import { useEffect, useRef } from "react";
import { api, modalityOf, type DocInfo, type Evidence } from "../api";

interface Props {
  doc: DocInfo;
  page: number;
  evidence: Evidence[];
  selected?: Evidence;
  onPage: (p: number) => void;
  onSelect: (id: string) => void;
}

/** The real page image with every evidence box of this page drawn in its modality colour,
 *  the selected one emphasised, and its localized cells (digital PDFs) marked in place. */
export default function PageViewer({ doc, page, evidence, selected, onPage, onSelect }: Props) {
  const info = doc.pages.find((p) => p.page === page);
  const w = selected?.page === page ? selected.page_size[0] : info?.width ?? 1000;
  const h = selected?.page === page ? selected.page_size[1] : info?.height ?? 1400;
  const onPageEv = evidence.filter((e) => e.page === page);
  const pages = doc.pages.map((p) => p.page);
  const i = pages.indexOf(page);
  const sheet = useRef<HTMLDivElement>(null);
  // keep the selected box in view: scroll the page so its top sits a little below the viewer top
  const scrollToSel = () => {
    const el = sheet.current;
    if (!el || !selected || selected.page !== page) return;
    const img = el.querySelector("img");
    if (img && img.clientHeight) el.scrollTo({ top: Math.max(0, (selected.bbox[1] / h) * img.clientHeight - 60), behavior: "smooth" });
  };
  useEffect(scrollToSel, [selected?.id, page]); // eslint-disable-line react-hooks/exhaustive-deps
  const cellsListed = selected && selected.page === page && selected.localized_cells.length > 0;
  return (
    <section className="viewer" aria-label="page viewer">
      <div className="viewer-bar">
        <button onClick={() => i > 0 && onPage(pages[i - 1])} disabled={i <= 0} aria-label="previous page">‹</button>
        <span>page <b>{page}</b> / {pages.length}</span>
        <button onClick={() => i < pages.length - 1 && onPage(pages[i + 1])} disabled={i >= pages.length - 1}
                aria-label="next page">›</button>
        <span className="spacer" />
        <span className="legend"><i className="mod-text" />text <i className="mod-table" />table <i className="mod-figure" />figure</span>
      </div>
      <div className="sheet" ref={sheet}>
        <div className="page-wrap">{/* sized by the image, so the overlay scales with it (not with the scroll box) */}
        <img src={api.pageImage(doc.doc_id, page)} alt={`${doc.doc_id} page ${page}`} onLoad={scrollToSel} />
        <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="overlay">
          {onPageEv.map((e) => {
            const [x0, y0, x1, y1] = e.bbox;
            const sel = selected?.id === e.id;
            return (
              <g key={e.id} className={`box mod-${modalityOf(e.type)}${sel ? " sel" : ""}`} onClick={() => onSelect(e.id)}>
                <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} />
                <text x={x0} y={Math.max(y0 - 6, 14)}>{evidence.indexOf(e) + 1}</text>
              </g>
            );
          })}
          {selected && selected.page === page && selected.cell_boxes.map((c, k) => {
            const [x0, y0, x1, y1] = c.bbox;
            return <rect key={k} className="cellbox" x={x0 - 3} y={y0 - 2} width={x1 - x0 + 6} height={y1 - y0 + 4} />;
          })}
        </svg>
        </div>
      </div>
      {cellsListed && selected!.cell_boxes.length < selected!.localized_cells.length && (
        <p className="viewer-note">
          {selected!.cell_boxes.length === 0
            ? "Cells listed on the card: this page has no text layer, so only the table box is marked."
            : `${selected!.cell_boxes.length} of ${selected!.localized_cells.length} cells located on the page; repeated values without a unique row/column match are not marked.`}
        </p>
      )}
    </section>
  );
}
