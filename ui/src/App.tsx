import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { api, type AskResponse, type DocInfo, type Evidence } from "./api";
import AnswerPanel from "./components/AnswerPanel";
import UploadButton from "./components/UploadButton";
import EvidenceCard from "./components/EvidenceCard";
import PageViewer from "./components/PageViewer";

const params = new URLSearchParams(window.location.search);
const STILL = params.get("still") === "1";   // no animations (headless screenshots)

/** FLIP: cards slide from their old position to the new one when the list re-ranks after a vote. */
function useFlip(container: React.RefObject<HTMLElement>, key: string) {
  const prev = useRef(new Map<string, number>());
  useLayoutEffect(() => {
    const el = container.current;
    if (!el) return;
    const animate = !STILL && document.visibilityState === "visible"
      && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const now = new Map<string, number>();
    el.querySelectorAll<HTMLElement>("[data-flip-id]").forEach((n) => {
      n.getAnimations().forEach((a) => a.cancel());   // never stack: a frozen animation would hide the card
      const id = n.dataset.flipId!;
      const top = n.getBoundingClientRect().top;
      now.set(id, top);
      const before = prev.current.get(id);
      if (!animate) return;
      if (before === undefined) {
        if (prev.current.size) n.animate([{ opacity: 0.35, transform: "translateY(12px)" }, { opacity: 1, transform: "none" }],
          { duration: 380, easing: "cubic-bezier(.2,.7,.2,1)" });
      } else if (Math.abs(before - top) > 1) {
        n.animate([{ transform: `translateY(${before - top}px)` }, { transform: "none" }],
          { duration: 520, easing: "cubic-bezier(.2,.7,.2,1)" });
      }
    });
    prev.current = now;
    // safety net: if the page is not painting frames (background tab, headless), jump every animation to its end
    const t = window.setTimeout(() => {
      el.querySelectorAll<HTMLElement>("[data-flip-id]").forEach((n) => n.getAnimations().forEach((a) => a.finish()));
    }, 600);
    return () => window.clearTimeout(t);
  }, [container, key]);
}

export default function App() {
  const [docs, setDocs] = useState<DocInfo[]>([]);
  const [docId, setDocId] = useState<string>(params.get("doc") ?? "");
  const [query, setQuery] = useState<string>(params.get("q") ?? "");
  const [memory, setMemory] = useState<boolean>(params.get("memory") === "1");
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const [res, setRes] = useState<AskResponse | null>(null);
  const [items, setItems] = useState<Evidence[]>([]);
  const [citeIds, setCiteIds] = useState<Record<string, string>>({});   // answer eid -> component id
  const [selectedId, setSelectedId] = useState<string | undefined>();
  const [page, setPage] = useState<number>(1);
  const [moved, setMoved] = useState<Record<string, number>>({});
  const [stale, setStale] = useState(false);
  const [answerVote, setAnswerVote] = useState(0);
  const [busy, setBusy] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [exhausted, setExhausted] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const itemsRef = useRef<Evidence[]>([]);
  itemsRef.current = items;
  const autoRan = useRef(false);

  useFlip(listRef, items.map((e) => e.id).join("|"));

  useEffect(() => {
    api.documents().then((d) => {
      setDocs(d);
      if (!docId && d.length) setDocId(d[0].doc_id);
    }).catch((e) => setError(String(e)));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // after an upload: reload the list only (the current selection stays as it is)
  const refreshDocs = useCallback(() => { api.documents().then(setDocs).catch((e) => setError(String(e))); }, []);

  const doc = docs.find((d) => d.doc_id === docId);
  const selected = items.find((e) => e.id === selectedId);
  const votes = res?.votes ?? {};
  const cited = useMemo(() => new Set(Object.values(citeIds)), [citeIds]);

  const select = useCallback((id: string | undefined) => {
    setSelectedId(id);
    const e = items.find((x) => x.id === id);
    if (e) setPage(e.page);
  }, [items]);

  const applyResult = useCallback((r: AskResponse, fresh: boolean) => {
    const before = new Map(itemsRef.current.map((e, i) => [e.id, i]));
    setRes(r);
    setSessionId(r.session_id);
    setItems(r.evidence);
    setExhausted(false);
    if (fresh) {
      setCiteIds(Object.fromEntries(r.evidence.filter((e) => r.answer.citations.includes(e.eid)).map((e) => [e.eid, e.id])));
      setMoved({});
      setAnswerVote(0);
      setStale(false);
      const first = r.evidence.find((e) => r.answer.citations.includes(e.eid)) ?? r.evidence[0];
      setSelectedId(first?.id);
      if (first) setPage(first.page);
    } else {
      const m: Record<string, number> = {};
      r.evidence.forEach((e, i) => { const b = before.get(e.id); m[e.id] = b === undefined ? 0 : b - i; });
      setMoved(m);
      setStale(true);
    }
  }, []);

  const ask = useCallback(async (q = query, keepSession = true) => {
    if (!docId || !q.trim()) return;
    setBusy("ask"); setError("");
    try {
      const r = await api.ask({ doc_id: docId, query: q.trim(), session_id: keepSession ? sessionId : undefined, top_k: Number(params.get("k") ?? 5), memory });
      applyResult(r, true);
      return r;
    } catch (e) { setError(String(e)); }
    finally { setBusy(""); }
  }, [docId, query, sessionId, memory, applyResult]);

  const vote = useCallback(async (id: string, v: 1 | -1, base?: AskResponse) => {
    const r0 = base ?? res;
    if (!r0) return;
    setBusy("vote"); setError("");
    try {
      const r = await api.vote({ session_id: r0.session_id, query_id: r0.query_id, component_id: id, vote: v });
      applyResult(r, false);
      if (v < 0 && id === selectedId) setSelectedId(r.evidence[0]?.id);
      return r;
    } catch (e) { setError(String(e)); }
    finally { setBusy(""); }
  }, [res, applyResult, selectedId]);

  const showNext = useCallback(async () => {
    if (!res) return;
    setBusy("next");
    try {
      const n = await api.next(res.session_id, res.query_id);
      if (n.item) { setItems((xs) => [...xs, n.item!]); setSelectedId(n.item.id); setPage(n.item.page); }
      else setExhausted(true);
    } catch (e) { setError(String(e)); }
    finally { setBusy(""); }
  }, [res]);

  // keyboard: J / K move the selection, ↑ / ↓ vote on it, N shows the next item
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      const t = ev.target as HTMLElement;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
      if (!items.length || busy) return;
      const i = Math.max(0, items.findIndex((e) => e.id === selectedId));
      if (ev.key === "j" || ev.key === "J") select(items[Math.min(items.length - 1, i + 1)].id);
      else if (ev.key === "k" || ev.key === "K") select(items[Math.max(0, i - 1)].id);
      else if (ev.key === "ArrowUp" && selectedId) { ev.preventDefault(); vote(selectedId, 1); }
      else if (ev.key === "ArrowDown" && selectedId) { ev.preventDefault(); vote(selectedId, -1); }
      else if (ev.key === "n" || ev.key === "N") showNext();
      else return;
      document.querySelector(`[data-flip-id="${CSS.escape(items[i]?.id ?? "")}"]`)?.scrollIntoView({ block: "nearest" });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [items, selectedId, busy, select, vote, showNext]);

  // ?doc=&q=&auto=down:<id>,up:<id>,next,reanswer&sel=<id> : scripted state for screenshots and demos
  useEffect(() => {
    if (autoRan.current || !doc || !params.get("q")) return;
    autoRan.current = true;
    (async () => {
      let r = await ask(params.get("q")!, false);
      for (const step of (params.get("auto") ?? "").split(",").filter(Boolean)) {
        if (!r) break;
        const [kind, id] = step.split(":");
        if (kind === "next") { await showNext(); continue; }
        if (kind === "reanswer") {  // same session: the votes so far shape the evidence of the new answer
          r = await api.ask({ doc_id: r.doc_id, query: r.query, session_id: r.session_id, top_k: Number(params.get("k") ?? 5), memory });
          applyResult(r, true);
          continue;
        }
        r = (await vote(id, kind === "up" ? 1 : -1, r)) ?? r;
      }
      const sel = params.get("sel");
      if (sel) setTimeout(() => select(sel), 0);
    })();
  }, [doc]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { if (selectedId) select(selectedId); }, [items]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="app">
      <header className="masthead">
        <h1>DocScout<span>-X</span></h1>
        <p className="tagline">evidence, not documents</p>
        <span className="spacer" />
        <label className="docpick">
          <span>document</span>
          <select value={docId} onChange={(e) => { setDocId(e.target.value); setRes(null); setItems([]); setPage(1); }}>
            {docs.map((d) => (
              <option key={d.doc_id} value={d.doc_id}>{d.doc_id} · {d.store} · {d.pages.length} p</option>
            ))}
          </select>
        </label>
        <UploadButton onDone={refreshDocs} />
        <label className="toggle" title="cross-session memory: small capped prior from similar past questions">
          <input type="checkbox" checked={memory} onChange={(e) => setMemory(e.target.checked)} />
          <span className="knob" /> memory
        </label>
        <button className="session" onClick={() => { setSessionId(undefined); setRes(null); setItems([]); }}
                title="start a new anonymous session">
          session {sessionId ? sessionId.slice(0, 6) : "new"}
        </button>
      </header>

      <form className="askbar" onSubmit={(e) => { e.preventDefault(); ask(); }}>
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Ask about this document…" aria-label="question" />
        <button type="submit" disabled={!!busy || !query.trim()}>{busy === "ask" ? "Reading…" : "Ask"}</button>
      </form>
      {doc && doc.sample_questions.length > 0 && !res && (
        <div className="samples">
          {doc.sample_questions.map((q) => (
            <button key={q} onClick={() => { setQuery(q); ask(q); }}>{q}</button>
          ))}
        </div>
      )}
      {error && <div className="error" role="alert">{error}</div>}

      <main className="split">
        <div className="left">
          {res ? (
            <>
              <AnswerPanel res={res} stale={stale} answerVote={answerVote} busy={!!busy}
                           onCite={(eid) => select(citeIds[eid])}
                           onAnswerVote={async (v) => {
                             await api.voteAnswer({ session_id: res.session_id, query_id: res.query_id, vote: v });
                             setAnswerVote(v);
                           }}
                           onReanswer={() => ask(res.query, true)} />
              <div className="list-head">
                <h2>Evidence</h2>
                <span className="muted">{res.ranking_size} ranked components</span>
                <span className="spacer" />
                <span className="keys"><kbd>J</kbd><kbd>K</kbd> move <kbd>↑</kbd><kbd>↓</kbd> vote <kbd>N</kbd> next</span>
              </div>
              <div className="list" ref={listRef}>
                {items.map((e, i) => (
                  <EvidenceCard key={e.id} ev={e} index={i} selected={e.id === selectedId} vote={votes[e.id] ?? 0}
                                moved={moved[e.id]} cited={cited.has(e.id)} busy={!!busy}
                                onSelect={() => select(e.id)} onVote={(v) => vote(e.id, v)} />
                ))}
              </div>
              <button className="next" onClick={showNext} disabled={!!busy || exhausted}>
                {exhausted ? "No further matches" : busy === "next" ? "…" : "Show next"}
              </button>
            </>
          ) : (
            <div className="empty">
              <p>Ask a question. DocScout-X returns the exact paragraph, table cells or figure that answers it,
                with where it came from and why it was retrieved.</p>
              <p className="muted">Vote on evidence to re-rank it for this question (Rocchio feedback).</p>
            </div>
          )}
        </div>
        <div className="right">
          {doc && <PageViewer doc={doc} page={page} evidence={items} selected={selected} onPage={setPage} onSelect={select} />}
        </div>
      </main>
    </div>
  );
}
