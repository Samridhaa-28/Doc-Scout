import { useEffect, useRef, useState } from "react";
import { api, type IngestStatus } from "../api";

/** "Upload PDF" next to the document dropdown: POST /ingest, poll /ingest/status, then ask the parent
 *  to refresh the document list. It never changes the dropdown's selection. */
export default function UploadButton({ onDone }: { onDone: (docId: string) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [job, setJob] = useState<IngestStatus | null>(null);
  const [error, setError] = useState("");
  const busy = job?.status === "queued" || job?.status === "running";

  useEffect(() => {
    if (!busy) return;
    const t = window.setInterval(async () => {
      try {
        const s = await api.ingestStatus();
        setJob(s);
        if (s.status === "done" && s.doc_id) onDone(s.doc_id);
      } catch (e) { setError(String(e)); }
    }, 3000);
    return () => window.clearInterval(t);
  }, [busy, onDone]);

  const pick = async (f: File | undefined) => {
    if (!f) return;
    setError("");
    try {
      const r = await api.upload(f);
      setJob({ status: "queued", file: f.name, doc_id: r.doc_id, error: null, log_tail: [], elapsed_s: 0 });
    } catch (e) { setError(String(e)); }
    if (input.current) input.current.value = "";
  };

  const last = job?.log_tail?.[job.log_tail.length - 1];
  return (
    <span className="upload">
      <input ref={input} type="file" accept=".pdf,application/pdf" hidden onChange={(e) => pick(e.target.files?.[0])} />
      <button onClick={() => input.current?.click()} disabled={busy} title="add your own PDF to the demo store">
        {busy ? "Processing… (can take several minutes)" : "Upload PDF"}
      </button>
      {busy && last && <span className="upload-log" title={job?.log_tail.join("\n")}>{last.slice(0, 70)}</span>}
      {job?.status === "done" && <span className="upload-ok">added {job.doc_id}: pick it in the list</span>}
      {(job?.status === "error" || error) && (
        <span className="upload-err" title={job?.log_tail.join("\n")}>{error || job?.error}</span>
      )}
    </span>
  );
}
