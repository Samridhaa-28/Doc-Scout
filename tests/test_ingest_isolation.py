"""Crash isolation in docscout.ingestion.runner: a worker that dies is detected, restarted, and the next document still runs."""
import os

from docscout.ingestion.runner import Worker


def fake_worker(source, backend, task_q, result_q):
    """Module-level so the spawned child can import it. 'crash' kills the process like a native fault."""
    result_q.put(("ready", None, None))
    while True:
        task = task_q.get()
        if task is None:
            return
        doc_id, _ = task
        if doc_id == "crash":
            os._exit(3)
        if doc_id == "boom":
            result_q.put(("error", doc_id, "Traceback ... ValueError: boom"))
            continue
        result_q.put(("ok", doc_id, ([], {"status": "success", "pdf_backend": backend}, 0.0)))


def test_worker_survives_child_crash():
    w = Worker("mpdocvqa", "pypdfium2", target=fake_worker)
    try:
        kind, payload = w.run("doc1", None, timeout_s=60)
        assert kind == "ok" and payload[1]["pdf_backend"] == "pypdfium2"
        first_pid = w.proc.pid
        kind, payload = w.run("crash", None, timeout_s=60)
        assert (kind, payload) == ("crash", 3)
        assert w.proc.pid != first_pid and w.proc.is_alive()  # restarted
        kind, _ = w.run("boom", None, timeout_s=60)
        assert kind == "error"
        kind, payload = w.run("doc2", None, timeout_s=60)  # still works after crash and error
        assert kind == "ok"
    finally:
        w.close()
