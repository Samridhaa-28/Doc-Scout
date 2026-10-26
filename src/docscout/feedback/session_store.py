"""SQLite store of relevance votes: one table `votes`, used by session Rocchio and the cross-session memory.

A row is one vote: target 'evidence' (component_id set, used for ranking) or 'answer' (component_id NULL,
logged only, never read for ranking). The question's BGE vector is stored with the vote so the memory
can compare later questions to it without re-encoding.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS votes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    query TEXT NOT NULL,
    qtype TEXT NOT NULL,
    component_id TEXT,
    target TEXT NOT NULL DEFAULT 'evidence',
    vote INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    query_vec BLOB
);
CREATE INDEX IF NOT EXISTS votes_doc ON votes (doc_id, qtype);
CREATE INDEX IF NOT EXISTS votes_session ON votes (session_id, doc_id);
"""


def norm_query(q: str) -> str:
    """Session Rocchio applies to the same question: compare lower-cased, whitespace-collapsed text."""
    return " ".join(q.lower().split())


@dataclass
class Vote:
    session_id: str
    doc_id: str
    query: str
    qtype: str
    component_id: Optional[str]
    target: str
    vote: int
    timestamp: float
    query_vec: Optional[np.ndarray]


class FeedbackStore:
    """Thread-safe wrapper around one SQLite file (or ':memory:' for tests and simulations)."""

    def __init__(self, path: Union[str, Path] = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self._lock = threading.Lock()
        self._con = sqlite3.connect(self.path, check_same_thread=False)
        self._con.executescript(SCHEMA)

    def close(self) -> None:
        self._con.close()

    def record(self, session_id: str, doc_id: str, query: str, qtype: str, component_id: Optional[str], vote: int,
               target: str = "evidence", query_vec: Optional[np.ndarray] = None,
               timestamp: Optional[float] = None) -> int:
        """Insert one vote (+1 relevant / -1 not relevant); returns the row id."""
        if vote not in (1, -1):
            raise ValueError("vote must be +1 or -1")
        if target not in ("evidence", "answer"):
            raise ValueError("target must be 'evidence' or 'answer'")
        if target == "evidence" and not component_id:
            raise ValueError("an evidence vote needs a component_id")
        blob = None if query_vec is None else np.asarray(query_vec, dtype=np.float32).tobytes()
        with self._lock:
            cur = self._con.execute(
                "INSERT INTO votes (session_id, doc_id, query, qtype, component_id, target, vote, timestamp, query_vec)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (session_id, doc_id, query, qtype, component_id if target == "evidence" else None, target, int(vote),
                 float(timestamp if timestamp is not None else time.time()), blob))
            self._con.commit()
            return int(cur.lastrowid)

    def _rows(self, sql: str, args: Tuple) -> List[Vote]:
        with self._lock:
            rows = self._con.execute(sql, args).fetchall()
        return [Vote(r[0], r[1], r[2], r[3], r[4], r[5], int(r[6]), float(r[7]),
                     None if r[8] is None else np.frombuffer(r[8], dtype=np.float32)) for r in rows]

    _COLS = "session_id, doc_id, query, qtype, component_id, target, vote, timestamp, query_vec"

    def session_votes(self, session_id: str, doc_id: str, query: str) -> Dict[str, int]:
        """{component_id: vote} for this session, document and question; the latest vote per component wins."""
        rows = self._rows(f"SELECT {self._COLS} FROM votes WHERE session_id=? AND doc_id=? AND target='evidence'"
                          " ORDER BY timestamp, id", (session_id, doc_id))
        nq = norm_query(query)
        out: Dict[str, int] = {}
        for v in rows:
            if norm_query(v.query) == nq and v.component_id:
                out[v.component_id] = v.vote
        return out

    def doc_votes(self, doc_id: str, qtype: str, exclude_session: Optional[str] = None) -> List[Vote]:
        """Evidence votes on a document for one query type (memory candidates), optionally without one session."""
        sql = f"SELECT {self._COLS} FROM votes WHERE doc_id=? AND qtype=? AND target='evidence'"
        args: Tuple = (doc_id, qtype)
        if exclude_session is not None:
            sql += " AND session_id<>?"
            args += (exclude_session,)
        return self._rows(sql + " ORDER BY timestamp, id", args)

    def count(self, target: Optional[str] = None) -> int:
        with self._lock:
            if target is None:
                return int(self._con.execute("SELECT COUNT(*) FROM votes").fetchone()[0])
            return int(self._con.execute("SELECT COUNT(*) FROM votes WHERE target=?", (target,)).fetchone()[0])
