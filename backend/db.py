"""
db.py — EduMentor SQLite analytics layer.

Creates edumentor.db in the same directory as this file (backend/).
All tables are created on first startup via init_db().
Thread-safe: uses WAL journal mode + check_same_thread=False.
"""
import sqlite3
import os
import uuid
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── DB path: backend/edumentor.db ─────────────────────────────────────────────
_DB_PATH = Path(__file__).resolve().parent / "edumentor.db"


def _get_conn() -> sqlite3.Connection:
    """Open a thread-safe connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── Schema Initialisation ─────────────────────────────────────────────────────
def init_db() -> None:
    """Create all tables if they don't exist. Safe to call multiple times."""
    ddl = """
    CREATE TABLE IF NOT EXISTS students (
        student_id    TEXT PRIMARY KEY,
        username      TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        year          TEXT,
        created_at    TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS turns (
        turn_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id        TEXT NOT NULL,
        session_id        TEXT NOT NULL,
        timestamp         TEXT DEFAULT (datetime('now')),
        query_text        TEXT NOT NULL,
        query_words       INTEGER,
        topic             TEXT,
        guard_fired       TEXT,
        response_text     TEXT,
        response_words    INTEGER,
        retrieval_ms      INTEGER,
        llm_ms            INTEGER,
        tts_ms            INTEGER,
        total_ms          INTEGER,
        is_vague          INTEGER DEFAULT 0,
        is_repeat         INTEGER DEFAULT 0,
        stuck_loop        INTEGER DEFAULT 0,
        chunks_found      INTEGER DEFAULT 0,
        top_chunk_score   REAL,
        FOREIGN KEY (student_id) REFERENCES students(student_id)
    );

    CREATE TABLE IF NOT EXISTS sessions (
        session_id   TEXT PRIMARY KEY,
        student_id   TEXT NOT NULL,
        started_at   TEXT DEFAULT (datetime('now')),
        last_active  TEXT DEFAULT (datetime('now')),
        turn_count   INTEGER DEFAULT 0,
        topics_hit   TEXT,
        FOREIGN KEY (student_id) REFERENCES students(student_id)
    );

    CREATE INDEX IF NOT EXISTS idx_turns_student  ON turns(student_id);
    CREATE INDEX IF NOT EXISTS idx_turns_session  ON turns(session_id);
    CREATE INDEX IF NOT EXISTS idx_turns_topic    ON turns(topic);
    CREATE INDEX IF NOT EXISTS idx_sessions_student ON sessions(student_id);
    """
    with _get_conn() as conn:
        conn.executescript(ddl)
    logger.info(f"[DB] edumentor.db initialised at {_DB_PATH}")


# ── Student helpers ───────────────────────────────────────────────────────────
def create_student(username: str, password_hash: str, year: Optional[str] = None) -> dict:
    """Insert a new student. Returns the created student dict."""
    student_id = str(uuid.uuid4())
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO students (student_id, username, password_hash, year) VALUES (?,?,?,?)",
            (student_id, username, password_hash, year),
        )
    return {"student_id": student_id, "username": username, "year": year}


def get_student_by_username(username: str) -> Optional[dict]:
    """Fetch a student row by username, or None if not found."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM students WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def get_student_by_id(student_id: str) -> Optional[dict]:
    """Fetch a student row by student_id, or None if not found."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
    return dict(row) if row else None


# ── Turn helpers ──────────────────────────────────────────────────────────────
def insert_turn(turn_data: dict) -> None:
    """Insert one analytics turn. Unknown keys are silently ignored."""
    cols = [
        "student_id", "session_id", "query_text", "query_words",
        "topic", "guard_fired", "response_text", "response_words",
        "retrieval_ms", "llm_ms", "tts_ms", "total_ms",
        "is_vague", "is_repeat", "stuck_loop", "chunks_found", "top_chunk_score",
    ]
    keys = [c for c in cols if c in turn_data]
    placeholders = ",".join("?" * len(keys))
    values = [turn_data[k] for k in keys]
    sql = f"INSERT INTO turns ({','.join(keys)}) VALUES ({placeholders})"
    try:
        with _get_conn() as conn:
            conn.execute(sql, values)
    except Exception as exc:
        logger.error(f"[DB] insert_turn failed: {exc}")


# ── Session helpers ───────────────────────────────────────────────────────────
def update_session(session_id: str, student_id: str, topic: str) -> None:
    """Upsert the sessions row and increment turn_count."""
    try:
        with _get_conn() as conn:
            existing = conn.execute(
                "SELECT session_id, topics_hit FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            if existing:
                # Update existing session
                topics_set: set = set(
                    (existing["topics_hit"] or "").split(",")
                ) - {""}
                topics_set.add(topic)
                conn.execute(
                    """UPDATE sessions
                       SET last_active = datetime('now'),
                           turn_count  = turn_count + 1,
                           topics_hit  = ?
                       WHERE session_id = ?""",
                    (",".join(sorted(topics_set)), session_id),
                )
            else:
                # New session
                conn.execute(
                    """INSERT INTO sessions
                       (session_id, student_id, turn_count, topics_hit)
                       VALUES (?, ?, 1, ?)""",
                    (session_id, student_id, topic),
                )
    except Exception as exc:
        logger.error(f"[DB] update_session failed: {exc}")


def get_session_turn_count(session_id: str) -> int:
    """Return current turn_count for a session (0 if session not found)."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT turn_count FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row["turn_count"] if row else 0
    except Exception as exc:
        logger.error(f"[DB] get_session_turn_count failed: {exc}")
        return 0


# ── Analytics helpers ─────────────────────────────────────────────────────────
def get_student_stats(student_id: str) -> dict:
    """Return an aggregated stats dict for the dashboard /stats endpoint."""
    try:
        with _get_conn() as conn:
            # Basic counts
            row = conn.execute(
                """SELECT
                       COUNT(DISTINCT session_id) AS total_sessions,
                       COUNT(*)                   AS total_turns,
                       AVG(response_words)        AS avg_response_words,
                       AVG(total_ms)              AS avg_total_latency_ms,
                       SUM(is_vague)              AS vague_count,
                       SUM(is_repeat)             AS repeat_count,
                       MIN(timestamp)             AS first_session,
                       MAX(timestamp)             AS last_session
                   FROM turns
                   WHERE student_id = ?""",
                (student_id,),
            ).fetchone()

            totals = dict(row) if row else {}
            total_turns = totals.get("total_turns") or 0
            vague_count = totals.get("vague_count") or 0
            repeat_count = totals.get("repeat_count") or 0

            # Average session length
            sess_avg = conn.execute(
                """SELECT AVG(turn_count) AS avg_len
                   FROM sessions WHERE student_id = ?""",
                (student_id,),
            ).fetchone()

            # Topic breakdown
            topic_rows = conn.execute(
                """SELECT topic, COUNT(*) AS cnt
                   FROM turns WHERE student_id = ?
                   GROUP BY topic""",
                (student_id,),
            ).fetchall()
            topic_breakdown = {r["topic"]: r["cnt"] for r in topic_rows if r["topic"]}

            # Most asked topic
            most_asked = max(topic_breakdown, key=topic_breakdown.get) if topic_breakdown else None

            # Guard breakdown
            guard_rows = conn.execute(
                """SELECT COALESCE(guard_fired, 'llm_generation') AS gf, COUNT(*) AS cnt
                   FROM turns WHERE student_id = ?
                   GROUP BY gf""",
                (student_id,),
            ).fetchall()
            guard_breakdown = {r["gf"]: r["cnt"] for r in guard_rows}

            # Active days
            active_days_row = conn.execute(
                """SELECT COUNT(DISTINCT DATE(timestamp)) AS active_days
                   FROM turns WHERE student_id = ?""",
                (student_id,),
            ).fetchone()

        return {
            "total_sessions": totals.get("total_sessions") or 0,
            "total_turns": total_turns,
            "total_questions": total_turns,
            "avg_session_length_turns": round(
                (sess_avg["avg_len"] or 0) if sess_avg else 0, 2
            ),
            "topic_breakdown": topic_breakdown,
            "most_asked_topic": most_asked,
            "vague_query_rate": round(vague_count / total_turns, 4) if total_turns else 0.0,
            "repeat_query_rate": round(repeat_count / total_turns, 4) if total_turns else 0.0,
            "avg_response_words": round(totals.get("avg_response_words") or 0, 2),
            "avg_total_latency_ms": round(totals.get("avg_total_latency_ms") or 0, 2),
            "guard_breakdown": guard_breakdown,
            "active_days": (active_days_row["active_days"] if active_days_row else 0),
            "first_session": totals.get("first_session"),
            "last_session": totals.get("last_session"),
        }
    except Exception as exc:
        logger.error(f"[DB] get_student_stats failed: {exc}")
        return {}


def get_student_turns(student_id: str, limit: int = 100, offset: int = 0) -> list:
    """Return recent turns for the student, most recent first."""
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                """SELECT turn_id, timestamp, query_text, topic,
                          response_text, total_ms, guard_fired, is_vague, chunks_found
                   FROM turns WHERE student_id = ?
                   ORDER BY timestamp DESC
                   LIMIT ? OFFSET ?""",
                (student_id, limit, offset),
            ).fetchall()
        return [
            {
                "turn_id": r["turn_id"],
                "timestamp": r["timestamp"],
                "query_text": r["query_text"],
                "topic": r["topic"],
                "response_text": r["response_text"],
                "total_ms": r["total_ms"],
                "guard_fired": r["guard_fired"],
                "is_vague": bool(r["is_vague"]),
                "chunks_found": r["chunks_found"],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error(f"[DB] get_student_turns failed: {exc}")
        return []


def get_topic_breakdown(student_id: str) -> dict:
    """Return {topic: count} for all turns by the student."""
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                """SELECT topic, COUNT(*) AS cnt
                   FROM turns WHERE student_id = ?
                   GROUP BY topic""",
                (student_id,),
            ).fetchall()
        return {r["topic"]: r["cnt"] for r in rows if r["topic"]}
    except Exception as exc:
        logger.error(f"[DB] get_topic_breakdown failed: {exc}")
        return {}


def get_weak_areas(student_id: str) -> list:
    """
    Return topics where the student either:
      - repeats questions frequently (high is_repeat rate), or
      - consistently gets 0 chunks found (RAG knowledge gap).
    Ordered by question_count DESC.
    """
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                """SELECT
                       topic,
                       COUNT(*) AS question_count,
                       AVG(is_repeat) AS repeat_rate,
                       AVG(CASE WHEN chunks_found = 0 THEN 1.0 ELSE 0.0 END) AS chunks_found_rate
                   FROM turns
                   WHERE student_id = ? AND topic IS NOT NULL
                   GROUP BY topic
                   HAVING question_count >= 2
                   ORDER BY question_count DESC""",
                (student_id,),
            ).fetchall()

            result = []
            for r in rows:
                # Fetch up to 3 example questions for this topic
                ex_rows = conn.execute(
                    """SELECT query_text FROM turns
                       WHERE student_id = ? AND topic = ?
                       ORDER BY timestamp DESC LIMIT 3""",
                    (student_id, r["topic"]),
                ).fetchall()
                examples = [e["query_text"] for e in ex_rows]
                result.append(
                    {
                        "topic": r["topic"],
                        "question_count": r["question_count"],
                        "repeat_rate": round(r["repeat_rate"], 4),
                        "chunks_found_rate": round(r["chunks_found_rate"], 4),
                        "example_questions": examples,
                    }
                )
        return result
    except Exception as exc:
        logger.error(f"[DB] get_weak_areas failed: {exc}")
        return []
