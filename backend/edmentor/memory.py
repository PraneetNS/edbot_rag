"""
edmentor/memory.py
──────────────────
Conversation turn history for Edmentor.
Wraps ai_core/memory/sqlite_memory.py to store and retrieve
the last N (user, assistant) turn pairs per session.

This enables context continuity:
    Turn 1: "how do I solve binary trees?"
    Turn 2: "okay now do that for graphs"
    → Turn 2 has access to Turn 1 so the reference is resolved.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

# Store in the same chroma_db folder for consistency
_DB_PATH = Path(__file__).resolve().parent.parent.parent / "chroma_db" / "edmentor_turns.db"


class EdmentorMemory:
    """
    Lightweight SQLite-backed conversation turn store.
    Each row: (session_id, turn_index, user_msg, assistant_msg, timestamp)
    """

    def __init__(self, db_path: Path = _DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edmentor_turns (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT    NOT NULL,
                    user_msg    TEXT    NOT NULL,
                    assistant_msg TEXT  NOT NULL,
                    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session ON edmentor_turns(session_id, id)"
            )
            conn.commit()

    def save_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Append a completed turn (user + assistant) to history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO edmentor_turns (session_id, user_msg, assistant_msg) VALUES (?, ?, ?)",
                    (session_id, user_msg, assistant_msg),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"EdmentorMemory.save_turn error: {e}")

    def get_last_turns(self, session_id: str, n: int = 2) -> List[Dict[str, str]]:
        """
        Retrieve the last N turns for a session.

        Returns:
            List of dicts with keys 'user' and 'assistant', ordered oldest → newest.
            Example (n=2):
            [
                {"user": "how do I do binary trees?", "assistant": "Before you touch trees..."},
                {"user": "now do graphs", "assistant": "Graphs build on the same idea..."},
            ]
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT user_msg, assistant_msg
                    FROM edmentor_turns
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (session_id, n),
                )
                rows = cursor.fetchall()
            # Reverse so oldest turn comes first
            rows.reverse()
            return [{"user": r[0], "assistant": r[1]} for r in rows]
        except Exception as e:
            logger.error(f"EdmentorMemory.get_last_turns error: {e}")
            return []

    def clear_session(self, session_id: str) -> None:
        """Delete all turns for a session (called on /clear)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM edmentor_turns WHERE session_id = ?",
                    (session_id,),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"EdmentorMemory.clear_session error: {e}")


# Module-level singleton
memory = EdmentorMemory()
