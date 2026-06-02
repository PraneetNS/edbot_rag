import sqlite3
import json
import logging
from pathlib import Path
from ai_core.config import WORKSPACE_DIR

logger = logging.getLogger(__name__)

DB_PATH = WORKSPACE_DIR / "chroma_db" / "session_memory.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_memory (
            session_id TEXT PRIMARY KEY,
            active_topic TEXT,
            active_intent TEXT,
            workflow TEXT,
            mode TEXT,
            last_courses TEXT,
            active_goal TEXT,
            target_domain_val TEXT,
            target_domain_conf TEXT,
            weak_subject_val TEXT,
            weak_subject_conf TEXT
        )
    """)
    conn.commit()
    conn.close()

class SqliteMemory:
    """Persistent SQLite-backed store for user session states."""
    def __init__(self):
        init_db()

    def get_conn(self):
        return sqlite3.connect(str(DB_PATH))

    def load_session(self, session_id: str) -> dict:
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM session_memory WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "session_id": row[0],
                "active_topic": row[1],
                "active_intent": row[2],
                "workflow": row[3],
                "mode": row[4],
                "last_courses_discussed": json.loads(row[5] or "[]"),
                "active_goal": row[6],
                "memory_profile": {
                    "target_domain": {"value": row[7], "confidence": row[8]},
                    "weak_subject": {"value": row[9], "confidence": row[10]}
                }
            }
        return None

    def save_session(self, session_id: str, state: dict):
        conn = self.get_conn()
        cursor = conn.cursor()
        
        profile = state.get("memory_profile", {})
        td = profile.get("target_domain", {"value": "Computer Science", "confidence": "high"})
        ws = profile.get("weak_subject", {"value": "None detected", "confidence": "medium"})
        
        cursor.execute("""
            INSERT INTO session_memory (
                session_id, active_topic, active_intent, workflow, mode, 
                last_courses, active_goal, target_domain_val, target_domain_conf, 
                weak_subject_val, weak_subject_conf
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                active_topic=excluded.active_topic,
                active_intent=excluded.active_intent,
                workflow=excluded.workflow,
                mode=excluded.mode,
                last_courses=excluded.last_courses,
                active_goal=excluded.active_goal,
                target_domain_val=excluded.target_domain_val,
                target_domain_conf=excluded.target_domain_conf,
                weak_subject_val=excluded.weak_subject_val,
                weak_subject_conf=excluded.weak_subject_conf
        """, (
            session_id,
            state.get("active_topic", "general"),
            state.get("active_intent", "COURSE_QUERY"),
            state.get("workflow", "general"),
            state.get("mode", "academic_mentor"),
            json.dumps(state.get("last_courses_discussed", [])),
            state.get("active_goal", "Explore courses & placements"),
            td.get("value", "Computer Science"),
            td.get("confidence", "high"),
            ws.get("value", "None detected"),
            ws.get("confidence", "medium")
        ))
        conn.commit()
        conn.close()
