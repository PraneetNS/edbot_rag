"""
student_tracker.py — Non-blocking analytics tap for every EduMentor turn.

Called via asyncio.create_task() so it never adds latency to the main pipeline.
"""
import asyncio
import logging

logger = logging.getLogger(__name__)

# ── Topic keyword map ─────────────────────────────────────────────────────────
TOPIC_MAP: dict[str, list[str]] = {
    "dsa": [
        "dsa", "array", "string", "tree", "graph", "dp",
        "recursion", "leetcode", "algorithm", "complexity",
        "binary", "heap", "stack", "queue", "sorting", "hash",
    ],
    "placement": [
        "placement", "interview", "amazon", "google", "tcs",
        "infosys", "campus", "oa", "online test", "round",
    ],
    "internship": [
        "internship", "intern", "ppo", "stipend", "cold email",
    ],
    "resume": [
        "resume", "cv", "project", "github", "portfolio",
    ],
    "career": [
        "career", "roadmap", "backend", "frontend", "ml",
        "machine learning", "devops", "full stack", "domain",
    ],
    "mindset": [
        "burnout", "behind", "struggling", "lost", "scared",
        "anxious", "motivation", "quit", "give up", "imposter",
    ],
    "higher_studies": [
        "ms", "gate", "gre", "masters", "abroad", "phd",
    ],
    "cs_concepts": [
        "react", "python", "javascript", "sql", "database",
        "api", "docker", "git", "linux", "networking", "os",
    ],
    "general": [],   # fallback — no keywords needed
}


def classify_topic(query: str) -> str:
    """Return the best matching topic for the query string."""
    q = query.lower()
    scores: dict[str, int] = {
        topic: sum(1 for kw in kws if kw in q)
        for topic, kws in TOPIC_MAP.items()
        if kws  # skip "general" which has empty keyword list
    }
    if not scores:
        return "general"
    best = max(scores, key=scores.get)
    return best if scores.get(best, 0) > 0 else "general"


# ── Main async tracker ────────────────────────────────────────────────────────
async def track_turn(
    student_id: str,
    session_id: str,
    query: str,
    response: str,
    guard_fired: str | None,
    timing: dict,
    retrieval_info: dict,
    is_vague: bool,
    is_repeat: bool,
    stuck_loop: bool,
) -> None:
    """
    Persist one analytics turn to SQLite.

    Designed to run as asyncio.create_task() — any exception is caught and
    logged so the main response pipeline is never affected.
    """
    try:
        # Run the synchronous DB calls in the default thread-pool executor
        # to avoid blocking the event loop.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _sync_track_turn, {
            "student_id":     student_id,
            "session_id":     session_id,
            "query":          query,
            "response":       response,
            "guard_fired":    guard_fired,
            "timing":         timing,
            "retrieval_info": retrieval_info,
            "is_vague":       is_vague,
            "is_repeat":      is_repeat,
            "stuck_loop":     stuck_loop,
        })
    except Exception as exc:
        logger.error(f"[StudentTracker] track_turn failed: {exc}")


def _sync_track_turn(data: dict) -> None:
    """Synchronous DB write — called via executor from track_turn."""
    from db import insert_turn, update_session

    student_id = data["student_id"]
    session_id = data["session_id"]
    query      = data["query"]
    response   = data["response"]
    timing     = data.get("timing", {})
    ret_info   = data.get("retrieval_info", {})

    topic = classify_topic(query)

    turn_data = {
        "student_id":      student_id,
        "session_id":      session_id,
        "query_text":      query,
        "query_words":     len(query.split()),
        "topic":           topic,
        "guard_fired":     data.get("guard_fired"),
        "response_text":   response,
        "response_words":  len(response.split()),
        "retrieval_ms":    timing.get("retrieval_ms", 0),
        "llm_ms":          timing.get("llm_ms", 0),
        "tts_ms":          timing.get("tts_ms", 0),
        "total_ms":        timing.get("total_ms", 0),
        "is_vague":        int(data.get("is_vague", False)),
        "is_repeat":       int(data.get("is_repeat", False)),
        "stuck_loop":      int(data.get("stuck_loop", False)),
        "chunks_found":    ret_info.get("chunks_found", 0),
        "top_chunk_score": ret_info.get("top_score"),
    }

    insert_turn(turn_data)
    update_session(session_id, student_id, topic)
    logger.debug(
        f"[StudentTracker] Saved turn — student={student_id[:8]} "
        f"topic={topic} guard={data.get('guard_fired')}"
    )
