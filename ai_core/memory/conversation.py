import re
import logging
from ai_core.memory.sqlite_memory import SqliteMemory

logger = logging.getLogger(__name__)

class ConversationResolver:
    """
    Manages conversational memory, pronoun resolution, and state updates.
    Attempts to resolve queries like 'give projects using it' to 'give projects using dynamic programming' in-memory,
    skipping LLM rewrites for latency optimization.
    """
    def __init__(self):
        self.db = SqliteMemory()

    def get_session_state(self, session_id: str) -> dict:
        state = self.db.load_session(session_id)
        if not state:
            state = {
                "session_id": session_id,
                "active_topic": "general",
                "active_intent": "COURSE_QUERY",
                "workflow": "general",
                "mode": "academic_mentor",
                "last_courses_discussed": [],
                "active_goal": "Explore courses & placements",
                "memory_profile": {
                    "target_domain": {"value": "Computer Science", "confidence": "high"},
                    "weak_subject": {"value": "None detected", "confidence": "medium"}
                }
            }
            self.db.save_session(session_id, state)
        return state

    def update_session_state(self, session_id: str, question: str, intent: str):
        state = self.get_session_state(session_id)
        q = question.lower().strip()
        
        # 1. Update Topic & Mode & Intent
        if any(k in q for k in ["placement", "job", "career", "interview", "resume", "recruit"]):
            state["active_topic"] = "placements"
            state["active_intent"] = "PLACEMENT_GUIDANCE"
            state["mode"] = "placement_coach"
            state["active_goal"] = "Prepare for placement drives & resume review"
            if "resume" in q:
                state["memory_profile"]["target_domain"] = {"value": "Resume Building", "confidence": "high"}
            else:
                state["memory_profile"]["target_domain"] = {"value": "Career Prep", "confidence": "high"}
                
        elif any(k in q for k in ["vtu", "certif", "stamp", "verify"]):
            state["active_topic"] = "certifications"
            state["active_intent"] = "COURSE_QUERY"
            state["mode"] = "academic_mentor"
            state["active_goal"] = "VTU Certification Stamp Verification"
            state["memory_profile"]["target_domain"] = {"value": "VTU Certifications", "confidence": "high"}
            
        elif any(k in q for k in ["support", "help", "ticket", "login", "portal", "contact", "mail"]):
            state["active_topic"] = "support"
            state["active_intent"] = "COURSE_QUERY"
            state["mode"] = "support_assistant"
            state["active_goal"] = "LMS Portal & Ticket Support"
            state["memory_profile"]["target_domain"] = {"value": "LMS Technical Support", "confidence": "high"}
            
        elif any(k in q for k in ["course", "syllabus", "react", "python", "java", "dsa", "programming", "code", "database", "sql"]):
            state["active_topic"] = "courses"
            state["active_intent"] = "COURSE_QUERY"
            state["mode"] = "academic_mentor"
            state["active_goal"] = "Master engineering curriculum & concepts"
            
            # Detect target domain from keywords
            if "react" in q or "frontend" in q or "web" in q:
                state["memory_profile"]["target_domain"] = {"value": "React / Web Dev", "confidence": "high"}
                if "React JS" not in state["last_courses_discussed"]:
                    state["last_courses_discussed"].append("React JS")
            elif "python" in q:
                state["memory_profile"]["target_domain"] = {"value": "Python Development", "confidence": "high"}
                if "Python" not in state["last_courses_discussed"]:
                    state["last_courses_discussed"].append("Python")
            elif "java" in q:
                state["memory_profile"]["target_domain"] = {"value": "Java Programming", "confidence": "high"}
                if "Java" not in state["last_courses_discussed"]:
                    state["last_courses_discussed"].append("Java")
            elif "dsa" in q or "algorithm" in q or "tree" in q or "search" in q:
                state["memory_profile"]["target_domain"] = {"value": "Data Structures & Algos", "confidence": "high"}
                if "DSA" not in state["last_courses_discussed"]:
                    state["last_courses_discussed"].append("DSA")
            elif "sql" in q or "database" in q or "db" in q:
                state["memory_profile"]["target_domain"] = {"value": "Database Management", "confidence": "high"}
                if "SQL/DBMS" not in state["last_courses_discussed"]:
                    state["last_courses_discussed"].append("SQL/DBMS")
                    
        # Update dynamic active topic if a direct academic subject is asked
        subject_matches = re.findall(r'\b(react|python|java|dsa|sql|database|dynamic programming|recursion|time complexity)\b', q)
        if subject_matches:
            state["active_topic"] = subject_matches[-1]
                    
        # Detect weak subject
        if any(k in q for k in ["confused", "stuck", "difficult", "hard", "help with", "explain", "don't understand"]):
            if "dsa" in q or "algorithm" in q:
                state["memory_profile"]["weak_subject"] = {"value": "Data Structures", "confidence": "high"}
            elif "programming" in q or "coding" in q:
                state["memory_profile"]["weak_subject"] = {"value": "Programming Basics", "confidence": "medium"}
            elif "sql" in q or "database" in q:
                state["memory_profile"]["weak_subject"] = {"value": "SQL Queries", "confidence": "high"}
                
        self.db.save_session(session_id, state)

    def needs_rewrite(self, query: str) -> bool:
        vague_terms = ["it", "that", "this", "those", "above", "previous", "earlier", "before"]
        words = re.findall(r'\b\w+\b', query.lower())
        return any(x in words for x in vague_terms)

    def resolve_pronouns(self, session_id: str, query: str) -> str:
        """Attempts cheap regex-based pronoun resolution in-memory using active topic."""
        if not self.needs_rewrite(query):
            return query

        state = self.get_session_state(session_id)
        topic = state.get("active_topic", "general")
        
        if topic == "general":
            return query
            
        # Compile pronoun resolver regex
        pronouns_regex = re.compile(r'\b(it|that|this|those|above)\b', re.IGNORECASE)
        resolved = pronouns_regex.sub(topic, query)
        
        logger.info(f"In-Memory resolved query: '{query}' -> '{resolved}'")
        return resolved
