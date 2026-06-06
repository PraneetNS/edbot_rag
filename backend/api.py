from pathlib import Path
import sys
import os
import time
import re
import asyncio
import logging
from typing import Generator

# Force HuggingFace to offline mode to avoid remote check delays/timeouts
os.environ["HF_HUB_OFFLINE"] = "1"

# Fix Windows cp1252 console encoding (crashes on ✓ ✗ chars without this)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

logger = logging.getLogger("api")


from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Allow imports from backend root
sys.path.append(str(Path(__file__).resolve().parent))

from rag.retrieval.retriever import (
    load_rag_index,
    get_edumentor_query_engine,
    load_rag_index_5k,
    get_edumentor_query_engine_5k,
    check_ollama_active,
    retrieve_chunks_for_edmentor,
)
from rag.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, CHROMA_COLLECTION_5K

# ── Edmentor voice-mentor components (lazy load) ──────────────────────────────
try:
    from edmentor.guard import guard as edmentor_guard
    from edmentor.voice_limit import enforce_voice_limit, speaking_duration_label
    from edmentor.topic_classifier import EdmentorTopicClassifier
    from edmentor.memory import memory as edmentor_memory
    from edmentor.prompt import build_messages
    from edmentor.groq_client import llm as edmentor_llm
    from edmentor.tts_router import tts_router
    
    # Imports for confidence routing architecture
    from edmentor.intent_router import is_off_domain
    from edmentor.confidence_router import generate_response_with_routing, USE_LOCAL_MODEL
    from edmentor.safety_filter import edumentor_filter
    
    _EDMENTOR_READY = True
except Exception as _e:
    logger_tmp = __import__('logging').getLogger('api')
    logger_tmp.warning(f"Edmentor components not fully loaded: {_e}")
    _EDMENTOR_READY = False
    tts_router = None

# Setup base paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
import logging
logging.basicConfig(
    filename=str(LOGS_DIR / "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RAG_Pipeline")

app = FastAPI(title="EduBot RAG API")

# Mount static files
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Register Edmentor TTS router if available
if tts_router is not None:
    app.include_router(tts_router)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_stale_sessions_periodically())

@app.middleware("http")
async def log_requests_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {process_time:.4f}s"
    )
    return response

# ── Load index and query engine once at startup ──────────────────────────────
print("Loading RAG indexes and query engines ...")

# Primary: ultra-premium dataset
try:
    index = load_rag_index()
    query_engine = get_edumentor_query_engine(index)
    print("  ✓ Primary (ultra-premium) EduBot RAG engine ready.")
except Exception as e:
    print(f"  ✗ Primary engine failed: {e}")
    index = None
    query_engine = None

# Secondary: synthetic 5k dataset
try:
    index_5k = load_rag_index_5k()
    query_engine_5k = get_edumentor_query_engine_5k(index_5k)
    print("  ✓ Edumentor 5k RAG engine ready.")
except Exception as e:
    print(f"  ✗ 5k engine not available (run build_index_5k.py first): {e}")
    index_5k = None
    query_engine_5k = None

# Edmentor topic classifier (shares MiniLM embed model — no extra memory cost)
edmentor_topic_classifier = None
if _EDMENTOR_READY:
    try:
        edmentor_topic_classifier = EdmentorTopicClassifier()
        print("  ✓ Edmentor topic classifier ready.")
    except Exception as e:
        print(f"  ✗ Edmentor topic classifier failed: {e}")

# ── Session Storage and State Management ─────────────────────────────────────
class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.active_topic = "general"
        self.active_intent = "COURSE_QUERY"
        self.workflow = "general"
        self.mode = "academic_mentor"
        self.last_courses_discussed = []
        self.active_goal = "Explore courses & placements"
        self.memory_profile = {
            "target_domain": {"value": "Computer Science", "confidence": "high"},
            "weak_subject": {"value": "None detected", "confidence": "medium"}
        }
        self.recommendations = []
        self.last_accessed = time.time()

sessions = {}

def get_or_create_session(session_id: str) -> SessionState:
    if not session_id:
        session_id = "default"
    
    session = sessions.get(session_id)
    if session is None:
        # Enforce LRU cache limit of 1000 sessions
        if len(sessions) >= 1000:
            # Find and evict the oldest session that is not "default"
            non_default = [(sid, s.last_accessed) for sid, s in sessions.items() if sid != "default"]
            if non_default:
                oldest_sid = min(non_default, key=lambda x: x[1])[0]
                sessions.pop(oldest_sid, None)
                logger.info(f"Evicted LRU session: {oldest_sid}")
        
        session = SessionState(session_id)
        session.recommendations = generate_recommendations("general")
        sessions[session_id] = session
    
    session.last_accessed = time.time()
    return session

async def cleanup_stale_sessions_periodically():
    """Background task to remove sessions inactive for more than 2 hours."""
    while True:
        await asyncio.sleep(1800)  # Check every 30 minutes
        try:
            now = time.time()
            stale_keys = [
                sid for sid, s in sessions.items()
                if sid != "default" and now - s.last_accessed > 7200
            ]
            for sid in stale_keys:
                sessions.pop(sid, None)
                logger.info(f"Cleaned up stale session: {sid}")
        except Exception as e:
            logger.error(f"Error in session cleanup task: {e}")

# ── Recommendations Generator ────────────────────────────────────────────────
def generate_recommendations(topic: str) -> list[dict]:
    if topic == "placements":
        return [
            {
                "badge": "Career Guide",
                "title": "Placement Prep Roadmap",
                "description": "Access the step-by-step roadmap to crack product-based companies.",
                "query_trigger": "Tell me about the placement preparation roadmap",
                "action_text": "View Roadmap"
            },
            {
                "badge": "Resume Review",
                "title": "Resume Mentoring",
                "description": "Learn the best tips to optimize your resume for ATS screening.",
                "query_trigger": "What are the best resume tips for engineering placements?",
                "action_text": "Get Resume Tips"
            }
        ]
    elif topic == "certifications":
        return [
            {
                "badge": "VTU Info",
                "title": "Certificate Stamp Verification",
                "description": "Learn about the double-stamp system on VTU co-branded certificates.",
                "query_trigger": "Tell me about the VTU certificate stamp",
                "action_text": "Verify Stamps"
            },
            {
                "badge": "Downloads",
                "title": "Download Guide",
                "description": "Instructions to download your verified certificate from the LMS portal.",
                "query_trigger": "How do I download my VTU certificate?",
                "action_text": "Download Guide"
            }
        ]
    elif topic == "support":
        return [
            {
                "badge": "LMS Help",
                "title": "Technical Support Contacts",
                "description": "Get email addresses and contact info for the tech support team.",
                "query_trigger": "How can I contact the academic support team?",
                "action_text": "Get Contacts"
            },
            {
                "badge": "Troubleshoot",
                "title": "Login Issues",
                "description": "Resolve issues related to credential mismatch or loading spinner.",
                "query_trigger": "I am having trouble with logging in.",
                "action_text": "Fix Login"
            }
        ]
    elif topic == "courses":
        return [
            {
                "badge": "React JS",
                "title": "React Course Syllabus",
                "description": "Check the modules, project assignments, and duration for React training.",
                "query_trigger": "What is the React JS course syllabus?",
                "action_text": "View React Syllabus"
            },
            {
                "badge": "Full Curriculum",
                "title": "Engineering Core Courses",
                "description": "Browse our catalog of CS concepts, placement prep, and programming tracks.",
                "query_trigger": "What courses are currently offered by Edutainer?",
                "action_text": "Browse Courses"
            }
        ]
    else:  # general
        return [
            {
                "badge": "Welcome",
                "title": "Explore Edutainer",
                "description": "Find out what courses are currently offered by Edutainer.",
                "query_trigger": "What courses are currently offered by Edutainer?",
                "action_text": "Explore Courses"
            },
            {
                "badge": "Placements",
                "title": "Placement Assistance",
                "description": "Learn how Edutainer helps you prepare for placements and internships.",
                "query_trigger": "Can you help me prepare for placements?",
                "action_text": "Get Placement Help"
            }
        ]

# ── Dynamic Session State Updater ───────────────────────────────────────────
def update_session_state(session: SessionState, question: str):
    q = question.lower().strip()
    
    # 1. Update Topic & Mode & Intent
    if any(k in q for k in ["placement", "job", "career", "interview", "resume", "recruit"]):
        session.active_topic = "placements"
        session.active_intent = "PLACEMENT_GUIDANCE"
        session.mode = "placement_coach"
        session.active_goal = "Prepare for placement drives & resume review"
        if "resume" in q:
            session.memory_profile["target_domain"] = {"value": "Resume Building", "confidence": "high"}
        else:
            session.memory_profile["target_domain"] = {"value": "Career Prep", "confidence": "high"}
            
    elif any(k in q for k in ["vtu", "certif", "stamp", "verify"]):
        session.active_topic = "certifications"
        session.active_intent = "COURSE_QUERY"
        session.mode = "academic_mentor"
        session.active_goal = "VTU Certification Stamp Verification"
        session.memory_profile["target_domain"] = {"value": "VTU Certifications", "confidence": "high"}
        
    elif any(k in q for k in ["support", "help", "ticket", "login", "portal", "contact", "mail"]):
        session.active_topic = "support"
        session.active_intent = "COURSE_QUERY"
        session.mode = "support_assistant"
        session.active_goal = "LMS Portal & Ticket Support"
        session.memory_profile["target_domain"] = {"value": "LMS Technical Support", "confidence": "high"}
        
    elif any(k in q for k in ["course", "syllabus", "react", "python", "java", "dsa", "programming", "code", "database", "sql"]):
        session.active_topic = "courses"
        session.active_intent = "COURSE_QUERY"
        session.mode = "academic_mentor"
        session.active_goal = "Master engineering curriculum & concepts"
        
        # Try to detect target domain from keywords
        if "react" in q or "frontend" in q or "web" in q:
            session.memory_profile["target_domain"] = {"value": "React / Web Dev", "confidence": "high"}
            if "React JS" not in session.last_courses_discussed:
                session.last_courses_discussed.append("React JS")
        elif "python" in q:
            session.memory_profile["target_domain"] = {"value": "Python Development", "confidence": "high"}
            if "Python" not in session.last_courses_discussed:
                session.last_courses_discussed.append("Python")
        elif "java" in q:
            session.memory_profile["target_domain"] = {"value": "Java Programming", "confidence": "high"}
            if "Java" not in session.last_courses_discussed:
                session.last_courses_discussed.append("Java")
        elif "dsa" in q or "algorithm" in q or "tree" in q or "search" in q:
            session.memory_profile["target_domain"] = {"value": "Data Structures & Algos", "confidence": "high"}
            if "DSA" not in session.last_courses_discussed:
                session.last_courses_discussed.append("DSA")
        elif "sql" in q or "database" in q or "db" in q:
            session.memory_profile["target_domain"] = {"value": "Database Management", "confidence": "high"}
            if "SQL/DBMS" not in session.last_courses_discussed:
                session.last_courses_discussed.append("SQL/DBMS")
                
    # Detect weak subject if user asks for explanation/help on difficult things
    if any(k in q for k in ["confused", "stuck", "difficult", "hard", "help with", "explain", "don't understand"]):
        if "dsa" in q or "algorithm" in q:
            session.memory_profile["weak_subject"] = {"value": "Data Structures", "confidence": "high"}
        elif "programming" in q or "coding" in q:
            session.memory_profile["weak_subject"] = {"value": "Programming Basics", "confidence": "medium"}
        elif "sql" in q or "database" in q:
            session.memory_profile["weak_subject"] = {"value": "SQL Queries", "confidence": "high"}
            
    # Generate contextual recommendations based on active topic
    session.recommendations = generate_recommendations(session.active_topic)

# ── Educational Guardrail Layer ────────────────────────────────────────────────
def is_educational_query(query: str) -> bool:
    query_lower = query.lower().strip()
    
    # Immediately block unsafe or explicitly out-of-scope categories
    safety_blocklist = [
        r"\bhack(ing|er|s)?\b", r"\bexploits?\b", r"\bbypass\s+security\b", r"\bpolitics?\b", 
        r"\belections?\b", r"\bspam\b", r"\brecipes?\b", r"\bmovies?\b", r"\bgames?\b", 
        r"\bbake\s+a\s+cake\b", r"\btell\s+a\s+joke\b", r"\bweather\b"
    ]
    for pattern in safety_blocklist:
        if re.search(pattern, query_lower):
            return False
            
    educational_keywords = [
        "course", "placement", "internship", "certif", "lms", "support", "help",
        "contact", "learn", "study", "exam", "admission", "fee", "price", "duration",
        "react", "python", "java", "coding", "partner", "vtu", "job", "career",
        "syllabus", "curriculum", "schedule", "class", "project", "assignment", "bot", 
        "hello", "hi", "hey", "edutainer", "edubot", "who are you", "what is your name", "help",
        "what", "how", "why", "who", "where", "when", "can", "explain", "tell", "describe",
        "programming", "code", "developer", "development", "software", "database", "sql",
        "computer", "science", "engineering", "math", "algorithm", "data", "structure",
        "learn", "study", "teach", "explain", "tutorial", "guide", "concept", "javascript",
        "c++", "c#", "ruby", "rust", "go", "php", "web", "html", "css", "git", "github",
        "frontend", "backend", "fullstack", "technology", "network", "security",
        "2nd year", "second year", "3rd year", "third year", "4th year", "fourth year",
        "final year", "sophomore", "junior", "senior", "capstone", "dsa", "oop", 
        "system design", "resume", "interview", "mentor", "roadmaps", "placement prep"
    ]
    
    if any(keyword in query_lower for keyword in educational_keywords):
        return True
    return False

REJECTION_RESPONSE = (
    "I am EduBot, your dedicated AI Engineering Academic Mentor. I can only assist you with questions "
    "related to engineering courses, core CS concepts (DSA, OOP, SQL), placements, internships, VTU certifications, "
    "and LMS support.\n\n"
    "Please let me know how I can help you with your academic or career development in engineering!"
)

# ── Pydantic Request/Response Models ──────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    session_id: str = None

class QueryResponse(BaseModel):
    question: str
    response: str
    active_topic: str
    active_intent: str
    active_goal: str | None
    mode: str
    memory_profile: dict
    recommendations: list[dict]
    retrieval_confidence: float | None = 1.0

class ClearRequest(BaseModel):
    session_id: str

class StateResponse(BaseModel):
    session_id: str
    active_topic: str
    active_intent: str
    workflow: str
    mode: str
    last_courses_discussed: list[str]
    active_goal: str | None
    memory_profile: dict
    recommendations: list[dict]
    retrieval_confidence: float | None = 1.0

# ── API Routes ────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    # 0. Input Validation
    q_stripped = req.question.strip()
    session = get_or_create_session(req.session_id)
    
    if not q_stripped:
        return QueryResponse(
            question=req.question,
            response="Query cannot be empty. Please ask a valid question.",
            active_topic=session.active_topic,
            active_intent="INVALID_QUERY",
            active_goal=session.active_goal,
            mode=session.mode,
            memory_profile=session.memory_profile,
            recommendations=session.recommendations,
            retrieval_confidence=0.0
        )
        
    if len(req.question) > 1500:
        return QueryResponse(
            question=req.question,
            response="Query is too long (maximum 1500 characters). Please simplify your question.",
            active_topic=session.active_topic,
            active_intent="INVALID_QUERY",
            active_goal=session.active_goal,
            mode=session.mode,
            memory_profile=session.memory_profile,
            recommendations=session.recommendations,
            retrieval_confidence=0.0
        )

    # 1. Guardrail Check
    if not is_educational_query(req.question):
        return QueryResponse(
            question=req.question,
            response=REJECTION_RESPONSE,
            active_topic=session.active_topic,
            active_intent="OUT_OF_SCOPE",
            active_goal=session.active_goal,
            mode=session.mode,
            memory_profile=session.memory_profile,
            recommendations=session.recommendations,
            retrieval_confidence=0.0
        )
        
    # 2. Update session state
    update_session_state(session, req.question)
    
    # 3. Query RAG engine
    if query_engine is not None:
        try:
            response_obj = query_engine.query(req.question)
            response_text = str(response_obj)
            
            # Simple retrieval confidence calculation
            confidence = 1.0
            source_nodes = getattr(response_obj, "source_nodes", [])
            if source_nodes:
                # Average of top scores normalized (rough estimate)
                scores = [nws.score for nws in source_nodes if nws.score is not None]
                if scores:
                    confidence = min(1.0, max(0.1, sum(scores) / len(scores) / 10.0))
            
            logger.info(
                f"[RAG Query] Session: {req.session_id} | "
                f"Topic: {session.active_topic} | Intent: {session.active_intent} | "
                f"Confidence: {confidence:.4f}"
            )
            return QueryResponse(
                question=req.question,
                response=response_text,
                active_topic=session.active_topic,
                active_intent=session.active_intent,
                active_goal=session.active_goal,
                mode=session.mode,
                memory_profile=session.memory_profile,
                recommendations=session.recommendations,
                retrieval_confidence=confidence
            )
        except Exception as e:
            logger.error(f"Error querying RAG engine: {e}")
            
    # Offline or error fallback
    return QueryResponse(
        question=req.question,
        response="I'm sorry, I encountered an issue querying the academic mentor database. Please try again later.",
        active_topic=session.active_topic,
        active_intent=session.active_intent,
        active_goal=session.active_goal,
        mode=session.mode,
        memory_profile=session.memory_profile,
        recommendations=session.recommendations,
        retrieval_confidence=0.0
    )

@app.post("/query/stream")
async def query_stream_endpoint(req: QueryRequest):
    # 0. Input Validation
    q_stripped = req.question.strip()
    session = get_or_create_session(req.session_id)
    
    if not q_stripped:
        async def reject_generator():
            yield "Query cannot be empty. Please ask a valid question."
        return StreamingResponse(reject_generator(), media_type="text/event-stream")
        
    if len(req.question) > 1500:
        async def reject_generator():
            yield "Query is too long (maximum 1500 characters). Please simplify your question."
        return StreamingResponse(reject_generator(), media_type="text/event-stream")

    # 1. Guardrail Check
    if not is_educational_query(req.question):
        async def reject_generator():
            yield REJECTION_RESPONSE
        return StreamingResponse(reject_generator(), media_type="text/event-stream")
        
    # 2. Update session state
    update_session_state(session, req.question)
    
    # 3. Get response from RAG engine
    response_text = "I'm sorry, I encountered an issue querying the academic mentor database. Please try again later."
    if query_engine is not None:
        try:
            response_obj = query_engine.query(req.question)
            response_text = str(response_obj)
        except Exception as e:
            logger.error(f"Error querying RAG engine for stream: {e}")
            
    # 4. Stream generator with small delay to simulate typing effect
    async def event_generator():
        # Yield in small chunks of words/tokens for smooth typing effect
        words = re.findall(r'\S+\s*', response_text)
        chunk = ""
        for i, word in enumerate(words):
            chunk += word
            if i % 3 == 0 or i == len(words) - 1:
                yield chunk
                chunk = ""
                await asyncio.sleep(0.02)
                
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/clear")
async def clear_endpoint(req: ClearRequest):
    if req.session_id in sessions:
        del sessions[req.session_id]
    return {"status": "cleared"}


# ── Edumentor 5k Query Endpoint ──────────────────────────────────────────
@app.post("/query/5k", response_model=QueryResponse)
async def query_5k_endpoint(req: QueryRequest):
    """
    RAG query endpoint backed by the deduplicated edumentor_synthetic_5k dataset.
    Useful for comparing answer quality against the /query (ultra-premium) endpoint.
    """
    q_stripped = req.question.strip()
    session    = get_or_create_session(req.session_id)

    if not q_stripped:
        return QueryResponse(
            question=req.question,
            response="Query cannot be empty. Please ask a valid question.",
            active_topic=session.active_topic, active_intent="INVALID_QUERY",
            active_goal=session.active_goal, mode=session.mode,
            memory_profile=session.memory_profile, recommendations=session.recommendations,
            retrieval_confidence=0.0,
        )

    if not is_educational_query(req.question):
        return QueryResponse(
            question=req.question, response=REJECTION_RESPONSE,
            active_topic=session.active_topic, active_intent="OUT_OF_SCOPE",
            active_goal=session.active_goal, mode=session.mode,
            memory_profile=session.memory_profile, recommendations=session.recommendations,
            retrieval_confidence=0.0,
        )

    update_session_state(session, req.question)

    if query_engine_5k is None:
        return QueryResponse(
            question=req.question,
            response="The 5k knowledge base is not yet indexed. Run `build_index_5k.py` first.",
            active_topic=session.active_topic, active_intent=session.active_intent,
            active_goal=session.active_goal, mode=session.mode,
            memory_profile=session.memory_profile, recommendations=session.recommendations,
            retrieval_confidence=0.0,
        )

    try:
        response_obj  = query_engine_5k.query(req.question)
        response_text = str(response_obj)
        source_nodes  = getattr(response_obj, "source_nodes", [])
        scores = [n.score for n in source_nodes if n.score is not None]
        confidence = min(1.0, max(0.1, sum(scores) / len(scores) / 10.0)) if scores else 1.0
        return QueryResponse(
            question=req.question, response=response_text,
            active_topic=session.active_topic, active_intent=session.active_intent,
            active_goal=session.active_goal, mode=session.mode,
            memory_profile=session.memory_profile, recommendations=session.recommendations,
            retrieval_confidence=confidence,
        )
    except Exception as e:
        logger.error(f"Error querying 5k RAG engine: {e}")
        return QueryResponse(
            question=req.question,
            response="An error occurred while querying the 5k knowledge base.",
            active_topic=session.active_topic, active_intent=session.active_intent,
            active_goal=session.active_goal, mode=session.mode,
            memory_profile=session.memory_profile, recommendations=session.recommendations,
            retrieval_confidence=0.0,
        )

@app.get("/state/{session_id}", response_model=StateResponse)
async def state_endpoint(session_id: str):
    session = get_or_create_session(session_id)
    return StateResponse(
        session_id=session.session_id,
        active_topic=session.active_topic,
        active_intent=session.active_intent,
        workflow=session.workflow,
        mode=session.mode,
        last_courses_discussed=session.last_courses_discussed,
        active_goal=session.active_goal,
        memory_profile=session.memory_profile,
        recommendations=session.recommendations,
        retrieval_confidence=1.0
    )

@app.get("/api/stats")
async def stats_endpoint():
    from rag.database.chroma_manager import ChromaManager

    def _collection_stats(collection_name: str) -> dict:
        try:
            mgr = ChromaManager(persist_dir=CHROMA_PERSIST_DIR, collection_name=collection_name)
            col = mgr.get_collection()
            total = col.count()
            data  = col.get(include=["metadatas"])
            topic_counts: dict = {}
            for meta in (data.get("metadatas") or []):
                if meta:
                    t = meta.get("topic", "unknown")
                    topic_counts[t] = topic_counts.get(t, 0) + 1
            return {"total_chunks": total, "topics": topic_counts}
        except Exception as exc:
            return {"error": str(exc)}

    return {
        "primary_collection":  {"name": CHROMA_COLLECTION_NAME,  **_collection_stats(CHROMA_COLLECTION_NAME)},
        "edumentor_5k":        {"name": CHROMA_COLLECTION_5K,     **_collection_stats(CHROMA_COLLECTION_5K)},
    }

@app.get("/health")
async def health():
    from rag.database.chroma_manager import ChromaManager
    ollama_ok = check_ollama_active()

    def _chunk_count(collection_name: str) -> int:
        try:
            return ChromaManager(
                persist_dir=CHROMA_PERSIST_DIR, collection_name=collection_name
            ).get_collection().count()
        except Exception:
            return 0

    return {
        "status":                "ok",
        "ollama_online":         ollama_ok,
        "primary_engine_ready": query_engine is not None,
        "5k_engine_ready":       query_engine_5k is not None,
        "primary_chunks":        _chunk_count(CHROMA_COLLECTION_NAME),
        "5k_chunks":             _chunk_count(CHROMA_COLLECTION_5K),
        "edmentor_ready":        _EDMENTOR_READY and edmentor_topic_classifier is not None,
        "edmentor_llm":          edmentor_llm.status() if _EDMENTOR_READY else {},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EDMENTOR ENDPOINTS
#  Voice-first engineering mentor persona with:
#    ① Domain guard (before any RAG/LLM)
#    ② Centroid-cosine topic classifier (~10ms)
#    ③ Topic-filtered ChromaDB retrieval
#    ④ Last-2-turn conversation history
#    ⑤ Groq LLM (primary) with Ollama fallback
#    ⑥ Hard 80-word voice limit enforcement
# ══════════════════════════════════════════════════════════════════════════════

class EdmentorRequest(BaseModel):
    question: str
    session_id: str = "default"

class EdmentorResponse(BaseModel):
    response: str
    topic: str
    speaking_duration: str
    word_count: int
    groq_used: bool


def _edmentor_not_ready_response(question: str) -> EdmentorResponse:
    msg = (
        "I am having a startup issue right now. That is on me. "
        "Give me a moment and try again."
    )
    return EdmentorResponse(
        response=msg, topic="general",
        speaking_duration="~4s", word_count=len(msg.split()), groq_used=False,
    )


@app.post("/edmentor/query", response_model=EdmentorResponse)
async def edmentor_query(req: EdmentorRequest):
    """
    Edmentor voice mentor query endpoint.
    Uses confidence-based routing and safety filtering.
    """
    if not _EDMENTOR_READY or edmentor_topic_classifier is None:
        return _edmentor_not_ready_response(req.question)

    q = req.question.strip()
    if not q:
        return EdmentorResponse(
            response="I did not catch that. Ask me again.",
            topic="general", speaking_duration="~2s", word_count=6, groq_used=False,
        )

    # 1. Intent check / domain guard
    # We first run the standard check to capture character locks/identity responses
    is_blocked, guard_response, reason = edmentor_guard.check(q)
    if is_blocked:
        if reason in ("identity", "character_lock"):
            final_response = edumentor_filter(guard_response)
        else:
            final_response = "That's outside my lane. I'm here for engineering, placements, DSA, and your career. What do you need help with there?"
            
        return EdmentorResponse(
            response=final_response,
            topic="general",
            speaking_duration=speaking_duration_label(final_response),
            word_count=len(final_response.split()),
            groq_used=False,
        )

    if is_off_domain(q):
        final_response = "That's outside my lane. I'm here for engineering, placements, DSA, and your career. What do you need help with there?"
        return EdmentorResponse(
            response=final_response,
            topic="general",
            speaking_duration=speaking_duration_label(final_response),
            word_count=len(final_response.split()),
            groq_used=False,
        )

    # 2. Topic classify (for metadata response/UI)
    topic = edmentor_topic_classifier.classify(q)

    # 3. LLM generation with confidence routing (local vs interim Groq)
    response_raw, routing_mode = await generate_response_with_routing(q, req.session_id)

    # 4. Safety filter & 250-word sentence boundary truncation
    final_response = edumentor_filter(response_raw)

    # Save turn to memory
    edmentor_memory.save_turn(req.session_id, q, final_response)

    groq_used = "groq_interim" in routing_mode

    logger.info(
        f"[Edmentor Query] Session: {req.session_id} | "
        f"Topic: {topic} | Routing: {routing_mode} | "
        f"Words: {len(final_response.split())}"
    )

    return EdmentorResponse(
        response=final_response,
        topic=topic,
        speaking_duration=speaking_duration_label(final_response),
        word_count=len(final_response.split()),
        groq_used=groq_used,
    )


@app.post("/edmentor/query/stream")
async def edmentor_query_stream(req: EdmentorRequest):
    """
    Streaming variant of /edmentor/query using confidence-based routing.
    Streams final safety-filtered text.
    """
    if not _EDMENTOR_READY or edmentor_topic_classifier is None:
        async def _not_ready():
            yield _edmentor_not_ready_response(req.question).response
        return StreamingResponse(_not_ready(), media_type="text/event-stream")

    q = req.question.strip()
    if not q:
        async def _empty():
            yield "I did not catch that. Ask me again."
        return StreamingResponse(_empty(), media_type="text/event-stream")

    # 1. Intent check / domain guard
    is_blocked, guard_response, reason = edmentor_guard.check(q)
    if is_blocked:
        if reason in ("identity", "character_lock"):
            final_response = edumentor_filter(guard_response)
        else:
            final_response = "That's outside my lane. I'm here for engineering, placements, DSA, and your career. What do you need help with there?"
            
        async def _guard_reject():
            yield final_response
        return StreamingResponse(_guard_reject(), media_type="text/event-stream")

    if is_off_domain(q):
        final_response = "That's outside my lane. I'm here for engineering, placements, DSA, and your career. What do you need help with there?"
        async def _intent_reject():
            yield final_response
        return StreamingResponse(_intent_reject(), media_type="text/event-stream")

    # 2. Generate response with confidence routing
    response_raw, routing_mode = await generate_response_with_routing(q, req.session_id)
    final_response = edumentor_filter(response_raw)

    # Save to memory
    edmentor_memory.save_turn(req.session_id, q, final_response)

    async def token_generator():
        # Stream word-by-word with a very short sleep for typing effect
        words = re.findall(r'\S+\s*', final_response)
        for w in words:
            yield w
            await asyncio.sleep(0.01)

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.delete("/edmentor/session/{session_id}")
async def edmentor_clear_session(session_id: str):
    """Clear Edmentor conversation history for a session."""
    if _EDMENTOR_READY:
        edmentor_memory.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/edmentor/health")
async def edmentor_health():
    """Edmentor-specific health check."""
    if not _EDMENTOR_READY:
        return {"ready": False, "reason": "components_not_loaded"}
    return {
        "ready": True,
        "topic_classifier": edmentor_topic_classifier is not None,
        **edmentor_llm.status(),
    }

