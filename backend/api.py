from pathlib import Path
import sys
import os
import time
import re
import asyncio
import logging
import json
from typing import Generator, Optional
from datetime import datetime, timedelta

# Force HuggingFace to offline mode to avoid remote check delays/timeouts
os.environ["HF_HUB_OFFLINE"] = "1"

# Fix Windows cp1252 console encoding (crashes on ✓ ✗ chars without this)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

logger = logging.getLogger("api")


from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

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
    from edmentor.voice_limit import speaking_duration_label
    from edmentor.topic_classifier import EdmentorTopicClassifier
    from edmentor.memory import memory as edmentor_memory
    from edmentor.qwen_client import qwen_client
    from edmentor.tts_router import tts_router
    
    # Imports for confidence routing architecture
    from edmentor.intent_router import is_off_domain
    from edmentor.confidence_router import generate_response_with_routing
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

@app.on_event("startup")
async def startup_event():
    # ── Initialise SQLite analytics DB ───────────────────────────────────────
    try:
        from db import init_db
        init_db()
        print("[DB] edumentor.db initialised.")
    except Exception as _dbe:
        print(f"[DB] init_db failed: {_dbe}")

    asyncio.create_task(cleanup_stale_sessions_periodically())
    
    try:
        print("[WARMUP] Warming up ChromaDB retriever...")
        from edmentor.rag_engine import retrieve
        dummy_docs = await retrieve("dsa arrays warmup")
        print(f"[WARMUP] Done — {len(dummy_docs)} docs returned")
    except Exception as e:
        print(f"[WARMUP] Failed: {e}")

    if _EDMENTOR_READY:
        try:
            from edmentor.confidence_router import check_ollama_health
            async def run_health_check():
                ok = await check_ollama_health()
                if not ok:
                    logger.warning("[WARNING] Ollama is down or model is not loaded!")
                    print("[WARNING] Ollama is down or model is not loaded!")
            asyncio.create_task(run_health_check())
        except Exception as e:
            logger.warning(f"Failed to start Ollama health check task: {e}")

@app.on_event("startup")
async def startup_warmup():
    print("[WARMUP] Starting TTS warmup...")
    try:
        from edmentor.tts_router import _load_kokoro, _try_kokoro
        _load_kokoro()
        audio, _ = _try_kokoro("Here is the roadmap for your placement preparation.", "af_heart", 0.95)
        if audio:
            print("[WARMUP] TTS warmup completed successfully.")
        else:
            print("[WARMUP] TTS warmup failed (no audio bytes).")
    except Exception as e:
        print(f"[WARMUP] TTS warmup error: {e}")

# Mount static files
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Register Edmentor TTS router if available
if tts_router is not None:
    app.include_router(tts_router)



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
        "system design", "resume", "interview", "mentor", "roadmaps", "placement prep",
        # Engineering disciplines & core concepts
        "circuit", "transistor", "signal processing", "control system", "vlsi", "microcontroller",
        "thermodynamics", "fluid mechanics", "heat transfer", "structural analysis", "concrete", "steel",
        "electromagnetism", "power system", "transformer", "generator", "solid mechanics",
        "mechanics", "kinematics", "machine design", "signal", "processing", "aerodynamics", "hydraulics",
        "chemical", "electronics", "electrical", "civil", "mechanical", "volt", "current",
        "resistor", "capacitor", "inductor", "diode", "op-amp", "amplifier", "digital electronics",
        "microprocessor", "embedded systems", "cad", "fem", "fea", "structural", "surveying",
        "soil mechanics", "thermodynamic", "entropy", "enthalpy", "refrigeration", "engine", "combustion",
        "fluids", "pressure", "bernoulli", "signals", "fourier", "laplace", "z-transform", "dsp",
        "feedback", "transfer function", "stability", "bode plot", "nyquist",
        "derivation", "formula", "numerical", "problems", "solving", "concepts", "theory", "fundamentals"
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
            response_text = edumentor_filter(str(response_obj))
            
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
            response_text = edumentor_filter(str(response_obj))
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
        response_text = edumentor_filter(str(response_obj))
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
        "qwen_available":        qwen_client.is_available() if _EDMENTOR_READY else False,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EDMENTOR ENDPOINTS
#  Voice-first engineering mentor persona with:
#    ① Domain guard (before any RAG/LLM)
#    ② Centroid-cosine topic classifier (~10ms)
#    ③ Topic-filtered ChromaDB retrieval
#    ④ Last-2-turn conversation history
#    ⑤ Local Qwen LLM with ChromaDB RAG fallback
#    ⑥ Hard 80-word voice limit enforcement
# ══════════════════════════════════════════════════════════════════════════════

class ShowBlock(BaseModel):
    type: str
    lang: str = ""
    content: str

class EdmentorRequest(BaseModel):
    question: str
    session_id: str = "default"

class EdmentorResponse(BaseModel):
    response: str
    topic: str
    speaking_duration: str
    word_count: int
    shows: list[ShowBlock] = []

def parse_dual_output(text: str) -> dict:
    from edmentor.confidence_router import StreamingDualParser
    parser = StreamingDualParser()
    events = parser.feed(text)
    events += parser.finalize()
    
    speak_parts = []
    shows = []
    for ev in events:
        if ev["type"] == "text":
            speak_parts.append(ev["content"])
        elif ev["type"] == "show":
            shows.append({
                "type": ev["show_type"],
                "lang": ev["lang"],
                "content": ev["content"]
            })
    
    speak_text = " ".join(speak_parts).strip()
    
    if "<speak>" not in text and "<show" not in text:
        from edmentor.output_sanitiser import sanitise
        speak_text = sanitise(speak_text)
        
    return {
        "speak": speak_text,
        "shows": shows
    }


def _edmentor_not_ready_response(question: str) -> EdmentorResponse:
    msg = (
        "I am having a startup issue right now. That is on me. "
        "Give me a moment and try again."
    )
    return EdmentorResponse(
        response=msg, topic="general",
        speaking_duration="~4s", word_count=len(msg.split()),
    )


async def generate_tts_async(text: str) -> str:
    """
    Call the Kokoro TTS endpoint and return a base-64-encoded WAV string.
    Returns "" if Kokoro is unavailable — caller must NOT fall back to
    Web Speech API; it should simply omit the audio event.
    """
    import base64
    import httpx
    import time

    start_tts = time.time()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "http://localhost:8000/edmentor/tts",
                json={"text": text, "voice": "af_heart", "speed": 0.95},
            )
        if resp.status_code != 200:
            logger.error(f"[TTS] Kokoro HTTP {resp.status_code} — no audio emitted")
            return ""
        content_type = resp.headers.get("content-type", "")
        if "audio" not in content_type:
            # JSON response means tts_unavailable
            body = resp.json()
            reason = body.get("reason", "unknown")
            logger.error(f"[TTS] Kokoro unavailable — reason={reason}. Fix Kokoro; do NOT fall back to Web Speech API.")
            return ""
        audio_b64 = base64.b64encode(resp.content).decode("utf-8")
        word_count = len(text.split())
        tts_ms = int((time.time() - start_tts) * 1000)
        print(f"[TTS] Kokoro — input_words={word_count} tts_ms={tts_ms}")
        return audio_b64
    except Exception as exc:
        logger.error(f"[TTS] Kokoro call failed: {exc}. No audio emitted. Fix Kokoro before re-enabling TTS.")
        return ""


async def generate_tts_stream_async(text: str):
    """
    Directly streams Kokoro TTS audio chunks as base-64 encoded WAV strings.
    """
    import base64
    import time
    import io
    import asyncio
    try:
        from edmentor.tts_router import _kokoro_instance
        if _kokoro_instance is None:
            return
            
        import soundfile as sf
        
        start_tts = time.time()
        samples, sample_rate = _kokoro_instance.create(text, voice="af_heart", speed=0.95, lang="en-us")
        
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV")
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode("utf-8")
        yield audio_b64
            
        tts_ms = int((time.time() - start_tts) * 1000)
        print(f"[TTS Stream] completed in {tts_ms}ms")
    except Exception as exc:
        logger.error(f"[TTS Stream] Kokoro stream failed: {exc}")


# ════════════════════════════════════════════════════════════════════════════════
#  AUTH LAYER  (Part 3)
#  Simple JWT auth — no OAuth, no external service.
#  Libraries: python-jose[cryptography], passlib[bcrypt]
# ════════════════════════════════════════════════════════════════════════════════
from dotenv import load_dotenv
load_dotenv()

try:
    from jose import JWTError, jwt
    import bcrypt as _bcrypt_lib
    _AUTH_READY = True
except ImportError:
    _AUTH_READY = False
    logger.warning("[AUTH] python-jose or bcrypt not installed. Auth routes will return 503.")

_SECRET_KEY    = os.getenv("SECRET_KEY", "change-me-in-production-please")
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRE_H  = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

_http_bearer = HTTPBearer(auto_error=False)


def _hash_password(plain: str) -> str:
    """Hash a plaintext password with bcrypt."""
    salt = _bcrypt_lib.gensalt()
    return _bcrypt_lib.hashpw(plain.encode("utf-8"), salt).decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    try:
        return _bcrypt_lib.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def _create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=_JWT_EXPIRE_H)
    return jwt.encode(payload, _SECRET_KEY, algorithm=_JWT_ALGORITHM)


async def get_current_student(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_http_bearer),
    x_student_id: Optional[str] = Header(default=None, alias="X-Student-ID"),
) -> dict:
    """
    FastAPI dependency.
    Tries Bearer token first; falls back to X-Student-ID header (anonymous mode
    for backwards-compatible testing).
    Raises 401 if token is present but invalid.
    """
    # ─ No credentials at all — check X-Student-ID fallback ──────────────────
    if credentials is None:
        if x_student_id:
            return {"student_id": x_student_id, "username": x_student_id, "year": None}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not _AUTH_READY:
        raise HTTPException(status_code=503, detail="Auth module not installed")

    try:
        payload = jwt.decode(credentials.credentials, _SECRET_KEY, algorithms=[_JWT_ALGORITHM])
        student_id: str = payload.get("student_id")
        username: str   = payload.get("username")
        if not student_id:
            raise ValueError("missing student_id")
        return {"student_id": student_id, "username": username, "year": payload.get("year")}
    except (JWTError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_current_student_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_http_bearer),
    x_student_id: Optional[str] = Header(default=None, alias="X-Student-ID"),
) -> dict:
    """
    Like get_current_student but never raises 401 — returns anonymous instead.
    Used on /edmentor/* so existing tests without a token continue to work.
    """
    try:
        return await get_current_student(credentials, x_student_id)
    except HTTPException:
        return {"student_id": "anonymous", "username": "anonymous", "year": None}


# ── Auth Pydantic models ──────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    password: str
    year:     Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str


# ── Auth routes ───────────────────────────────────────────────────────────────────
@app.post("/auth/register", status_code=201)
async def auth_register(req: RegisterRequest):
    """Create a new student account."""
    if not _AUTH_READY:
        raise HTTPException(status_code=503, detail="Auth module not installed")
    from db import create_student, get_student_by_username
    import sqlite3
    if get_student_by_username(req.username):
        raise HTTPException(status_code=400, detail="Username already taken")
    try:
        student = create_student(
            username=req.username,
            password_hash=_hash_password(req.password),
            year=req.year,
        )
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already taken")
    return {"student_id": student["student_id"], "username": student["username"]}


@app.post("/auth/login")
async def auth_login(req: LoginRequest):
    """Verify password and return a JWT access token."""
    if not _AUTH_READY:
        raise HTTPException(status_code=503, detail="Auth module not installed")
    from db import get_student_by_username
    student = get_student_by_username(req.username)
    if not student or not _verify_password(req.password, student["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    token = _create_token({
        "student_id": student["student_id"],
        "username":   student["username"],
        "year":       student.get("year"),
    })
    return {"access_token": token, "token_type": "bearer"}


@app.get("/auth/me")
async def auth_me(student: dict = Depends(get_current_student)):
    """Return the currently authenticated student."""
    return {
        "student_id": student["student_id"],
        "username":   student["username"],
        "year":       student.get("year"),
    }


# ════════════════════════════════════════════════════════════════════════════════
#  ANALYTICS DASHBOARD ROUTES  (Part 4)
#  All routes require get_current_student.
#  Students can only query their own data.
# ════════════════════════════════════════════════════════════════════════════════

def _enforce_own_data(student_id_path: str, student: dict):
    """Raise 403 if the token owner doesn't match the requested student_id."""
    if student["student_id"] != student_id_path:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own data.",
        )


@app.get("/dashboard/{student_id}/stats")
async def dashboard_stats(
    student_id: str,
    student: dict = Depends(get_current_student),
):
    """Aggregated learning stats for the dashboard."""
    _enforce_own_data(student_id, student)
    from db import get_student_stats
    return get_student_stats(student_id)


@app.get("/dashboard/{student_id}/timeline")
async def dashboard_timeline(
    student_id: str,
    limit: int = 50,
    offset: int = 0,
    student: dict = Depends(get_current_student),
):
    """Recent Q&A history ordered by timestamp DESC."""
    _enforce_own_data(student_id, student)
    from db import get_student_turns
    return get_student_turns(student_id, limit=limit, offset=offset)


@app.get("/dashboard/{student_id}/gaps")
async def dashboard_gaps(
    student_id: str,
    student: dict = Depends(get_current_student),
):
    """Topics where the student repeats questions or has RAG knowledge gaps."""
    _enforce_own_data(student_id, student)
    from db import get_weak_areas
    return get_weak_areas(student_id)


@app.get("/dashboard/{student_id}/insights")
async def dashboard_insights(
    student_id: str,
    student: dict = Depends(get_current_student),
):
    """Plain-text mentor insights computed from analytics — no LLM call."""
    _enforce_own_data(student_id, student)
    from db import get_student_stats, get_weak_areas
    import sqlite3
    from datetime import date

    stats = get_student_stats(student_id)
    weak  = get_weak_areas(student_id)

    topic_breakdown: dict = stats.get("topic_breakdown", {})
    total_turns = stats.get("total_turns", 0)

    # ─ Focus recommendation ───────────────────────────────────────────────────────
    focus_recommendation = "Keep exploring different topics to build a well-rounded profile."
    if topic_breakdown:
        sorted_topics = sorted(topic_breakdown.items(), key=lambda x: x[1], reverse=True)
        top_topic, top_count = sorted_topics[0]
        if len(sorted_topics) > 1:
            bottom_topic, bottom_count = sorted_topics[-1]
            if top_count >= 3 * max(bottom_count, 1):
                focus_recommendation = (
                    f"You've asked {top_count} questions on {top_topic} but only "
                    f"{bottom_count} on {bottom_topic} — consider broadening before placement season."
                )

    # ─ Consistency signal (active days in last 7) ────────────────────────────
    active_days = stats.get("active_days", 0)
    consistency_signal = "Not enough data yet to assess consistency."
    if active_days >= 5:
        consistency_signal = f"You've been active {active_days} of the last 7 days — good momentum."
    elif active_days >= 3:
        consistency_signal = f"You've been active {active_days} days recently — aim for daily practice."
    elif active_days > 0:
        consistency_signal = f"You've only been active {active_days} day(s) recently — try to study a little each day."

    # ─ Strongest / weakest topic ──────────────────────────────────────────────
    strongest_topic = "None yet"
    weakest_topic   = "None yet"
    if topic_breakdown:
        strongest_topic = max(topic_breakdown, key=topic_breakdown.get)
        weakest_topic   = min(topic_breakdown, key=topic_breakdown.get)
        wt_count = topic_breakdown[weakest_topic]
        weakest_topic = (
            f"{weakest_topic.replace('_',' ').title()} — only {wt_count} question(s) ever. "
            "If this is relevant to your goals, ask more."
        )

    # ─ Session pattern ──────────────────────────────────────────────────────────
    avg_len = stats.get("avg_session_length_turns", 0)
    session_pattern = "Start more sessions to build a study pattern."
    if avg_len >= 5:
        session_pattern = f"Your sessions average {avg_len:.1f} questions — you go deep when you sit down. Great habit."
    elif avg_len >= 2:
        session_pattern = f"Your sessions average {avg_len:.1f} questions. Try to push deeper in each session."
    elif avg_len > 0:
        session_pattern = f"Your sessions are short ({avg_len:.1f} questions on average). Try to ask follow-up questions."

    return {
        "focus_recommendation": focus_recommendation,
        "consistency_signal":   consistency_signal,
        "strongest_topic":      strongest_topic,
        "weakest_topic":        weakest_topic,
        "session_pattern":      session_pattern,
    }


# ── Static pages for login / dashboard ──────────────────────────────────────────────────────────
@app.get("/login.html")
async def serve_login():
    return FileResponse(str(STATIC_DIR / "login.html"))

@app.get("/dashboard.html")
async def serve_dashboard():
    return FileResponse(str(STATIC_DIR / "dashboard.html"))


@app.post("/edmentor/query", response_model=EdmentorResponse)
async def edmentor_query(
    req: EdmentorRequest,
    student: dict = Depends(get_current_student_optional),
):
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
            topic="general", speaking_duration="~2s", word_count=6,
        )

    # 1. Topic classify (for metadata response/UI)
    topic = edmentor_topic_classifier.classify(q)

    # 2. Complete generation with confidence routing (includes memory storage inside the router)
    raw_response, routing_mode = await generate_response_with_routing(
        q, req.session_id, student_id=student["student_id"]
    )

    parsed = parse_dual_output(raw_response)
    speak_text = parsed["speak"]
    shows = parsed["shows"]
    if routing_mode == "technical-concept":
        shows = []

    logger.info(
        f"[Edmentor Query] Session: {req.session_id} | "
        f"Topic: {topic} | Routing: {routing_mode} | "
        f"Words: {len(speak_text.split())}"
    )

    return EdmentorResponse(
        response=speak_text,
        topic=topic,
        speaking_duration=speaking_duration_label(speak_text),
        word_count=len(speak_text.split()),
        shows=shows,
    )


@app.get("/edmentor/query/stream")
async def edmentor_query_stream(
    question: str,
    session_id: str = "default",
    engine: str = "rag_qwen",
    student: dict = Depends(get_current_student_optional),
):
    """
    Streaming variant of /edmentor/query using confidence-based routing.
    Streams final safety-filtered text and audio interleaved as SSE events.
    """
    if not _EDMENTOR_READY or edmentor_topic_classifier is None:
        async def _not_ready():
            yield "event: text\ndata: I am having a startup issue right now. That is on me. Give me a moment and try again.\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        return StreamingResponse(_not_ready(), media_type="text/event-stream")

    q = question.strip()
    if not q:
        async def _empty():
            yield "event: text\ndata: I did not catch that. Ask me again.\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    # 1. Get response based on engine choice
    if engine == "rag":
        if query_engine is not None:
            response_obj = query_engine.query(q)
            response_text = str(response_obj)
            if _EDMENTOR_READY:
                from edmentor.safety_filter import edumentor_filter
                response_text = edumentor_filter(response_text)
        else:
            response_text = "Primary RAG engine is currently unavailable."

        async def static_event_generator():
            from edmentor.confidence_router import split_into_sentences
            raw_sentences = split_into_sentences(response_text.strip())
            sentences = [s for s in raw_sentences if len(s.split()) >= 3]
            for sentence in sentences:
                safe_sentence = sentence.replace('\n', ' ')
                yield f"event: text\ndata: {safe_sentence}\n\n"
                async for audio_b64 in generate_tts_stream_async(sentence):
                    yield f"event: audio\ndata: {audio_b64}\n\n"
            yield "event: done\ndata: {}\n\n"
        return StreamingResponse(static_event_generator(), media_type="text/event-stream")
    else:
        # True real-time streaming using confidence routing & LLM streaming generator
        from edmentor.confidence_router import generate_stream_with_routing

        async def streaming_event_generator():
            try:
                async for event in generate_stream_with_routing(
                    q, session_id, student_id=student["student_id"]
                ):
                    if event["type"] == "text":
                        sentence = event["content"]
                        safe_sentence = sentence.replace('\n', ' ')
                        # Yield the sentence text to trigger typing immediately
                        yield f"event: text\ndata: {safe_sentence}\n\n"
                        # Generate and yield the audio chunk for the sentence
                        async for audio_b64 in generate_tts_stream_async(sentence):
                            yield f"event: audio\ndata: {audio_b64}\n\n"
                    elif event["type"] == "show":
                        show_data = {
                            "type": event["show_type"],
                            "lang": event["lang"],
                            "content": event["content"]
                        }
                        yield f"event: show\ndata: {json.dumps(show_data)}\n\n"
                yield "event: done\ndata: {}\n\n"
            except Exception as e:
                logger.error(f"Error in streaming event_generator: {e}")
                yield "event: error\ndata: An error occurred during streaming.\n\n"

        return StreamingResponse(streaming_event_generator(), media_type="text/event-stream")


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
    from edmentor.qwen_client import qwen_client
    return {
        "ready": True,
        "topic_classifier": edmentor_topic_classifier is not None,
        "qwen_available": qwen_client.is_available(),
    }

