from pathlib import Path
import sys
import time
import re
import asyncio
from typing import Generator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Allow imports from backend root
sys.path.append(str(Path(__file__).resolve().parent))

from rag.retrieval.retriever import load_rag_index, get_edumentor_query_engine, check_ollama_active
from rag.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

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

# ── Load index and query engine once at startup ──────────────────────────────
print("Loading RAG index and query engine...")
try:
    index = load_rag_index()
    query_engine = get_edumentor_query_engine(index)
    print("EduBot Modernized RAG API ready.")
except Exception as e:
    print(f"Error loading database index or query engine: {e}")
    index = None
    query_engine = None

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

sessions = {}

def get_or_create_session(session_id: str) -> SessionState:
    if not session_id:
        session_id = "default"
    if session_id not in sessions:
        sessions[session_id] = SessionState(session_id)
        # Initial recommendations
        sessions[session_id].recommendations = generate_recommendations("general")
    return sessions[session_id]

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
    try:
        from rag.database.chroma_manager import ChromaManager
        chroma_manager = ChromaManager(
            persist_dir=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME
        )
        collection = chroma_manager.get_collection()
        total_chunks = collection.count()
        
        # Get all metadatas to analyze
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        
        # Count chunks by file_name
        file_counts = {}
        for meta in metadatas:
            if meta:
                file_name = meta.get("file_name", "unknown")
                file_counts[file_name] = file_counts.get(file_name, 0) + 1
                
        return {
            "total_chunks": total_chunks,
            "distinct_sources": len(file_counts),
            "sources": file_counts
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health():
    ollama_ok = check_ollama_active()
    chunks_count = 0
    if query_engine is not None:
        try:
            from rag.database.chroma_manager import ChromaManager
            chroma_manager = ChromaManager(
                persist_dir=CHROMA_PERSIST_DIR,
                collection_name=CHROMA_COLLECTION_NAME
            )
            chunks_count = chroma_manager.get_collection().count()
        except Exception:
            chunks_count = 0
            
    return {
        "status": "ok",
        "chunks_indexed": chunks_count,
        "ollama_online": ollama_ok
    }
