import os
import sys
import time
import logging
import asyncio
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Allow imports from backend root and ai_core
AI_CORE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = AI_CORE_DIR.parent
BACKEND_DIR = WORKSPACE_DIR / "backend"
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))
if str(AI_CORE_DIR) not in sys.path:
    sys.path.append(str(AI_CORE_DIR))



from ai_core.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    RAG_MODE
)
from ai_core.safety.guard import is_educational_query, REJECTION_RESPONSE
from ai_core.intent.classifier import IntentClassifier
from ai_core.memory.conversation import ConversationResolver
from ai_core.rag.retriever import EduMentorAsyncRetriever, load_rag_index_v3
from ai_core.llm.qwen_client import QwenOllamaClient
from ai_core.llm.prompt_builder import PromptBuilder
from rag.database.chroma_manager import ChromaManager

# Setup logs
LOGS_DIR = AI_CORE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=str(LOGS_DIR / "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RAG_Pipeline_V3")

app = FastAPI(title="EduMentor Enterprise RAG API v3")

# Resolve Static files folder (located in backend/static)
STATIC_DIR = BACKEND_DIR / "static"
if not STATIC_DIR.exists():
    # Try local workspace fallbacks
    STATIC_DIR = WORKSPACE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Startup Initialization ──────────────────────────────────────────────────
print("Loading BAAI embedding index & model components...")
try:
    index = load_rag_index_v3()
    retriever = EduMentorAsyncRetriever(index)
    intent_classifier = IntentClassifier(retriever.embed_model)
    resolver = ConversationResolver()
    llm_client = QwenOllamaClient()
    prompt_builder = PromptBuilder()
    print("EduMentor AI RAG System v3 loaded successfully.")
except Exception as e:
    print(f"Error loading system components at startup: {e}")
    index = None
    retriever = None
    intent_classifier = None
    resolver = None
    llm_client = None
    prompt_builder = None

# ── Pydantic Models ──────────────────────────────────────────────────────────
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

# ── Helper Recommendations Generator ─────────────────────────────────────────
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

# ── API Routes ────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_ui():
    index_path = STATIC_DIR / "index.html"
    return FileResponse(str(index_path))

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    # 0. Input Validation
    q_stripped = req.question.strip()
    session_id = req.session_id or "default"
    session = resolver.get_session_state(session_id)
    
    if not q_stripped:
        return QueryResponse(
            question=req.question,
            response="Query cannot be empty. Please ask a valid question.",
            active_topic=session["active_topic"],
            active_intent="INVALID_QUERY",
            active_goal=session["active_goal"],
            mode=session["mode"],
            memory_profile=session["memory_profile"],
            recommendations=generate_recommendations(session["active_topic"]),
            retrieval_confidence=0.0
        )
        
    if len(req.question) > 1500:
        return QueryResponse(
            question=req.question,
            response="Query is too long (maximum 1500 characters). Please simplify your question.",
            active_topic=session["active_topic"],
            active_intent="INVALID_QUERY",
            active_goal=session["active_goal"],
            mode=session["mode"],
            memory_profile=session["memory_profile"],
            recommendations=generate_recommendations(session["active_topic"]),
            retrieval_confidence=0.0
        )

    # 1. Guardrail Check
    if not is_educational_query(req.question):
        return QueryResponse(
            question=req.question,
            response=REJECTION_RESPONSE,
            active_topic=session["active_topic"],
            active_intent="OUT_OF_SCOPE",
            active_goal=session["active_goal"],
            mode=session["mode"],
            memory_profile=session["memory_profile"],
            recommendations=generate_recommendations(session["active_topic"]),
            retrieval_confidence=0.0
        )

    # 2. Fast Cosine Intent Classification
    intent = intent_classifier.classify(req.question)
    
    # 3. Update memory session state
    resolver.update_session_state(session_id, req.question, intent)
    session = resolver.get_session_state(session_id)
    
    # 4. In-Memory Pronoun Resolver
    resolved_query = resolver.resolve_pronouns(session_id, req.question)
    
    # 5. Route through Hybrid search & confidence Reranking if needed
    use_rag = False
    context = ""
    rerank_score = 0.0
    
    if intent == "rag" and retriever is not None:
        use_rag, context, debug_log = await retriever.retrieve_pipeline(resolved_query, session)
        rerank_score = debug_log.get("rerank_score", 0.0)

    # 6. Formulate Prompts
    if not llm_client.check_active():
        # Generate high-fidelity offline formatted RAG response
        if use_rag and context:
            explanation = context
            if "Mentor Explanation:" in context:
                parts = context.split("Mentor Explanation:")
                if len(parts) > 1:
                    explanation = parts[1].split("Topic:")[0].strip()
            # Clean HTML tags
            explanation = re.sub(r'<[^>]*>', '', explanation)
            response_text = (
                f"*(EduMentor - Offline Database Mode)*\n\n"
                f"### Senior Mentor Explanation:\n"
                f"{explanation}\n\n"
                f"### Actionable Mentoring Steps:\n"
                f"1. Focus your study on **{session['active_topic']}** from verified references.\n"
                f"2. Apply these concepts directly in hands-on programming projects.\n"
                f"3. Make mistakes early, debug extensively, and analyze worst-case time complexities.\n\n"
                f"*(Retrieved from verified engineering knowledge base)*"
            )
        else:
            response_text = (
                f"*(EduMentor - Offline Database Mode)*\n\n"
                f"Hello! I am EduMentor, your AI Academic Mentor.\n\n"
                f"My local LLM daemon is currently offline, so I am running in high-precision database search mode. "
                f"I currently do not have verified matches in my database for this specific question. "
                f"However, as your mentor, I recommend searching our LMS portal or raising a ticket under Support!"
            )
    elif use_rag:
        prompt = prompt_builder.build_rag_prompt(req.question, context)
        response_text = await llm_client.generate(prompt)
    else:
        prompt = prompt_builder.build_direct_prompt(req.question, session)
        system = prompt_builder.get_direct_system_prompt()
        response_text = await llm_client.generate(prompt, system_prompt=system)
        
    return QueryResponse(
        question=req.question,
        response=response_text,
        active_topic=session["active_topic"],
        active_intent=intent,
        active_goal=session["active_goal"],
        mode=session["mode"],
        memory_profile=session["memory_profile"],
        recommendations=generate_recommendations(session["active_topic"]),
        retrieval_confidence=rerank_score
    )

@app.post("/query/stream")
async def query_stream_endpoint(req: QueryRequest):
    # 0. Input Validation
    q_stripped = req.question.strip()
    session_id = req.session_id or "default"
    session = resolver.get_session_state(session_id)
    
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

    # 2. Fast Cosine Intent Classification
    intent = intent_classifier.classify(req.question)
    
    # 3. Update memory session state
    resolver.update_session_state(session_id, req.question, intent)
    session = resolver.get_session_state(session_id)
    
    # 4. In-Memory Pronoun Resolver
    resolved_query = resolver.resolve_pronouns(session_id, req.question)
    
    # 5. Route through Hybrid search & confidence Reranking if needed
    use_rag = False
    context = ""
    
    if intent == "rag" and retriever is not None:
        use_rag, context, _ = await retriever.retrieve_pipeline(resolved_query, session)

    # 6. Formulate Prompts & Stream
    if not llm_client.check_active():
        # Generate offline response
        if use_rag and context:
            explanation = context
            if "Mentor Explanation:" in context:
                parts = context.split("Mentor Explanation:")
                if len(parts) > 1:
                    explanation = parts[1].split("Topic:")[0].strip()
            # Clean HTML tags
            explanation = re.sub(r'<[^>]*>', '', explanation)
            response_text = (
                f"*(EduMentor - Offline Database Mode)*\n\n"
                f"### Senior Mentor Explanation:\n"
                f"{explanation}\n\n"
                f"### Actionable Mentoring Steps:\n"
                f"1. Focus your study on **{session['active_topic']}** from verified references.\n"
                f"2. Apply these concepts directly in hands-on programming projects.\n"
                f"3. Make mistakes early, debug extensively, and analyze worst-case time complexities.\n\n"
                f"*(Retrieved from verified engineering knowledge base)*"
            )
        else:
            response_text = (
                f"*(EduMentor - Offline Database Mode)*\n\n"
                f"Hello! I am EduMentor, your AI Academic Mentor.\n\n"
                f"My local LLM daemon is currently offline, so I am running in high-precision database search mode. "
                f"I currently do not have verified matches in my database for this specific question. "
                f"However, as your mentor, I recommend searching our LMS portal or raising a ticket under Support!"
            )
            
        async def event_generator():
            # Stream the offline response token by token
            import re
            words = re.findall(r'\S+\s*', response_text)
            chunk = ""
            for i, word in enumerate(words):
                chunk += word
                if i % 3 == 0 or i == len(words) - 1:
                    yield chunk
                    chunk = ""
                    await asyncio.sleep(0.01)
                    
        return StreamingResponse(event_generator(), media_type="text/event-stream")
        
    elif use_rag:
        prompt = prompt_builder.build_rag_prompt(req.question, context)
        token_stream = llm_client.generate_stream(prompt)
    else:
        prompt = prompt_builder.build_direct_prompt(req.question, session)
        system = prompt_builder.get_direct_system_prompt()
        token_stream = llm_client.generate_stream(prompt, system_prompt=system)
        
    async def event_generator():
        async for token in token_stream:
            yield token
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/clear")
async def clear_endpoint(req: ClearRequest):
    # Reset persistent memory using resolver DB
    state = {
        "session_id": req.session_id,
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
    resolver.db.save_session(req.session_id, state)
    return {"status": "cleared"}

@app.get("/state/{session_id}", response_model=StateResponse)
async def state_endpoint(session_id: str):
    session = resolver.get_session_state(session_id)
    return StateResponse(
        session_id=session["session_id"],
        active_topic=session["active_topic"],
        active_intent=session["active_intent"],
        workflow=session["workflow"],
        mode=session["mode"],
        last_courses_discussed=session["last_courses_discussed"],
        active_goal=session["active_goal"],
        memory_profile=session["memory_profile"],
        recommendations=generate_recommendations(session["active_topic"]),
        retrieval_confidence=1.0
    )

@app.get("/api/stats")
async def stats_endpoint():
    try:
        chroma_manager = ChromaManager(
            persist_dir=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME
        )
        collection = chroma_manager.get_collection()
        total_chunks = collection.count()
        
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        
        file_counts = {}
        for meta in metadatas:
            if meta:
                file_name = meta.get("source", "unknown")
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
    ollama_ok = llm_client.check_active()
    chunks_count = 0
    if index is not None:
        try:
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
        "ollama_online": ollama_ok,
        "rag_mode": RAG_MODE
    }
