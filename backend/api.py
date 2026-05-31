from pathlib import Path
import sys
import time
import requests
import re

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Allow imports from backend root
sys.path.append(str(Path(__file__).resolve().parent))

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from rag.prompt import SYSTEM_PROMPT

# Import new enterprise RAG modules
from rag.query_encoder import QueryEncoder
from rag.hybrid_retriever import HybridRetriever
from rag.metadata_router import MetadataRouter
from rag.reranker import Reranker
from rag.context_synthesizer import ContextSynthesizer
from rag.response_cleaner import ResponseCleaner

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_store"
STATIC_DIR = BASE_DIR / "static"

# Ensure logs directory exists
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

# ── Load index once at startup ───────────────────────────────────────────────
print("Loading BAAI embedding model...")
query_encoder = QueryEncoder("BAAI/bge-large-en-v1.5")
embed_model = query_encoder.embed_model

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection("educational_mentor_knowledgebase")

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
    embed_model=embed_model,
)

# Initialize modular enterprise RAG pipeline components
hybrid_retriever = HybridRetriever(index, collection, similarity_top_k=6)
metadata_router = MetadataRouter()
reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
context_synthesizer = ContextSynthesizer()
response_cleaner = ResponseCleaner()

print("EduBot Enterprise RAG API ready.")

# ── Session Storage and Helpers ───────────────────────────────────────────────
sessions = {}

def check_ollama_running() -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_or_create_session(session_id: str):
    if session_id not in sessions:
        ollama_online = check_ollama_running()
        engine = None
        if ollama_online:
            try:
                from llama_index.llms.ollama import Ollama
                llm = Ollama(model="mistral", system_prompt=SYSTEM_PROMPT, request_timeout=120.0)
                memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
                engine = index.as_chat_engine(
                    chat_mode="condense_plus_context",
                    llm=llm,
                    memory=memory,
                    similarity_top_k=2,
                    system_prompt=SYSTEM_PROMPT
                )
                print(f"Created chat engine for session: {session_id}")
            except Exception as e:
                print(f"Error creating Ollama chat engine for {session_id}: {e}")
                engine = None
        
        from rag.conversation_state import ConversationState
        from rag.memory import ContextMemory
        
        sessions[session_id] = {
            "chat_engine": engine,
            "created_at": time.time(),
            "state": ConversationState(session_id),
            "memory": ContextMemory(session_id)
        }
    return sessions[session_id]

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

# ── Enterprise Two-Stage Retrieval Pipeline ──────────────────────────────────
def retrieve_and_rerank_chunks(query: str, intent: str, active_topic: str, debug: bool = True) -> tuple[list, float]:
    """
    Executes the entire enterprise-grade retrieve-filter-route-rerank pipeline.
    Returns: (top_chunks, retrieval_confidence)
    """
    start_total = time.time()
    
    # 1. Broad Hybrid Retrieval (Dense Vector + Sparse Token-Overlap)
    t0 = time.time()
    hybrid_hits = hybrid_retriever.retrieve(query)
    lat_hybrid = time.time() - t0
    
    # 2. Quality Filtering (repetition, short text, UI noise, Jaccard duplicates)
    t0 = time.time()
    from rag.retrieval_filter import filter_retrieved_chunks
    quality_hits = filter_retrieved_chunks(hybrid_hits, similarity_threshold=0.75)
    
    if not quality_hits:
        quality_hits = hybrid_hits
    lat_filter = time.time() - t0
        
    # 3. Metadata Routing & Filtering based on Intent
    t0 = time.time()
    routed_hits = metadata_router.route_and_filter(quality_hits, intent)
    lat_route = time.time() - t0
    
    # 4. Cross-Encoder Reranking and Retrieval Confidence Score Computation
    t0 = time.time()
    top_hits, confidence = reranker.rerank(query, routed_hits, top_n=3)
    lat_rerank = time.time() - t0
    
    total_lat = time.time() - start_total
    
    logger.info(
        f"Pipeline stats for query='{query[:30]}...' | "
        f"hybrid={lat_hybrid:.4f}s ({len(hybrid_hits)} hits) | "
        f"filter={lat_filter:.4f}s ({len(quality_hits)} hits) | "
        f"route={lat_route:.4f}s ({len(routed_hits)} hits) | "
        f"rerank={lat_rerank:.4f}s ({len(top_hits)} hits) | "
        f"total={total_lat:.4f}s | confidence={confidence:.4f}"
    )
    
    if debug:
        print("\n=== ENTERPRISE RAG RETRIEVAL PIPELINE DEBUG ===")
        print(f"Query: '{query}'")
        print(f"Intent: {intent} | Active Topic: {active_topic}")
        print(f"Raw Hybrid Hits: {len(hybrid_hits)}")
        print(f"After Quality Filter: {len(quality_hits)}")
        print(f"After Metadata Routing: {len(routed_hits)}")
        print(f"Retrieval Confidence: {confidence:.4f}")
        print("Reranked Chunks:")
        for idx, h in enumerate(top_hits):
            file_name = h.node.metadata.get("file_name", "")
            excerpt = h.node.text.strip().replace('\n', ' ')[:80] + "..."
            print(f"  [{idx + 1}] File: {file_name} | CrossEncoder Score: {h.score:.4f}")
            print(f"      Excerpt: {excerpt}")
        print("================================================\n")
        
    return top_hits, confidence

def conversational_fallback(question: str) -> str:
    # Basic backwards compatibility helper
    hits, confidence = retrieve_and_rerank_chunks(question, "GENERAL", "general")
    from rag.formatter import conversational_fallback as formatter_fallback
    return formatter_fallback(question, hits)

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

# ── Stream Cleaning Layer ─────────────────────────────────────────────────────
def clean_stream_generator(response_gen):
    buffer = ""
    prefix_stripped = False
    
    banned_prefixes = [
        r"^based\s+on\s+(our|the|edutainer)\s+\w+\s+\w+,?\s*",
        r"^according\s+to\s+(our|the|edutainer)\s+\w+\s+\w+,?\s*",
        r"^based\s+on\s+the\s+provided\s+context,?\s*",
        r"^according\s+to\s+the\s+documents,?\s*",
        r"^based\s+on\s+our\s+knowledge\s+base,?\s*",
        r"^retrieved\s+information:?\s*",
        r"^q\s*:\s*",
        r"^a\s*:\s*",
        r"^question\s*:\s*",
        r"^answer\s*:\s*"
    ]
    
    for token in response_gen:
        if not prefix_stripped:
            buffer += token
            if len(buffer) >= 100 or "\n" in buffer:
                temp_buffer = buffer
                for p in banned_prefixes:
                    match = re.match(p, temp_buffer, re.IGNORECASE)
                    if match:
                        temp_buffer = temp_buffer[match.end():]
                
                temp_buffer = temp_buffer.lstrip()
                if temp_buffer:
                    temp_buffer = temp_buffer[0].upper() + temp_buffer[1:]
                prefix_stripped = True
                yield temp_buffer
            continue
        yield token

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    # 1. Guardrail Check
    if not is_educational_query(req.question):
        session_id = req.session_id or "default"
        session = get_or_create_session(session_id)
        visual_profile = session["memory"].get_visual_profile()
        return QueryResponse(
            question=req.question,
            response=REJECTION_RESPONSE,
            active_topic=session["state"].active_topic,
            active_intent="OUT_OF_SCOPE",
            active_goal=session["state"].active_goal,
            mode=session["state"].mode,
            memory_profile=visual_profile,
            recommendations=[],
            retrieval_confidence=0.0
        )
        
    # 2. Retrieve session first to extract active semantic memory details
    session_id = req.session_id or "default"
    session = get_or_create_session(session_id)
    state_obj = session["state"]
    
    # 3. Preprocess & Expand Query (injecting active topic memory!)
    from rag.preprocessor import preprocess_query
    expanded_query = preprocess_query(
        req.question, 
        active_topic=state_obj.active_topic, 
        last_courses=state_obj.last_courses_discussed
    )
        
    chat_engine = session["chat_engine"]
    
    # 4. Classify intent
    from rag.intent_router import classify_intent
    llm = chat_engine._llm if chat_engine is not None else None
    intent, score = classify_intent(expanded_query, llm)
    
    if intent == "OUT_OF_SCOPE":
        rejection_msg = "I am designed specifically for educational and LMS-related assistance on the Edutainer platform. Please let me know how I can support you in these academic domains!"
        visual_profile = session["memory"].get_visual_profile()
        return QueryResponse(
            question=req.question,
            response=rejection_msg,
            active_topic=state_obj.active_topic,
            active_intent=intent,
            active_goal=state_obj.active_goal,
            mode=state_obj.mode,
            memory_profile=visual_profile,
            recommendations=[],
            retrieval_confidence=0.0
        )
    
    # 5. Update session state and memory
    session["state"].update_state(req.question, intent, expanded_query)
    session["memory"].extract_memories(req.question, intent, expanded_query)
    
    visual_profile = session["memory"].get_visual_profile()
    matched_memory = session["memory"].retrieve_relevant_memories(expanded_query)
    
    # 6. Retrieve, Filter, Route, and Rerank chunks using two-stage pipeline
    filtered_hits, confidence = retrieve_and_rerank_chunks(expanded_query, intent, state_obj.active_topic)

    # 7. Generate contextual recommendations
    from rag.recommendation import generate_recommendations
    recommendations = generate_recommendations(intent, state_obj.active_topic, visual_profile)
    
    # 8. Process Response (synthesizing and cleaning)
    if chat_engine is not None and check_ollama_running():
        try:
            # Context synthesis layer for prompting LLM
            synthesized_context = context_synthesizer.synthesize(expanded_query, filtered_hits)
            contextual_query = req.question
            if matched_memory or synthesized_context:
                contextual_query = f"Synthesized Context:\n{synthesized_context}\n\n{matched_memory}\n\nStudent Query: {req.question}"
                
            response = chat_engine.chat(contextual_query)
            cleaned_llm_response = response_cleaner.clean(str(response.response))
            return QueryResponse(
                question=req.question,
                response=cleaned_llm_response,
                active_topic=state_obj.active_topic,
                active_intent=intent,
                active_goal=state_obj.active_goal,
                mode=state_obj.mode,
                memory_profile=visual_profile,
                recommendations=recommendations,
                retrieval_confidence=confidence
            )
        except Exception as e:
            print(f"Error during standard query chat: {e}")
            
    # Fallback to local high-fidelity synthesized responses if offline
    from rag.formatter import conversational_fallback as formatter_fallback
    fallback_msg = formatter_fallback(req.question, filtered_hits)
    cleaned_fallback = response_cleaner.clean(fallback_msg)
    
    return QueryResponse(
        question=req.question,
        response=cleaned_fallback,
        active_topic=state_obj.active_topic,
        active_intent=intent,
        active_goal=state_obj.active_goal,
        mode=state_obj.mode,
        memory_profile=visual_profile,
        recommendations=recommendations,
        retrieval_confidence=confidence
    )

@app.post("/query/stream")
async def query_stream_endpoint(req: QueryRequest):
    # 1. Guardrail Check
    if not is_educational_query(req.question):
        async def reject_generator():
            yield REJECTION_RESPONSE
        return StreamingResponse(reject_generator(), media_type="text/event-stream")
        
    # 2. Retrieve session first to extract active semantic memory details
    session_id = req.session_id or "default"
    session = get_or_create_session(session_id)
    state_obj = session["state"]
    
    # 3. Preprocess & Expand Query (injecting active topic memory!)
    from rag.preprocessor import preprocess_query
    expanded_query = preprocess_query(
        req.question, 
        active_topic=state_obj.active_topic, 
        last_courses=state_obj.last_courses_discussed
    )
    
    chat_engine = session["chat_engine"]
    
    # 4. Classify intent
    from rag.intent_router import classify_intent
    llm = chat_engine._llm if chat_engine is not None else None
    intent, score = classify_intent(expanded_query, llm)
    
    if intent == "OUT_OF_SCOPE":
        rejection_msg = "I am designed specifically for educational and LMS-related assistance on the Edutainer platform. Please let me know how I can support you in these academic domains!"
        async def reject_generator():
            yield rejection_msg
        return StreamingResponse(reject_generator(), media_type="text/event-stream")
        
    # 5. Update session state and memory
    session["state"].update_state(req.question, intent, expanded_query)
    session["memory"].extract_memories(req.question, intent, expanded_query)
    
    visual_profile = session["memory"].get_visual_profile()
    matched_memory = session["memory"].retrieve_relevant_memories(expanded_query)
    
    # 6. Retrieve, Filter, Route, and Rerank chunks using two-stage pipeline
    filtered_hits, confidence = retrieve_and_rerank_chunks(expanded_query, intent, state_obj.active_topic)

    # 7. Stream response
    if chat_engine is not None and check_ollama_running():
        try:
            # Context synthesis layer for prompting LLM
            synthesized_context = context_synthesizer.synthesize(expanded_query, filtered_hits)
            contextual_query = req.question
            if matched_memory or synthesized_context:
                contextual_query = f"Synthesized Context:\n{synthesized_context}\n\n{matched_memory}\n\nStudent Query: {req.question}"
                
            response = chat_engine.stream_chat(contextual_query)
            
            async def event_generator():
                for token in clean_stream_generator(response.response_gen):
                    yield token
            return StreamingResponse(event_generator(), media_type="text/event-stream")
        except Exception as e:
            print(f"Error during streaming query chat: {e}")
            
    # Fallback stream if offline
    from rag.formatter import conversational_fallback as formatter_fallback
    fallback_msg = formatter_fallback(req.question, filtered_hits)
    cleaned_fallback = response_cleaner.clean(fallback_msg)
    
    async def fallback_generator():
        yield cleaned_fallback
    return StreamingResponse(fallback_generator(), media_type="text/event-stream")

@app.post("/clear")
async def clear_endpoint(req: ClearRequest):
    if req.session_id in sessions:
        if sessions[req.session_id]["chat_engine"] is not None:
            try:
                sessions[req.session_id]["chat_engine"].reset()
            except Exception as e:
                print(f"Error resetting chat engine: {e}")
                
        # Re-instantiate State and Memory to reset completely
        from rag.conversation_state import ConversationState
        from rag.memory import ContextMemory
        sessions[req.session_id]["state"] = ConversationState(req.session_id)
        sessions[req.session_id]["memory"] = ContextMemory(req.session_id)
        
    return {"status": "cleared"}

@app.get("/state/{session_id}", response_model=StateResponse)
async def state_endpoint(session_id: str):
    session = get_or_create_session(session_id)
    s = session["state"]
    m = session["memory"]
    
    from rag.recommendation import generate_recommendations
    recommendations = generate_recommendations(s.active_intent, s.active_topic, m.get_visual_profile())
    
    return StateResponse(
        session_id=session_id,
        active_topic=s.active_topic,
        active_intent=s.active_intent,
        workflow=s.workflow,
        mode=s.mode,
        last_courses_discussed=s.last_courses_discussed,
        active_goal=s.active_goal,
        memory_profile=m.get_visual_profile(),
        recommendations=recommendations,
        retrieval_confidence=1.0
    )

@app.get("/health")
async def health():
    return {"status": "ok", "chunks_indexed": collection.count(), "ollama_online": check_ollama_running()}
