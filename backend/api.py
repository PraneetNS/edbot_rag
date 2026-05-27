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

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_store"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="EduBot RAG API")

# Mount static files
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Load index once at startup ───────────────────────────────────────────────
print("Loading embedding model...")
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection("edubot")

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
    embed_model=embed_model,
)
retriever = index.as_retriever(similarity_top_k=6)
print("EduBot API ready.")

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
                # Sliding memory window using ChatMemoryBuffer (2000 tokens)
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
        # General academic/programming relaxed terms
        "what", "how", "why", "who", "where", "when", "can", "explain", "tell", "describe",
        "programming", "code", "developer", "development", "software", "database", "sql",
        "computer", "science", "engineering", "math", "algorithm", "data", "structure",
        "learn", "study", "teach", "explain", "tutorial", "guide", "concept", "javascript",
        "c++", "c#", "ruby", "rust", "go", "php", "web", "html", "css", "git", "github",
        "frontend", "backend", "fullstack", "technology", "network", "security"
    ]
    
    # Allow short standard greetings and generic platform queries
    if any(keyword in query_lower for keyword in educational_keywords):
        return True
    return False

REJECTION_RESPONSE = (
    "I am EduBot, your dedicated Edutainer AI Academic Mentor. I can only assist you with questions "
    "related to courses, placements, internships, certifications, LMS support, and academics. \n\n"
    "Please let me know how I can help you in any of these educational areas!"
)

# ── Conversational Fallback Mode ──────────────────────────────────────────────
def retrieve_and_rerank_chunks(query: str, intent: str, active_topic: str, debug: bool = True) -> list:
    """
    Retrieves, filters, and reranks chunks based on active intent, topic, keyword matches,
    and vector similarity scores.
    """
    # 1. Perform LlamaIndex vector retrieval (top 6 chunks)
    raw_hits = retriever.retrieve(query)
    
    # 2. Intent-Based Document Filtering
    filtered_hits = []
    placement_intents = ["PLACEMENT_GUIDANCE", "INTERNSHIP_GUIDANCE"]
    support_intents = ["LMS_SUPPORT", "CERTIFICATION_SUPPORT"]
    
    for h in raw_hits:
        file_name = h.node.metadata.get("file_name", "").lower()
        if intent in placement_intents:
            if "homepage" in file_name or "about" in file_name or "courses" in file_name:
                filtered_hits.append(h)
        elif intent in support_intents:
            if "support" in file_name or "faq" in file_name or "homepage" in file_name:
                filtered_hits.append(h)
        elif intent == "COURSE_QUERY":
            if "courses" in file_name or "faq" in file_name:
                filtered_hits.append(h)
        else:
            filtered_hits.append(h)
            
    if not filtered_hits:
        filtered_hits = raw_hits
        
    # 3. Quality Filtering (repetition, short text, UI spam)
    from rag.retrieval_filter import filter_retrieved_chunks
    quality_hits = filter_retrieved_chunks(filtered_hits, similarity_threshold=0.75)
    
    # 4. Custom Semantic Reranking
    reranked_hits = []
    query_words = set(re.findall(r'\b\w{4,}\b', query.lower()))
    
    for h in quality_hits:
        text = h.node.text
        # Baseline vector similarity score
        score = h.score if h.score is not None else 0.5
        
        # Word overlap boost (ratio of query keywords present in chunk)
        overlap_score = 0.0
        if query_words:
            chunk_words_lower = text.lower()
            matches = sum(1 for w in query_words if w in chunk_words_lower)
            overlap_score = (matches / len(query_words)) * 0.25 # Max boost of 0.25
            
        # Topic relevance boost
        topic_boost = 0.0
        if active_topic and active_topic.lower() in text.lower():
            topic_boost = 0.15 # Max boost of 0.15
            
        final_score = score + overlap_score + topic_boost
        # Attach the custom score back to the hit object for inspection
        h.score = final_score
        reranked_hits.append((h, final_score))
        
    # Sort by final custom semantic score in descending order
    reranked_hits.sort(key=lambda x: x[1], reverse=True)
    
    # Print console debug logs
    if debug:
        print("\n=== DEVELOPER RETRIEVAL DEBUG MODE ===")
        print(f"Query: '{query}'")
        print(f"Intent: {intent} | Active Topic: {active_topic}")
        print(f"Raw Hits Retrieved: {len(raw_hits)}")
        print(f"After Intent Filter: {len(filtered_hits)}")
        print(f"After Quality Filter: {len(quality_hits)}")
        print("Reranked Chunks:")
        for idx, (h, score) in enumerate(reranked_hits):
            file_name = h.node.metadata.get("file_name", "")
            excerpt = h.node.text.strip().replace('\n', ' ')[:80] + "..."
            print(f"  [{idx + 1}] File: {file_name} | Custom Score: {score:.3f}")
            print(f"      Excerpt: {excerpt}")
        print("======================================\n")
        
    # Return top 2 or 3 chunks
    return [item[0] for item in reranked_hits[:3]]

def conversational_fallback(question: str) -> str:
    hits = retrieve_and_rerank_chunks(question, "GENERAL", "general")
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
            # Buffer up to 100 characters to detect and strip any banned prefix at the start
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
        # Format a clean safety rejection conforming to schema
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
            recommendations=[]
        )
        
    # 2. Preprocess & Expand Query
    from rag.preprocessor import preprocess_query
    expanded_query = preprocess_query(req.question)
        
    # 3. Retrieve session
    session_id = req.session_id or "default"
    session = get_or_create_session(session_id)
    
    chat_engine = session["chat_engine"]
    
    # 4. Classify intent
    from rag.intent_router import classify_intent
    llm = chat_engine._llm if chat_engine is not None else None
    intent, score = classify_intent(expanded_query, llm)
    
    # Strict Guardrail reject for OUT_OF_SCOPE
    if intent == "OUT_OF_SCOPE":
        rejection_msg = "I am designed specifically for educational and LMS-related assistance on the Edutainer platform. Please let me know how I can support you in these academic domains!"
        visual_profile = session["memory"].get_visual_profile()
        return QueryResponse(
            question=req.question,
            response=rejection_msg,
            active_topic=session["state"].active_topic,
            active_intent=intent,
            active_goal=session["state"].active_goal,
            mode=session["state"].mode,
            memory_profile=visual_profile,
            recommendations=[]
        )
    
    # 5. Update session state and memory
    session["state"].update_state(req.question, intent, expanded_query)
    session["memory"].extract_memories(req.question, intent, expanded_query)
    
    state_obj = session["state"]
    visual_profile = session["memory"].get_visual_profile()
    matched_memory = session["memory"].retrieve_relevant_memories(expanded_query)
    
    # 6. Retrieve, Filter and Rerank chunks (Intent-Aware and Quality-Filtered)
    filtered_hits = retrieve_and_rerank_chunks(expanded_query, intent, state_obj.active_topic)

    # 7. Generate contextual recommendations
    from rag.recommendation import generate_recommendations
    recommendations = generate_recommendations(intent, state_obj.active_topic, visual_profile)
    
    # 8. Process Response
    from rag.formatter import clean_response, conversational_fallback as formatter_fallback
    
    if chat_engine is not None and check_ollama_running():
        try:
            # Inject relevant matched memories into the query
            contextual_query = req.question
            if matched_memory:
                contextual_query = f"{matched_memory}\n\nStudent Query: {req.question}"
                
            response = chat_engine.chat(contextual_query)
            cleaned_llm_response = clean_response(str(response.response))
            return QueryResponse(
                question=req.question,
                response=cleaned_llm_response,
                active_topic=state_obj.active_topic,
                active_intent=intent,
                active_goal=state_obj.active_goal,
                mode=state_obj.mode,
                memory_profile=visual_profile,
                recommendations=recommendations
            )
        except Exception as e:
            print(f"Error during standard query chat: {e}")
            
    # Fallback if offline
    fallback_msg = formatter_fallback(req.question, filtered_hits)
    cleaned_fallback = clean_response(fallback_msg)
    
    return QueryResponse(
        question=req.question,
        response=cleaned_fallback,
        active_topic=state_obj.active_topic,
        active_intent=intent,
        active_goal=state_obj.active_goal,
        mode=state_obj.mode,
        memory_profile=visual_profile,
        recommendations=recommendations
    )

@app.post("/query/stream")
async def query_stream_endpoint(req: QueryRequest):
    # 1. Guardrail Check
    if not is_educational_query(req.question):
        async def reject_generator():
            yield REJECTION_RESPONSE
        return StreamingResponse(reject_generator(), media_type="text/event-stream")
        
    # 2. Preprocess & Expand Query
    from rag.preprocessor import preprocess_query
    expanded_query = preprocess_query(req.question)
    
    # 3. Retrieve session
    session_id = req.session_id or "default"
    session = get_or_create_session(session_id)
    
    chat_engine = session["chat_engine"]
    
    # 4. Classify intent
    from rag.intent_router import classify_intent
    llm = chat_engine._llm if chat_engine is not None else None
    intent, score = classify_intent(expanded_query, llm)
    
    # Strict Safety Guardrail reject for OUT_OF_SCOPE
    if intent == "OUT_OF_SCOPE":
        rejection_msg = "I am designed specifically for educational and LMS-related assistance on the Edutainer platform. Please let me know how I can support you in these academic domains!"
        async def reject_generator():
            yield rejection_msg
        return StreamingResponse(reject_generator(), media_type="text/event-stream")
        
    # 5. Update session state and memory
    session["state"].update_state(req.question, intent, expanded_query)
    session["memory"].extract_memories(req.question, intent, expanded_query)
    
    state_obj = session["state"]
    visual_profile = session["memory"].get_visual_profile()
    matched_memory = session["memory"].retrieve_relevant_memories(expanded_query)
    
    # 6. Retrieve, Filter and Rerank chunks (Intent-Aware and Quality-Filtered)
    filtered_hits = retrieve_and_rerank_chunks(expanded_query, intent, state_obj.active_topic)

    # 7. Stream response
    from rag.formatter import clean_response, conversational_fallback as formatter_fallback
    
    if chat_engine is not None and check_ollama_running():
        try:
            # Inject relevant matched memories into the query
            contextual_query = req.question
            if matched_memory:
                contextual_query = f"{matched_memory}\n\nStudent Query: {req.question}"
                
            response = chat_engine.stream_chat(contextual_query)
            
            async def event_generator():
                for token in clean_stream_generator(response.response_gen):
                    yield token
            return StreamingResponse(event_generator(), media_type="text/event-stream")
        except Exception as e:
            print(f"Error during streaming query chat: {e}")
            
    # Fallback stream if offline
    fallback_msg = formatter_fallback(req.question, filtered_hits)
    cleaned_fallback = clean_response(fallback_msg)
    
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
        recommendations=recommendations
    )

@app.get("/health")
async def health():
    return {"status": "ok", "chunks_indexed": collection.count(), "ollama_online": check_ollama_running()}
