import os
import sys
import logging
import asyncio
from pathlib import Path
from langchain_community.llms import Ollama

logger = logging.getLogger(__name__)

# Initialize the Ollama models
# qwen2.5:7b is primary, gemma2:9b is fallback
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

logger.info(f"Initializing Ollama LLMs with base URL: {ollama_base_url}")

llm_qwen = Ollama(
    model="qwen2.5:3b",
    base_url=ollama_base_url,
    temperature=0.3,
    top_p=0.85,
    num_predict=120,
    repeat_penalty=1.15,
    stop=["Student:", "User:", "Human:", "\n\n\n"],
)

llm_gemma = Ollama(
    model="gemma2:9b",
    base_url=ollama_base_url,
    temperature=0.3,
    top_p=0.85,
    num_predict=120,
    repeat_penalty=1.15,
    stop=["Student:", "User:", "Human:", "\n\n\n"],
)

def is_actually_off_domain(query: str) -> bool:
    """Helper to detect off-domain queries in python before calling a potentially offline LLM."""
    from edmentor.intent_router import ON_DOMAIN_KEYWORDS
    q = query.lower().strip()
    
    # Generic allowed terms including companies and student emotional support keywords
    allowed_keywords = ON_DOMAIN_KEYWORDS + [
        "dsa", "roadmap", "study", "learn", "mentor", "exam", "engineering", "btech", "be", 
        "accenture", "amazon", "google", "tcs", "cognizant", "infosys", "wipro",
        "feel", "behind", "batch", "anxious", "scared", "stressed", "depressed", 
        "burnout", "motivated", "motivation", "pressure", "struggle", "worry", 
        "sad", "fear", "fail", "roadmaps"
    ]
    return not any(kw in q for kw in allowed_keywords)

async def check_ollama_health() -> bool:
    try:
        # Call Ollama with a one-token ping
        llm_qwen.invoke("ping")
        return True
    except Exception:
        return False

def is_stuck_loop(session_id: str, response: str) -> bool:
    from edmentor.memory import get_last_responses
    from difflib import SequenceMatcher
    import re
    last = get_last_responses(session_id)
    if len(last) < 2:
        return False
    # Normalize: lowercase, strip punctuation
    norm = lambda t: re.sub(r'[^\w\s]', '', t.lower().strip())
    normed = norm(response)
    # If 2 of last 3 responses are >85% similar to current
    matches = sum(
        1 for r in last
        if SequenceMatcher(None, norm(r), normed).ratio() > 0.85
    )
    return matches >= 2

async def generate_response_with_routing(query: str, session_id: str = "default", pre_retrieved_docs: list = None) -> tuple[str, str]:
    """
    LangChain-based EduMentor request flow.
    Routes queries through the exact 11-step request handler call sequence.
    """
    from edmentor.input_guard import clean_input, check_jailbreak, is_vague, JAILBREAK_RESPONSE, VAGUE_RESPONSE, FALLBACK_RESPONSE
    from edmentor.guard import check_identity, check_greeting, IDENTITY_RESPONSE, GREETING_RESPONSE, guard
    from edmentor.rag_engine import retrieve
    from edmentor.prompt_builder import build_prompt
    from edmentor.output_sanitiser import sanitise
    from edmentor.memory import memory as edmentor_memory, touch_session, get_last_responses
    from datetime import datetime
    import re
    import time

    # Touch session to update last active timestamp
    touch_session(session_id)
    t0 = time.time()

    logger.info(f"Routing query: {query} [Session: {session_id}]")

    # 1. clean_input(text) -> None -> return FALLBACK_RESPONSE
    cleaned = clean_input(query)
    if cleaned is None:
        logger.info("Query blocked by clean_input / STT noise filter.")
        return FALLBACK_RESPONSE, "stt_noise"

    # Query expansion using conversation history
    mem = edmentor_memory.get_or_create_session(session_id)
    history_messages = mem.chat_memory.messages
    
    rewritten = cleaned
    if history_messages:
        # Check if the query is referential or short
        q_lower = cleaned.lower()
        referential_indicators = {
            "that", "it", "this", "those", "them", "these", "there", 
            "do so", "do it", "exercises", "examples", "more", "explain", "code", "coding"
        }
        words = set(re.findall(r'\b\w+\b', q_lower))
        is_referential = bool(words & referential_indicators)
        is_short = len(words) < 5
        
        if is_referential or is_short:
            prev_user = None
            for msg in reversed(history_messages):
                if msg.type == "human":
                    prev_user = msg.content
                    break
            if prev_user:
                rewritten = f"{cleaned} {prev_user}"
                logger.info(f"[Query Expansion] Expanded '{cleaned}' to '{rewritten}'")

    # 2. check_jailbreak(text) -> True -> return JAILBREAK_RESPONSE
    if check_jailbreak(cleaned):
        logger.info("Query blocked by check_jailbreak.")
        return JAILBREAK_RESPONSE, "jailbreak_guard"

    # 3. check_identity(text) -> True -> return IDENTITY_RESPONSE (with reveal logic check)
    if check_identity(cleaned):
        logger.info("Query intercepted by check_identity.")
        # If it's a reveal query (prompt reveal, RAG architecture, model info), return reveal response
        if any(kw in cleaned.lower() for kw in ["system prompt", "using rag", "rag", "model", "reveal", "instructions"]):
            return "I am Edmentor, your engineering mentor. What are you working on?", "identity_guard"
        return IDENTITY_RESPONSE, "identity_guard"

    # 4. check_greeting(text) -> True -> return GREETING_RESPONSE
    if check_greeting(cleaned):
        logger.info("Query intercepted by check_greeting.")
        return GREETING_RESPONSE, "first_turn_greeting"

    # 4.5. Domain Guard Check (off-domain redirection)
    if is_actually_off_domain(rewritten):
        return "That is outside what I focus on. Ask me about DSA, placements, internships, resume, or your career.", "domain_guard"
        
    is_blocked, guard_response, reason = guard.check(rewritten)
    if is_blocked:
        if reason in ("blocklist", "no_keyword_match"):
            return "That is outside what I focus on. Ask me about DSA, placements, internships, resume, or your career.", "domain_guard"

    # 5. is_vague(text) -> True -> return VAGUE_RESPONSE
    if is_vague(cleaned):
        logger.info("Query blocked by is_vague.")
        return VAGUE_RESPONSE, "vague_guard"

    t_guard = time.time()

    # 6. Retrieve relevant documents from ChromaDB (LangChain retriever)
    if pre_retrieved_docs is not None:
        docs = pre_retrieved_docs
        logger.info("Using pre-retrieved docs from async overlap.")
    else:
        docs = await retrieve(cleaned)

    t_retrieval = time.time()

    # Get conversation history & profile from memory
    mem = edmentor_memory.get_or_create_session(session_id)
    history_vars = mem.load_memory_variables({})
    chat_history = history_vars.get("chat_history", [])
    profile = edmentor_memory.get_profile(session_id)

    # 7. Build prompt via build_prompt()
    prompt_template = build_prompt(docs, chat_history, cleaned, profile)
    messages = prompt_template.format_messages(chat_history=chat_history, question=cleaned)
    
    # Format messages array to standard text for Ollama LLM
    formatted_prompt = ""
    for msg in messages:
        if msg.type == "system":
            formatted_prompt += msg.content + "\n\n"
        elif msg.type == "human":
            formatted_prompt += f"Student: {msg.content}\n"
        elif msg.type == "ai":
            formatted_prompt += f"Mentor: {msg.content}\n"
            
    if not formatted_prompt.endswith("Mentor: "):
        formatted_prompt += "Mentor: "

    # 8. Call LLM (Ollama client invocation with fallback)
    response_text = ""
    try:
        try:
            logger.info("Invoking primary Qwen-2.5:3b model...")
            # Run standard LLM completion
            response_text = await llm_qwen.ainvoke(formatted_prompt)
        except Exception as e:
            logger.warning(f"Ollama qwen2.5:3b failed: {e}. Falling back to gemma2:9b...")
            response_text = await llm_gemma.ainvoke(formatted_prompt)
    except Exception as e:
        # Log the error with timestamp
        print(f"[LLM ERROR] {datetime.now().isoformat()} — {e}")
        response_text = (
            "I am having a bit of trouble right now. "
            "Give me a moment and try again."
        )

    t_llm = time.time()

    # 9. Sanitise output via sanitise()
    sanitised_response = sanitise(response_text)

    # Detect stuck loop
    if is_stuck_loop(session_id, sanitised_response):
        sanitised_response = (
            "Let me come at this differently. "
            "What specific part are you stuck on right now?"
        )

    # Save to last_responses
    get_last_responses(session_id).append(sanitised_response)

    t_done = time.time()

    # Log timings
    print(
        f"[TIMING] session={session_id[:8]} "
        f"guard={1000*(t_guard-t0):.0f}ms "
        f"retrieval={1000*(t_retrieval-t_guard):.0f}ms "
        f"llm={1000*(t_llm-t_retrieval):.0f}ms "
        f"sanitise={1000*(t_done-t_llm):.0f}ms "
        f"total={1000*(t_done-t0):.0f}ms"
    )

    # 10. Save query & response to memory
    edmentor_memory.save_turn(session_id, cleaned, sanitised_response)

    # 11. Return response
    return sanitised_response, "llm_generation"
