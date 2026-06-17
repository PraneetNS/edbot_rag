import os
import sys
import logging
import asyncio
import random
from pathlib import Path
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

import re

def classify_query_intent(query: str) -> str:
    q = query.lower().strip()
    
    # 1. code-request patterns
    code_keywords = [
        r"\bcode\b", r"\bcoding\b", r"\bprogram(s|ming)?\b", r"\bscript(s)?\b",
        r"\bwrite a function\b", r"\bimplement(ation)?\b", r"\bjava\b", r"\bpython\b",
        r"\bc\+\+\b", r"\bc#\b", r"\bjavascript\b", r"\bhtml\b", r"\bcss\b", r"\brust\b", r"\bgo(lang)?\b"
    ]
    if any(re.search(pat, q) for pat in code_keywords):
        return "code-request"
        
    # 2. visual-request patterns
    visual_keywords = [
        r"\broadmap(s)?\b", r"\bworkflow(s)?\b", r"\bchecklist(s)?\b", r"\btable(s)?\b",
        r"\bflowchart(s)?\b", r"\bdiagram(s)?\b", r"\bstep-by-step\b", r"\bsteps to\b",
        r"\bphases?\b", r"\bmilestones?\b", r"\bplan\b", r"\bcheck list\b"
    ]
    if any(re.search(pat, q) for pat in visual_keywords):
        return "visual-request"
        
    # 3. technical-concept patterns
    concept_keywords = [
        r"\btheorem\b", r"\bformula\b", r"\bequation\b", r"\bderivation\b", r"\bproof\b",
        r"\bthermodynamics?\b", r"\bentropy\b", r"\benthalpy\b", r"\bbernoulli\b", r"\bfourier\b",
        r"\blaplace\b", r"\bbode plot\b", r"\bnyquist\b", r"\bstability\b", r"\bcircuit\b",
        r"\btransistor\b", r"\bdiode\b", r"\bop-amp\b", r"\bconcrete\b", r"\bstress\b",
        r"\bstrain\b", r"\balgorithm\b", r"\bcomplexity\b", r"\bpointer\b", r"\bheap\b",
        r"\bhash\b", r"\brecursion\b", r"\btree\b", r"\bgraph\b", r"\bdbms\b", r"\bsql\b",
        r"\bnetworking\b", r"\bdsp\b", r"\bsignal processing\b", r"\bcontrol system\b",
        r"\bfluid mechanics\b", r"\bheat transfer\b", r"\bstructural analysis\b",
        r"\bkinematics\b", r"\bmachine design\b", r"\baerodynamics\b", r"\bhydraulics\b",
        r"\btransformer\b", r"\bgenerator\b", r"\bsolid mechanics\b", r"\bmechanics\b",
        r"\bsemiconductor\b", r"\bload balancing\b", r"\bcache\b", r"\bcaching\b",
        r"\bcompiler\b", r"\binterpreter\b", r"\bos\b", r"\boperating system\b", r"\bprocess scheduling\b",
        r"\bdeadlock\b", r"\bvirtual memory\b", r"\bconcurrency\b", r"\bthread\b", r"\bmutex\b"
    ]
    if any(re.search(pat, q) for pat in concept_keywords):
        return "technical-concept"
        
    # 4. conversational patterns
    conversational_keywords = [
        r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bgreet\b", r"\byo\b", r"\bsup\b",
        r"\bcareer\b", r"\bplacement(s)?\b", r"\bjob(s)?\b", r"\binternship(s)?\b",
        r"\bresume(s)?\b", r"\bmotivation\b", r"\banxious\b", r"\bstressed\b",
        r"\bdepressed\b", r"\bburnout\b", r"\bprepare\b", r"\bprep\b", r"\badvice\b",
        r"\btips\b", r"\binterview(s)?\b", r"\bmentor\b", r"\bguide\b", r"\bguidance\b",
        r"\bstudy tips\b", r"\bwhat is your name\b", r"\bwho are you\b"
    ]
    if any(re.search(pat, q) for pat in conversational_keywords):
        return "conversational"
        
    return "uncertain"

def find_sentence_boundary(text: str) -> int | None:
    """Finds the end index of the first valid sentence boundary in text, or None."""
    pos = 0
    while True:
        match = re.search(r'(?<=[.!?])(?:\s+|$)', text[pos:])
        if not match:
            return None
        
        # Absolute end index in the original text
        end_idx = pos + match.end()
        candidate = text[:end_idx].strip()
        
        # Check if the text up to the boundary ends with an abbreviation
        lower_candidate = candidate.lower().rstrip()
        is_abbrev = False
        
        # List of common abbreviations
        abbreviations = ["e.g.", "i.e.", "vs.", "approx.", "etc.", "dr.", "mr.", "ms.", "prof.", "al."]
        for abbrev in abbreviations:
            if lower_candidate.endswith(abbrev):
                is_abbrev = True
                break
                
        # Also check for single letter/number followed by dot (like initials or list numbers)
        if not is_abbrev:
            if re.search(r'\b[a-zA-Z]\.$', lower_candidate) or re.search(r'\b\d+\.$', lower_candidate):
                is_abbrev = True
                
        if not is_abbrev:
            return end_idx
            
        # Move search position past the current matched boundary start
        pos = pos + match.start() + 1

def split_into_sentences(text: str) -> list[str]:
    sentences = []
    current_text = text
    while True:
        boundary = find_sentence_boundary(current_text)
        if boundary is None:
            if current_text.strip():
                sentences.append(current_text.strip())
            break
        if boundary <= 0:
            break
        sentences.append(current_text[:boundary].strip())
        current_text = current_text[boundary:]
    return [s for s in sentences if s]


def _cap_sentence_words(sentence: str, max_words: int = 20) -> list[str]:
    """
    If a sentence exceeds max_words, split it at the last clause boundary
    (comma or semicolon) before the word limit.  If no boundary exists,
    yield the sentence intact so we never cut mid-word.
    Returns a list of one or two strings.
    """
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]

    # Find the best split point: last comma/semicolon up to word max_words
    prefix = " ".join(words[:max_words])
    # Walk backwards from max_words position looking for a clause boundary
    split_idx = -1
    for ch in (',', ';'):
        idx = prefix.rfind(ch)
        if idx > split_idx:
            split_idx = idx

    if split_idx == -1:
        # No clause boundary found — keep sentence intact to avoid awkward cuts
        return [sentence]

    part1 = sentence[:split_idx].strip()
    part2 = sentence[split_idx + 1:].strip()
    result = []
    if part1:
        result.append(part1)
    if part2:
        # Recursively cap part2 if it is still too long
        result.extend(_cap_sentence_words(part2, max_words))
    return result


class StreamingDualParser:
    def __init__(self):
        self.buffer = ""

    def feed(self, chunk: str) -> list[dict]:
        self.buffer += chunk
        events = []
        
        while True:
            speak_start = self.buffer.find("<speak>")
            show_match = re.search(r'<show(?:\s+[^>]*)?>', self.buffer)
            
            first_tag = None
            if speak_start != -1 and show_match:
                if speak_start < show_match.start():
                    first_tag = "speak"
                else:
                    first_tag = "show"
            elif speak_start != -1:
                first_tag = "speak"
            elif show_match:
                first_tag = "show"
                
            if first_tag == "speak":
                speak_end = self.buffer.find("</speak>", speak_start + 7)
                if speak_end != -1:
                    pre_text = self.buffer[:speak_start].strip()
                    if pre_text:
                        events.append({"type": "text", "content": pre_text})
                    
                    content = self.buffer[speak_start + 7:speak_end].strip()
                    if content:
                        events.append({"type": "text", "content": content})
                    
                    self.buffer = self.buffer[speak_end + 8:]
                    continue
                else:
                    pre_text = self.buffer[:speak_start].strip()
                    if pre_text:
                        events.append({"type": "text", "content": pre_text})
                    self.buffer = self.buffer[speak_start:]
                    break
                    
            elif first_tag == "show":
                start_idx = show_match.start()
                tag_content = show_match.group(0)
                end_tag_idx = show_match.end()
                
                show_end = self.buffer.find("</show>", end_tag_idx)
                if show_end != -1:
                    pre_text = self.buffer[:start_idx].strip()
                    if pre_text:
                        events.append({"type": "text", "content": pre_text})
                        
                    content = self.buffer[end_tag_idx:show_end].strip()
                    
                    type_match = re.search(r'type=["\']([^"\']+)["\']', tag_content)
                    lang_match = re.search(r'lang=["\']([^"\']+)["\']', tag_content)
                    show_type = type_match.group(1) if type_match else "code"
                    show_lang = lang_match.group(1) if lang_match else ""
                    
                    events.append({
                        "type": "show",
                        "show_type": show_type,
                        "lang": show_lang,
                        "content": content
                    })
                    
                    self.buffer = self.buffer[show_end + 7:]
                    continue
                else:
                    pre_text = self.buffer[:start_idx].strip()
                    if pre_text:
                        events.append({"type": "text", "content": pre_text})
                    self.buffer = self.buffer[start_idx:]
                    break
            else:
                partial_match = re.search(r'<[^>]*$', self.buffer)
                if partial_match:
                    pre_text = self.buffer[:partial_match.start()].strip()
                    if pre_text:
                        events.append({"type": "text", "content": pre_text})
                    self.buffer = self.buffer[partial_match.start():]
                else:
                    boundary = find_sentence_boundary(self.buffer)
                    if boundary is not None and boundary > 0:
                        sentence_text = self.buffer[:boundary].strip()
                        if sentence_text:
                            events.append({"type": "text", "content": sentence_text})
                        self.buffer = self.buffer[boundary:]
                    else:
                        pass
                break
                
        return events

    def finalize(self) -> list[dict]:
        events = []
        if self.buffer.strip():
            speak_start = self.buffer.find("<speak>")
            show_match = re.search(r'<show(?:\s+[^>]*)?>', self.buffer)
            
            if speak_start != -1:
                pre_text = self.buffer[:speak_start].strip()
                if pre_text:
                    events.append({"type": "text", "content": pre_text})
                content = self.buffer[speak_start + 7:].strip()
                if content:
                    events.append({"type": "text", "content": content})
            elif show_match:
                pre_text = self.buffer[:show_match.start()].strip()
                if pre_text:
                    events.append({"type": "text", "content": pre_text})
                tag_content = show_match.group(0)
                type_match = re.search(r'type=["\']([^"\']+)["\']', tag_content)
                lang_match = re.search(r'lang=["\']([^"\']+)["\']', tag_content)
                show_type = type_match.group(1) if type_match else "code"
                show_lang = lang_match.group(1) if lang_match else ""
                content = self.buffer[show_match.end():].strip()
                if content:
                    events.append({
                        "type": "show",
                        "show_type": show_type,
                        "lang": show_lang,
                        "content": content
                    })
            else:
                content = self.buffer.strip()
                if content:
                    events.append({"type": "text", "content": content})
        self.buffer = ""
        return events


# ── Follow-up question bank (Part 6) ─────────────────────────────────────────
FOLLOWUP_TRIGGERS: dict[str, list[str | None]] = {
    "dsa": [
        "Which data structure are you working on?",
        "Have you solved any similar problems before?",
    ],
    "placement": [
        "Which companies are you targeting?",
        "How much time do you have before placements?",
    ],
    "internship": [
        "Are you in 2nd or 3rd year?",
        "Have you applied anywhere yet?",
    ],
    "resume": [
        "Do you have any projects to put on it?",
        "Is this for internships or full-time?",
    ],
    "career": [
        "Do you know which domain interests you most?",
        "Are you leaning towards product or service companies?",
    ],
    "mindset": [
        "What specifically is making you feel this way?",
        None,   # silence is sometimes better
    ],
    "higher_studies": [
        "Are you targeting MS or MTech?",
        "Have you started preparing for GRE or GATE?",
    ],
}


def should_ask_followup(
    topic: str,
    turn_count_in_session: int,
    last_query_was_vague: bool,
) -> str | None:
    """
    Return a follow-up question string (or None) to append to the response.
    Rules:
      - Never on the 1st turn of a session.
      - Only every 3rd turn (turn_count % 3 == 0).
      - Pick a random non-None option from FOLLOWUP_TRIGGERS for the topic.
    """
    if turn_count_in_session < 2:
        return None
    if turn_count_in_session % 3 != 0:
        return None
    options = [o for o in FOLLOWUP_TRIGGERS.get(topic, []) if o is not None]
    if not options:
        return None
    return random.choice(options)

# Initialize the Ollama models
# qwen2.5:3b is primary, gemma2:9b is fallback
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

logger.info(f"Initializing Ollama LLMs with base URL: {ollama_base_url}")

llm_qwen = OllamaLLM(
    model="qwen2.5:3b",
    base_url=ollama_base_url,
    temperature=0.3,
    top_p=0.85,
    num_predict=512,
    repeat_penalty=1.15,
    stop=["Student:", "User:", "Human:", "\n\n\n"],
    timeout=20,   # increased timeout for longer generations
)

llm_gemma = OllamaLLM(
    model="gemma2:9b",
    base_url=ollama_base_url,
    temperature=0.3,
    top_p=0.85,
    num_predict=512,
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
        "sad", "fear", "fail", "roadmaps",
        # Higher studies & career direction
        "ms", "masters", "master", "mtech", "m.tech", "mba", "phd", "gate", "gre", "gmat",
        "abroad", "usa", "germany", "canada", "university", "higher studies", "higher_studies",
        "job", "work", "salary", "package", "placed", "joining", "service", "product",
        "startup", "company", "corporate", "fresher", "experienced",
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

async def call_llm_with_timeout(prompt: str) -> str:
    for attempt in range(2):  # max 2 attempts total
        try:
            return await llm_qwen.ainvoke(prompt)
        except Exception as e:
            print(f"[LLM ATTEMPT {attempt+1} FAILED] {e}")
            if attempt == 0:
                await asyncio.sleep(0.5)  # brief pause before retry
                continue
            # Both attempts failed — try gemma fallback
            try:
                return await llm_gemma.ainvoke(prompt)
            except Exception as e2:
                print(f"[LLM GEMMA FALLBACK FAILED] {e2}")
                return "Give me a moment and try again."

async def generate_response_with_routing(
    query: str,
    session_id: str = "default",
    pre_retrieved_docs: list = None,
    student_id: str = "anonymous",
) -> tuple[str, str]:
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

    # ── analytics helpers imported lazily to avoid circular imports ────────
    def _fire_track(guard: str | None, resp: str, is_v: bool = False, is_r: bool = False, sl: bool = False):
        """Schedule a background analytics write without blocking."""
        try:
            from edmentor.student_tracker import track_turn
            asyncio.create_task(track_turn(
                student_id=student_id,
                session_id=session_id,
                query=query,
                response=resp,
                guard_fired=guard,
                timing={
                    "retrieval_ms": 0,
                    "llm_ms":       0,
                    "tts_ms":       0,
                    "total_ms":     int((time.time() - t0) * 1000),
                },
                retrieval_info={"chunks_found": 0, "top_score": None},
                is_vague=is_v,
                is_repeat=is_r,
                stuck_loop=sl,
            ))
        except Exception as _te:
            logger.warning(f"[Tracker] fire_track skipped: {_te}")

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
        _fire_track("jailbreak_guard", JAILBREAK_RESPONSE)
        return JAILBREAK_RESPONSE, "jailbreak_guard"

    # 3. check_identity(text) -> True -> return IDENTITY_RESPONSE (with reveal logic check)
    if check_identity(cleaned):
        logger.info("Query intercepted by check_identity.")
        # If it's a reveal query (prompt reveal, RAG architecture, model info), return reveal response
        if any(kw in cleaned.lower() for kw in ["system prompt", "using rag", "rag", "model", "reveal", "instructions"]):
            _id_resp = "I am Edmentor, your engineering mentor. What are you working on?"
            _fire_track("identity_guard", _id_resp)
            return _id_resp, "identity_guard"
        _fire_track("identity_guard", IDENTITY_RESPONSE)
        return IDENTITY_RESPONSE, "identity_guard"

    # 4. check_greeting(text) -> True -> return GREETING_RESPONSE
    if check_greeting(cleaned):
        logger.info("Query intercepted by check_greeting.")
        _fire_track("first_turn_greeting", GREETING_RESPONSE)
        return GREETING_RESPONSE, "first_turn_greeting"

    # 5. is_vague(text) -> True -> return VAGUE_RESPONSE
    # NOTE: vague check MUST run before domain guard — a query like 'ok'
    # has no domain keywords but is vague, not off-domain.
    _query_is_vague = is_vague(cleaned)
    if _query_is_vague:
        logger.info("Query blocked by is_vague.")
        _fire_track("vague_guard", VAGUE_RESPONSE, is_v=True)
        return VAGUE_RESPONSE, "vague_guard"

    # 4.5. Domain Guard Check (off-domain redirection)
    _domain_msg = "That is outside what I focus on. Ask me about DSA, placements, internships, resume, or your career."
    if is_actually_off_domain(rewritten):
        _fire_track("domain_guard", _domain_msg)
        return _domain_msg, "domain_guard"

    is_blocked, guard_response, reason = guard.check(rewritten)
    if is_blocked:
        if reason in ("blocklist", "no_keyword_match"):
            _fire_track("domain_guard", _domain_msg)
            return _domain_msg, "domain_guard"

    t_guard = time.time()

    # 5.5 Classify query intent and route RAG / Direct LLM
    intent = classify_query_intent(cleaned)
    logger.info(f"[Intent Routing] Query classified as: {intent}")

    # 6. Retrieve relevant documents from ChromaDB only for technical-concept or uncertain
    if intent in ("technical-concept", "uncertain"):
        if pre_retrieved_docs is not None:
            docs = pre_retrieved_docs
            logger.info("Using pre-retrieved docs from async overlap.")
        else:
            docs = await retrieve(cleaned)
    else:
        logger.info(f"Bypassing RAG for intent: {intent}")
        docs = []

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
    logger.info("Invoking primary Qwen-2.5:3b model...")
    response_text = await call_llm_with_timeout(formatted_prompt)

    t_llm = time.time()

    word_count_out = len(response_text.split())
    print(f"[LLM] Ollama — input_docs={len(docs)} output_words={word_count_out} llm_ms={1000*(t_llm-t_retrieval):.0f}")

    # Extract speak content to check for loops and follow-ups
    speak_parts = []
    for m in re.finditer(r'<speak>(.*?)</speak>', response_text, re.DOTALL):
        speak_parts.append(m.group(1).strip())
        
    if not ("<speak>" in response_text or "<show" in response_text):
        speak_text = sanitise(response_text)
    else:
        speak_text = " ".join(speak_parts).strip()

    # Detect stuck loop
    _is_stuck = is_stuck_loop(session_id, speak_text)
    if _is_stuck:
        response_text = (
            "<speak>Let me come at this differently. "
            "What specific part are you stuck on right now?</speak>"
        )
        speak_text = "Let me come at this differently. What specific part are you stuck on right now?"

    # Save speak text to last_responses
    get_last_responses(session_id).append(speak_text)

    t_done = time.time()

    # Log timings
    retrieval_ms = int(1000 * (t_retrieval - t_guard))
    llm_ms       = int(1000 * (t_llm - t_retrieval))
    total_ms     = int(1000 * (t_done - t0))
    print(
        f"[TIMING] session={session_id[:8]} "
        f"guard={1000*(t_guard-t0):.0f}ms "
        f"retrieval={retrieval_ms}ms "
        f"llm={llm_ms}ms "
        f"total={total_ms}ms"
    )

    # ── Part 6: Follow-up question injection ─────────────────────────────────
    try:
        from db import get_session_turn_count
        from edmentor.student_tracker import classify_topic
        _topic_for_fu = classify_topic(cleaned)
        _turn_count   = get_session_turn_count(session_id)
        _followup     = should_ask_followup(_topic_for_fu, _turn_count + 1, _query_is_vague)
        if _followup and len(speak_text.split()) < 45:
            if "<speak>" in response_text:
                last_speak_idx = response_text.rfind("</speak>")
                if last_speak_idx != -1:
                    response_text = (
                        response_text[:last_speak_idx] +
                        " " + _followup +
                        response_text[last_speak_idx:]
                    )
            else:
                response_text = response_text.rstrip() + " " + _followup
    except Exception as _fu_err:
        logger.warning(f"[Follow-up] skipped: {_fu_err}")

    # ── Part 2: Analytics tap (non-blocking) ─────────────────────────────────
    try:
        from edmentor.student_tracker import track_turn
        asyncio.create_task(track_turn(
            student_id=student_id,
            session_id=session_id,
            query=cleaned,
            response=response_text,
            guard_fired=None,
            timing={
                "retrieval_ms": retrieval_ms,
                "llm_ms":       llm_ms,
                "tts_ms":       0,
                "total_ms":     total_ms,
            },
            retrieval_info={
                "chunks_found": len(docs),
                "top_score":    None,
            },
            is_vague=_query_is_vague,
            is_repeat=False,
            stuck_loop=_is_stuck,
        ))
    except Exception as _track_err:
        logger.warning(f"[Tracker] create_task skipped: {_track_err}")

    # 10. Save query & response to memory
    edmentor_memory.save_turn(session_id, cleaned, response_text)

    # 11. Return response
    return response_text, intent


async def generate_stream_with_routing(
    query: str,
    session_id: str = "default",
    student_id: str = "anonymous",
):
    """
    LangChain-based EduMentor request flow (streaming version).
    Routes queries through the exact 11-step request handler call sequence
    and yields dictionary events for real-time text/audio streaming.
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

    logger.info(f"Streaming query: {query} [Session: {session_id}]")

    def _fire_track(guard: str | None, resp: str, is_v: bool = False, is_r: bool = False, sl: bool = False):
        """Schedule a background analytics write without blocking."""
        try:
            from edmentor.student_tracker import track_turn
            asyncio.create_task(track_turn(
                student_id=student_id,
                session_id=session_id,
                query=query,
                response=resp,
                guard_fired=guard,
                timing={
                    "retrieval_ms": 0,
                    "llm_ms":       0,
                    "tts_ms":       0,
                    "total_ms":     int((time.time() - t0) * 1000),
                },
                retrieval_info={"chunks_found": 0, "top_score": None},
                is_vague=is_v,
                is_repeat=is_r,
                stuck_loop=sl,
            ))
        except Exception as _te:
            logger.warning(f"[Tracker] fire_track skipped: {_te}")

    # 1. clean_input(text) -> None -> return FALLBACK_RESPONSE
    cleaned = clean_input(query)
    if cleaned is None:
        logger.info("Query blocked by clean_input / STT noise filter.")
        yield {"type": "routing", "content": "stt_noise"}
        yield {"type": "text", "content": FALLBACK_RESPONSE}
        _fire_track("stt_noise", FALLBACK_RESPONSE)
        return

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
        yield {"type": "routing", "content": "jailbreak_guard"}
        yield {"type": "text", "content": JAILBREAK_RESPONSE}
        _fire_track("jailbreak_guard", JAILBREAK_RESPONSE)
        return

    # 3. check_identity(text) -> True -> return IDENTITY_RESPONSE (with reveal logic check)
    if check_identity(cleaned):
        logger.info("Query intercepted by check_identity.")
        yield {"type": "routing", "content": "identity_guard"}
        if any(kw in cleaned.lower() for kw in ["system prompt", "using rag", "rag", "model", "reveal", "instructions"]):
            _id_resp = "I am Edmentor, your engineering mentor. What are you working on?"
            yield {"type": "text", "content": _id_resp}
            _fire_track("identity_guard", _id_resp)
        else:
            yield {"type": "text", "content": IDENTITY_RESPONSE}
            _fire_track("identity_guard", IDENTITY_RESPONSE)
        return

    # 4. check_greeting(text) -> True -> return GREETING_RESPONSE
    if check_greeting(cleaned):
        logger.info("Query intercepted by check_greeting.")
        yield {"type": "routing", "content": "first_turn_greeting"}
        yield {"type": "text", "content": GREETING_RESPONSE}
        _fire_track("first_turn_greeting", GREETING_RESPONSE)
        return

    # 5. is_vague(text) -> True -> return VAGUE_RESPONSE
    _query_is_vague = is_vague(cleaned)
    if _query_is_vague:
        logger.info("Query blocked by is_vague.")
        yield {"type": "routing", "content": "vague_guard"}
        yield {"type": "text", "content": VAGUE_RESPONSE}
        _fire_track("vague_guard", VAGUE_RESPONSE, is_v=True)
        return

    # 4.5. Domain Guard Check (off-domain redirection)
    _domain_msg = "That is outside what I focus on. Ask me about DSA, placements, internships, resume, or your career."
    if is_actually_off_domain(rewritten):
        yield {"type": "routing", "content": "domain_guard"}
        yield {"type": "text", "content": _domain_msg}
        _fire_track("domain_guard", _domain_msg)
        return

    is_blocked, guard_response, reason = guard.check(rewritten)
    if is_blocked:
        if reason in ("blocklist", "no_keyword_match"):
            yield {"type": "routing", "content": "domain_guard"}
            yield {"type": "text", "content": _domain_msg}
            _fire_track("domain_guard", _domain_msg)
            return

    t_guard = time.time()

    # 5.5 Classify query intent and route RAG / Direct LLM
    intent = classify_query_intent(cleaned)
    logger.info(f"[Intent Routing Stream] Query classified as: {intent}")

    # 6. Retrieve relevant documents from ChromaDB only for technical-concept or uncertain
    if intent in ("technical-concept", "uncertain"):
        docs = await retrieve(cleaned)
    else:
        logger.info(f"Bypassing RAG for streaming intent: {intent}")
        docs = []

    t_retrieval = time.time()

    # Get conversation history & profile from memory
    mem = edmentor_memory.get_or_create_session(session_id)
    history_vars = mem.load_memory_variables({})
    chat_history = history_vars.get("chat_history", [])
    profile = edmentor_memory.get_profile(session_id)

    # 7. Build prompt
    prompt_template = build_prompt(docs, chat_history, cleaned, profile)
    messages = prompt_template.format_messages(chat_history=chat_history, question=cleaned)
    
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

    # 8. Call LLM (Ollama client invocation with streaming)
    logger.info("Invoking primary Qwen-2.5:3b model in streaming mode...")
    yield {"type": "routing", "content": intent}
    
    t_llm_start = time.time()
    
    parser = StreamingDualParser()
    yielded_speak_sentences = []
    full_llm_response = ""
    show_was_yielded = False  # Track whether any show block was emitted
    
    async def get_llm_chunks():
        try:
            async for chunk in llm_qwen.astream(formatted_prompt):
                yield chunk
        except Exception as e:
            logger.error(f"[Qwen stream failed] {e}. Falling back to Gemma.")
            try:
                async for chunk in llm_gemma.astream(formatted_prompt):
                    yield chunk
            except Exception as e2:
                logger.error(f"[Gemma stream failed] {e2}")
                yield "Give me a moment and try again."

    async for chunk in get_llm_chunks():
        full_llm_response += chunk
        events = parser.feed(chunk)
        for event in events:
            if event["type"] == "text":
                sentences = split_into_sentences(event["content"])
                for raw_s in sentences:
                    for sentence in _cap_sentence_words(raw_s):
                        yield {"type": "text", "content": sentence}
                        yielded_speak_sentences.append(sentence)
            elif event["type"] == "show" and intent != "technical-concept":
                show_was_yielded = True
                yield {
                    "type": "show",
                    "show_type": event["show_type"],
                    "lang": event["lang"],
                    "content": event["content"]
                }

    # Finalize parser
    events = parser.finalize()
    for event in events:
        if event["type"] == "text":
            sentences = split_into_sentences(event["content"])
            for raw_s in sentences:
                for sentence in _cap_sentence_words(raw_s):
                    yield {"type": "text", "content": sentence}
                    yielded_speak_sentences.append(sentence)
        elif event["type"] == "show" and intent != "technical-concept":
            show_was_yielded = True
            yield {
                "type": "show",
                "show_type": event["show_type"],
                "lang": event["lang"],
                "content": event["content"]
            }

    # ── Show-block fallback for visual-request ────────────────────────────────
    # If the LLM was supposed to produce a show block (visual-request) but did
    # not include any <show> tags, synthesise one from the raw LLM text so the
    # student at least sees the content in the chat even when the model fails
    # to follow the tag instruction.
    if intent == "visual-request" and not show_was_yielded and full_llm_response.strip():
        # Strip any residual speak tags and leading/trailing whitespace
        raw_content = re.sub(r"</?speak>", "", full_llm_response).strip()
        # Remove any partial/malformed show tags that were never closed
        raw_content = re.sub(r"<show[^>]*>", "", raw_content)
        raw_content = re.sub(r"</show>", "", raw_content).strip()
        if raw_content:
            logger.warning(
                "[show_fallback] visual-request produced no show block — "
                "emitting fallback show event from raw LLM text."
            )
            yield {
                "type": "show",
                "show_type": "roadmap",
                "lang": "",
                "content": raw_content,
            }

    speak_response = " ".join(yielded_speak_sentences)

    # Detect stuck loop based on speak text
    _is_stuck = is_stuck_loop(session_id, speak_response)
    if _is_stuck:
        full_llm_response = (
            "<speak>Let me come at this differently. "
            "What specific part are you stuck on right now?</speak>"
        )
        speak_response = "Let me come at this differently. What specific part are you stuck on right now?"

    # Save to last_responses
    get_last_responses(session_id).append(speak_response)

    # ── Part 6: Follow-up question injection ─────────────────────────────────
    try:
        from db import get_session_turn_count
        from edmentor.student_tracker import classify_topic
        _topic_for_fu = classify_topic(cleaned)
        _turn_count   = get_session_turn_count(session_id)
        _followup     = should_ask_followup(_topic_for_fu, _turn_count + 1, _query_is_vague)
        if _followup and len(speak_response.split()) < 45:
            yield {"type": "text", "content": _followup}
            yielded_speak_sentences.append(_followup)
            speak_response = " ".join(yielded_speak_sentences)
            if "<speak>" in full_llm_response:
                last_speak_idx = full_llm_response.rfind("</speak>")
                if last_speak_idx != -1:
                    full_llm_response = (
                        full_llm_response[:last_speak_idx] +
                        " " + _followup +
                        full_llm_response[last_speak_idx:]
                    )
            else:
                full_llm_response = full_llm_response.rstrip() + " " + _followup
    except Exception as _fu_err:
        logger.warning(f"[Follow-up stream] skipped: {_fu_err}")

    # ── Part 2: Analytics tap (non-blocking) ─────────────────────────────────
    try:
        from edmentor.student_tracker import track_turn
        asyncio.create_task(track_turn(
            student_id=student_id,
            session_id=session_id,
            query=cleaned,
            response=full_llm_response,
            guard_fired=None,
            timing={
                "retrieval_ms": int(1000 * (t_retrieval - t_guard)),
                "llm_ms":       int(1000 * (time.time() - t_llm_start)),
                "tts_ms":       0,
                "total_ms":     int(1000 * (time.time() - t0)),
            },
            retrieval_info={
                "chunks_found": docs_len if 'docs_len' in locals() else len(docs),
                "top_score":    None,
            },
            is_vague=_query_is_vague,
            is_repeat=False,
            stuck_loop=_is_stuck,
        ))
    except Exception as _track_err:
        logger.warning(f"[Tracker stream] create_task failed: {_track_err}")

    # 10. Save query & response to memory
    edmentor_memory.save_turn(session_id, cleaned, full_llm_response)
