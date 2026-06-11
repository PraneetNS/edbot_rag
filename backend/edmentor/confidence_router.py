import os
import sys
import logging
import asyncio

# Ensure edmentor is in sys.path
from pathlib import Path
EDMENTOR_DIR = Path(__file__).resolve().parent
if str(EDMENTOR_DIR) not in sys.path:
    sys.path.append(str(EDMENTOR_DIR))

logger = logging.getLogger(__name__)

async def generate_response_with_routing(query: str, session_id: str = "default") -> tuple[str, str]:
    """
    Routes query to local response model / RAG retrieval.

    Gate ⓪ — STT noise filter (input_guard): catches "umm", "uh", etc.
    Gate ① — First-turn greeting: simple greetings get a warm entry response.
    Gate ② — RAG direct: everything else goes to rag_retrieve_and_respond.
    """
    # Gate ⓪: STT noise filter
    from edmentor.input_guard import check_stt_noise
    is_noise, noise_response = check_stt_noise(query)
    if is_noise:
        return noise_response, "stt_noise"

    # Gate ①: First-turn greeting check
    q_clean = query.strip().lower().rstrip("?.!")
    greetings = {
        "hello", "hi", "hey", "hola", "greetings",
        "good morning", "good evening", "yo", "sup", "heyy", "heyyy",
    }

    if not q_clean or q_clean in greetings or len(q_clean.split()) == 1:
        greeting_response = (
            "Hey. Tell me what you are working on or stuck on right now. "
            "DSA, placements, resume, internships, projects — whatever it is, let's get into it."
        )
        return greeting_response, "first_turn_greeting"

    # Gate ②: Retrieve from RAG directly and return the mentor text
    from edmentor.memory import memory as edmentor_memory
    import re
    
    last_turns = edmentor_memory.get_last_turns(session_id, n=1)
    rewritten_query = query
    if last_turns:
        q_lower = query.lower()
        referential_indicators = {
            "that", "it", "this", "those", "them", "these", "there", 
            "do so", "do it", "exercises", "examples", "more", "explain", "code", "coding"
        }
        words = set(re.findall(r'\b\w+\b', q_lower))
        is_referential = bool(words & referential_indicators)
        is_short = len(words) < 5
        
        if is_referential or is_short:
            prev_user = last_turns[-1]["user"]
            rewritten_query = f"{query} {prev_user}"
            logger.info(f"[Query Expansion] Expanded '{query}' to '{rewritten_query}'")

    from edmentor.rag_engine import rag_retrieve_and_respond
    response = await rag_retrieve_and_respond(rewritten_query, None, None)
    return response, "rag_direct"
