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
    - If it's a greeting, returns the first turn greeting behavior response.
    - Otherwise, runs direct RAG response pipeline via rag_retrieve_and_respond.
    """
    # 1. First-turn greeting check
    q_clean = query.strip().lower().rstrip("?.!")
    greetings = {"hello", "hi", "hey", "hola", "greetings", "good morning", "good evening", "yo", "sup", "heyy", "heyyy"}
    
    if not q_clean or q_clean in greetings or len(q_clean.split()) == 1:
        greeting_response = (
            "Hey. Tell me what you are working on or stuck on right now. "
            "DSA, placements, resume, internships, projects — whatever it is, let's get into it."
        )
        return greeting_response, "first_turn_greeting"

    # 2. Retrieve from RAG directly and return the mentor text
    from edmentor.rag_engine import rag_retrieve_and_respond
    response = await rag_retrieve_and_respond(query, None, None)
    return response, "rag_direct"
