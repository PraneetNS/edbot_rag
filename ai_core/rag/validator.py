import re
import logging
import requests
from ai_core.config import OLLAMA_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

class ContextValidator:
    """
    LLM-based validation layer that grades borderline retrieved chunks.
    Avoids latency on high/low confidence cases and performs quick YES/NO grading.
    """
    def __init__(self):
        pass

    def validate(self, question: str, context: str) -> bool:
        """
        Prompts local LLM to verify if the retrieved context answers the question.
        Returns True (YES) or False (NO).
        """
        try:
            # Quick check if Ollama is online
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1)
            if r.status_code != 200:
                logger.warning("Ollama offline. Context validation skipped (Auto-Approved).")
                return True
        except Exception:
            logger.warning("Ollama offline. Context validation skipped (Auto-Approved).")
            return True

        prompt = f"""Question:
{question}

Context:
{context}

Does context answer question?
Return only: YES or NO"""

        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 5  # Max tokens = 5 for maximum speed!
                }
            }
            logger.info("Running Ollama LLM Context Validation Judge...")
            r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=5)
            if r.status_code == 200:
                ans = r.json().get("response", "").strip().upper()
                clean_ans = re.sub(r'[^A-Z]', '', ans)
                
                if "YES" in clean_ans:
                    logger.info("LLM Judge classified context as: YES (Approved)")
                    return True
                else:
                    logger.info(f"LLM Judge classified context as: NO (Rejected, answer: {clean_ans})")
                    return False
        except Exception as e:
            logger.warning(f"Error during context validation: {e}. Auto-approving.")
            
        return True
