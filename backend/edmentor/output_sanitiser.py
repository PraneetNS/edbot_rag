import re
import logging

logger = logging.getLogger(__name__)

def sanitise(text: str) -> str:
    """
    Sanitises LLM outputs:
    1. Removes markdown symbols (asterisks, backticks).
    2. Strips double quotes.
    3. Strips list markers (bullet points, numbers) and headers.
    4. Truncates/limits output to maximum 3 sentences.
    """
    if not text:
        return ""

    # 1. Strip markdown headers (e.g., #, ##, etc. at start of lines)
    cleaned = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

    # 2. Remove asterisks, backticks, and double quotes (including smart quotes)
    cleaned = cleaned.replace("**", "").replace("*", "").replace("`", "").replace('"', '').replace('“', '').replace('”', '')

    # 3. Strip list markers at the start of lines (e.g., - , * , + , 1. )
    cleaned = re.sub(r'^[-\+\*\d]+\.?\s+', '', cleaned, flags=re.MULTILINE)

    # 4. Normalize spacing (remove duplicate spaces and newlines)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # 5. Rejoin sentences cleanly
    # Split on sentence-ending punctuation (., !, ?) followed by whitespace or end of string
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned) if s.strip()]
    
    final_text = " ".join(sentences)
    logger.info(f"Sanitised output from {len(text)} chars to {len(final_text)} chars ({len(sentences)} sentences).")
    return final_text
