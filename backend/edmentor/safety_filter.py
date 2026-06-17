"""
edmentor/safety_filter.py
─────────────────────────
Post-processing pipeline for every LLM output.

Executed in order:
  Step 1 — Strip template leaks
  Step 2 — Strip markdown
  Step 3 — Sentence-boundary word cap (max 60 words) via cap_for_voice()
  Step 4 — Empty guard
"""

import re

MAX_WORDS = 200


# ── Step 3 helper ─────────────────────────────────────────────────────────────

def cap_for_voice(text: str, max_words: int = MAX_WORDS) -> str:
    """
    Truncate text to max_words while respecting sentence boundaries.
    Prefers cutting at a sentence-end (.!?), then at a comma,
    then hard-truncates if neither is available.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = ' '.join(words[:max_words])
    match = re.search(r'[.!?][^.!?]*$', truncated)
    if match:
        return truncated[:match.start() + 1].strip()
    last_comma = truncated.rfind(',')
    if last_comma > len(truncated) // 2:
        return truncated[:last_comma].strip()
    return truncated.strip()


# ── Main filter ───────────────────────────────────────────────────────────────

def edumentor_filter(text: str, max_words: int = MAX_WORDS) -> str:
    """
    Full post-processing pipeline for EduMentor LLM output.
    Safe to call on any string; returns a clean, voice-ready response.
    """
    if not text or not text.strip():
        return "Can you tell me a bit more about what you're working on?"

    # ── Step 1: Strip template leaks ─────────────────────────────────────────
    template_patterns = [
        r'\*(EduMentor.*?)\*',
        r'###\s.*?\n',
        r'\*(Retrieved from.*?)\*',
        r'Knowledge context.*?Student says:',
        r'Reply as EduMentor.*',
    ]
    for p in template_patterns:
        text = re.sub(p, '', text, flags=re.DOTALL | re.IGNORECASE)

    # ── Step 2: Strip markdown ────────────────────────────────────────────────
    # Bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    # Inline code
    text = re.sub(r'`[^`]+`', '', text)
    # Fenced code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    # Headers (# ## ### ####)
    text = re.sub(r'(?m)^#{1,4}\s.*$', '', text)
    # Bullet points
    text = re.sub(r'(?m)^\s*[-*•]\s+', '', text)
    text = re.sub(r'\n\s*[-*•]\s+', ' ', text)
    # Numbered lists
    text = re.sub(r'(?m)^\s*\d+\.\s+', '', text)

    # Extra: strip any remaining bold/italic markers and standalone #
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.*?)_{1,3}', r'\1', text)
    text = text.replace('*', '').replace('#', '')

    # Strip parenthetical source refs
    text = re.sub(r'\(Source:.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Source:.*?\]', '', text, flags=re.IGNORECASE)

    # Remove AI filler phrases (retained from original functionality for test compatibility)
    # Remove fillers
    fillers = [
        "great question", "certainly", "sure!", "absolutely", 
        "of course", "i'd be happy to", "as an ai", "sure"
    ]
    for f in fillers:
        pattern = r'\b' + re.escape(f) + r'\b[,.!?]?'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Remove double inverted commas
    text = text.replace('"', '').replace('“', '').replace('”', '')

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[,\s.!?]+', '', text).strip()

    # ── Step 3: Sentence-boundary word cap ───────────────────────────────────
    text = cap_for_voice(text, max_words=max_words)

    # ── Step 4: Empty guard ───────────────────────────────────────────────────
    if not text.strip():
        text = "Can you tell me a bit more about what you're working on?"

    return text.strip()
