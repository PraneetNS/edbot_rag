"""
edmentor/voice_limit.py
───────────────────────
Hard 80-word post-processing truncation for Edmentor voice responses.
Applied BEFORE the API returns AND before text is piped to TTS.

Also strips markdown artifacts that would sound terrible when spoken:
    - Asterisks (**bold**, *italic*)
    - Hash headers (# ## ###)
    - Numbered lists (1. 2. 3.)
    - Dashes as bullet points (- item)
    - Backtick code blocks
    - Parenthetical sources (*Retrieved from...*)
"""

import re


def strip_markdown(text: str) -> str:
    """
    Remove markdown formatting that sounds terrible when spoken aloud.
    Preserves natural sentence structure.
    """
    # Remove fenced code blocks entirely
    text = re.sub(r"```[\s\S]*?```", "", text)

    # Remove inline code
    text = re.sub(r"`[^`]+`", lambda m: m.group(0).strip("`"), text)

    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)

    # Remove ATX headers (# ## ###)
    text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Convert numbered lists to natural flow
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Remove dash/asterisk bullet points
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)

    # Remove parenthetical meta-tags like *(EduMentor - Offline...)* or *(Retrieved from...)*
    text = re.sub(r"\*\(.*?\)\*", "", text)

    # Collapse multiple blank lines to single
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    return text.strip()


def enforce_voice_limit(text: str, max_words: int = 200) -> str:
    """
    Hard post-processing word-limit enforcement for voice output.
    Routes to safety_filter's edumentor_filter to maintain consistent cleaning.
    """
    from edmentor.safety_filter import edumentor_filter
    return edumentor_filter(text, max_words=max_words)



def word_count(text: str) -> int:
    """Return word count of a cleaned string."""
    return len(strip_markdown(text).split())


def speaking_duration_label(text: str, wps: float = 3.5) -> str:
    """
    Convert response text to an approximate speaking-duration label.
    Uses 3.5 words-per-second (natural mentor speaking pace).

    Returns a string like '~18s' for display in the UI.
    """
    count = word_count(text)
    seconds = round(count / wps)
    return f"~{max(seconds, 1)}s"
