"""
edmentor/input_guard.py
───────────────────────
STT (Speech-to-Text) noise filter — GATE ⓪, runs before DomainGuard.

Catches low-signal transcription artifacts:
  - Pure filler sounds: "umm", "uh", "hmm", "er", "ah"
  - Single non-word characters / punctuation only
  - Very short (≤2 char) inputs that cannot be a real query

Returns a friendly re-prompt: "Didn't catch that, go ahead."
"""

import re

# ── Exact filler / noise tokens ───────────────────────────────────────────────
_NOISE_TOKENS = frozenset([
    "umm", "um", "uh", "uhh", "uhhh",
    "hmm", "hm", "hmm", "hmmm",
    "er", "err", "errr",
    "ah", "ahh", "aah",
    "oh", "ohh",
    "mmm", "mm",
    "eh", "ehh",
    "huh",
])

STT_NOISE_RESPONSE = "Didn't catch that, go ahead."

# ── Regex: all-punctuation / all-whitespace / single character ────────────────
_NOISE_PATTERN = re.compile(r'^[\W\s]{0,3}$', re.UNICODE)


def is_stt_noise(text: str) -> bool:
    """
    Returns True if the input is almost certainly STT transcription noise
    (filler sound, empty string, or meaningless single character).
    """
    stripped = text.strip().lower().rstrip("?.!,;:")

    # Empty or purely punctuation/whitespace
    if not stripped or _NOISE_PATTERN.match(stripped):
        return True

    # Pure filler token(s)
    tokens = stripped.split()
    if all(t in _NOISE_TOKENS for t in tokens):
        return True

    # Suspiciously short (1-2 chars) with no recognized keyword value
    if len(stripped) <= 2 and not stripped.isalpha():
        return True

    return False


def check_stt_noise(text: str) -> tuple[bool, str]:
    """
    Returns (is_noise: bool, response: str).
    If is_noise is True, return `response` immediately without any RAG/LLM call.
    """
    if is_stt_noise(text):
        return True, STT_NOISE_RESPONSE
    return False, ""
