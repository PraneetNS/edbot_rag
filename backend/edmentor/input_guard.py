import re
import logging

logger = logging.getLogger(__name__)

# ── STT Noise / Fillers ───────────────────────────────────────────────────────
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

FALLBACK_RESPONSE = "Did not catch that. Go ahead."
_NOISE_PATTERN = re.compile(r'^[\W\s]{0,3}$', re.UNICODE)

def is_stt_noise(text: str) -> bool:
    """Returns True if the input is empty, filler sound, or meaningless punctuation."""
    stripped = text.strip().lower().rstrip("?.!,;:")
    if not stripped or _NOISE_PATTERN.match(stripped):
        return True
    tokens = stripped.split()
    if all(t in _NOISE_TOKENS for t in tokens):
        return True
    if len(stripped) <= 2 and not stripped.isalpha():
        return True
    return False

def clean_input(text: str) -> str | None:
    """
    Cleans user input.
    Returns None if the input is invalid or determined to be STT noise.
    """
    if not text:
        return None
    if is_stt_noise(text):
        return None
    return text.strip()


# ── Pre-LLM Jailbreak Guard ───────────────────────────────────────────────────
JAILBREAK_PATTERNS = [
    r"ignore.{0,10}instructions",           # covers: ignore your/all/previous/all your instructions
    r"ignore.{0,10}(prompt|system|rules)",
    r"(pretend|act|behave).{0,30}(you are|you're|ur).{0,30}(ai|gpt|claude|gemini|assistant)",
    r"you are now",
    r"developer mode",
    r"dan mode",
    r"jailbreak",
    r"no restrictions",
    r"forget (your|the|all).{0,20}(prompt|instructions|rules|system)",
    r"ignore (safety|guidelines|restrictions)",
    r"(bypass|override).{0,20}(filter|restriction|safety|rule)",
    r"tell me your (prompt|system|instructions)",
    r"reveal (your|the).{0,10}(prompt|instructions|system)",
]

JAILBREAK_RESPONSE = "I am Edmentor. I am here for engineering mentorship only."

def check_jailbreak(text: str) -> bool:
    """Scans query for common jailbreak patterns."""
    t = text.lower().strip()
    return any(re.search(p, t) for p in JAILBREAK_PATTERNS)


# ── Pre-LLM Vague Input Guard ─────────────────────────────────────────────────
VAGUE_PATTERNS = [
    r"^.{1,6}$",                          # very short (under 6 chars)
    r"^[^a-zA-Z]*$",                       # no letters at all
    r"^(ok|okay|k|yes|no|yep|nope|fine|cool|alright|sure|hmm|hm)$",
    r"^(what|how|why|when|where|who|which)\??$",  # bare question words
]

VAGUE_RESPONSE = "Can you tell me more about what you are working on or stuck on? I will give you a direct answer."

BARE_TWO_WORD = [
    "career paths", "career path", "career options",
    "dsa roadmap", "placement tips", "interview tips",
    "project ideas", "resume tips", "tech stack",
    "what next", "whats next", "next steps",
]

def is_vague(text: str) -> bool:
    """Checks if input is vague or missing context."""
    t = text.lower().strip()
    # Bare single-word domain terms with no context
    bare_domain = [
        "dsa", "placement", "internship", "resume",
        "career", "project", "interview", "coding",
        "algorithm", "programming"
    ]
    if t in bare_domain:
        return True
    if t in BARE_TWO_WORD:
        return True
    return any(re.search(p, t) for p in VAGUE_PATTERNS)
