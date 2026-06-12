"""
edmentor/guard.py
─────────────────
Domain boundary check — GATE ①, runs before ANY RAG call or LLM inference.
Pure string matching + regex. Executes in <1ms.

Edmentor ONLY handles:
    academics, DSA, algorithms, programming, projects, GitHub,
    internships, placements, resume, career planning,
    competitive programming, research, higher studies.

Character-lock queries ("who made you?", "what model are you?") are
intercepted here and returned with the exact locked response before
touching any retrieval or LLM.
"""

import re

# ── Exact character-lock response (spec: word-for-word) ─────────────────────
IDENTITY_RESPONSE = "I am Edmentor, your engineering mentor."
CHARACTER_LOCK_RESPONSE = "I am here to mentor you, not to be anything else."
OUT_OF_SCOPE_RESPONSE = "That is outside what I am here for. Ask me about your engineering journey."

# ── Identity / character-manipulation triggers ────────────────────────────────
_IDENTITY_PATTERNS = [
    r"\bwho\s+are\s+you\b",
    r"\bwhat\s+(is\s+)?your\s+name\b",
    r"\byour\s+name\b",
    r"\bintroduce\s+yourself\b",
]

_CHARACTER_LOCK_PATTERNS = [
    r"\bwhat\s+model\b",
    r"\bbuilt\s+on\b",
    r"\bpowered\s+by\b",
    r"\bgpt\b",
    r"\bclaude\b",
    r"\bgemini\b",
    r"\bllama\b",
    r"\bmistral\b",
    r"\btraining\s+data\b",
    r"\barchitecture\b",
    r"\bpretend\s+to\s+be\b",
    r"\bact\s+as\b",
    r"\brole\s*play\b",
    r"\bignore\s+(previous\s+)?instructions\b",
    r"\bforget\s+you\s+are\b",
    r"\bbreak\s+character\b",
    r"\byou\s+are\s+now\b",
    r"\bnew\s+persona\b",
]

# ── Hard out-of-scope blocklist ───────────────────────────────────────────────
_BLOCKLIST_PATTERNS = [
    r"\bmovies?\b",
    r"\bfilms?\b",
    r"\bseries\b",
    r"\bnetflix\b",
    r"\bwhat\s+to\s+watch\b",
    r"\brelationship[s]?\b",
    r"\bboyfriend\b",
    r"\bgirlfriend\b",
    r"\bnews\b",
    r"\bpolitics?\b",
    r"\belection[s]?\b",
    r"\bsport[s]?\b",
    r"\bcricket\b",
    r"\bfootball\b",
    r"\brecipe[s]?\b",
    r"\bcook(ing)?\b",
    r"\bweather\b",
    r"\bhoroscope\b",
    r"\bastrology\b",
    r"\bjoke[s]?\b",
    r"\bfunny\b",
    r"\bmeme[s]?\b",
    r"\bgossip\b",
    r"\bcelebrit(y|ies)\b",
    r"\bstock\s+market\b",
    r"\bcryptocurrenc(y|ies)\b",
    r"\bbetting\b",
    r"\bgambling\b",
    r"\bhack(ing|er)?\b",
    r"\bexploit[s]?\b",
    r"\bspam\b",
]

# ── In-scope keyword fast-path ────────────────────────────────────────────────
_IN_SCOPE_KEYWORDS = [
    "dsa", "algorithm", "data structure", "tree", "graph", "sort", "dynamic programming",
    "recursion", "binary", "heap", "stack", "queue", "linked list", "hash", "complexity",
    "placement", "internship", "job", "interview", "resume", "cv", "career", "recruit",
    "offer letter", "mock interview", "hr round", "technical round",
    "python", "java", "c++", "javascript", "typescript", "golang", "rust", "kotlin",
    "programming", "coding", "code", "software", "developer", "development",
    "project", "github", "git", "open source", "contribution", "pull request",
    "machine learning", "ml", "deep learning", "nlp", "ai", "neural network",
    "cloud", "aws", "gcp", "azure", "docker", "kubernetes", "jenkins", "ci/cd",
    "continuous integration", "continuous deployment", "terraform", "ansible",
    "nginx", "rest api", "graphql", "microservices", "agile", "scrum", "sdlc",
    "unit testing", "integration testing", "system design", "os", "operating system",
    "network", "computer science", "research", "paper", "phd", "masters", "gate",
    "higher studies", "college", "engineering", "semester", "cgpa", "gpa", "study",
    "learn", "edmentor", "mentor", "hello", "hi", "hey", "help", "what", "how",
    "why", "when", "can you", "explain", "tell me", "guide", "roadmap", "plan",
    "prepare", "practice", "competitive programming", "leetcode", "codeforces",
    "hackerrank", "cp",
]


class DomainGuard:
    """
    Stateless guard. Call `check(query)` — returns a tuple:
        (is_blocked: bool, response: str | None, reason: str)

    If is_blocked is True, return `response` immediately.
    If is_blocked is False, proceed to RAG.
    """

    def check(self, query: str):
        q = query.strip()
        q_lower = q.lower()

        # Normalize spaces and strip trailing punctuation for robust matching
        q_clean = re.sub(r"\s+", " ", q_lower).strip()
        q_norm = q_clean.rstrip("?.!,;:")

        # 1. Identity queries — answer with locked response
        for pattern in _IDENTITY_PATTERNS:
            if re.search(pattern, q_clean):
                return True, IDENTITY_RESPONSE, "identity"

        # 2. Character manipulation / model-probing — lock response
        for pattern in _CHARACTER_LOCK_PATTERNS:
            if re.search(pattern, q_clean):
                return True, CHARACTER_LOCK_RESPONSE, "character_lock"

        # 3. Hard blocklist — out of scope
        for pattern in _BLOCKLIST_PATTERNS:
            if re.search(pattern, q_clean):
                return True, OUT_OF_SCOPE_RESPONSE, "blocklist"

        # 4. Fast in-scope keyword pass
        if any(kw in q_clean or kw in q_norm for kw in _IN_SCOPE_KEYWORDS):
            return False, None, "in_scope"

        # 5. Default — treat as out of scope if no keywords matched
        # (very short / ambiguous queries land here)
        if len(q_clean.split()) < 3:
            # Very short queries — pass through, let the LLM handle vagueness
            return False, None, "short_query"

        return True, OUT_OF_SCOPE_RESPONSE, "no_keyword_match"


# Module-level singleton
guard = DomainGuard()

# ── Identity check (LangChain Rebuild) ────────────────────────────────────────
def check_identity(text: str) -> bool:
    q = text.lower().strip().rstrip("?.!")
    probing_keywords = [
        "chatgpt", "gemini", "claude", "ai assistant", "general chatbot", 
        "chatbot", "who are you", "your name", "introduce yourself", 
        "what are you", "are you an ai", "r u a rag", "r u a model", 
        "are you a rag", "are you a model", "system prompt", "using rag", 
        "reveal your instructions", "reveal instructions"
    ]
    return any(kw in q for kw in probing_keywords)

# ── Greeting check (LangChain Rebuild) ────────────────────────────────────────
GREETING_RESPONSE = (
    "Hey. Tell me what you are working on or stuck on right now. "
    "DSA, placements, resume, internships, projects — whatever it is, let's get into it."
)

_GREETINGS_SET = {
    "hello", "hi", "hey", "hola", "greetings", "hii", "hiii",
    "good morning", "good evening", "yo", "sup", "heyy", "heyyy",
}

def check_greeting(text: str) -> bool:
    q = text.lower().strip().rstrip("?.!")
    return q in _GREETINGS_SET or (len(q.split()) == 1 and q in _GREETINGS_SET)

