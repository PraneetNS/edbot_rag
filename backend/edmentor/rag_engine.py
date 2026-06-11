"""
edmentor/rag_engine.py
──────────────────────
RAG retrieval + prompt synthesis for EduMentor.

Retrieval:
  - similarity_top_k = 4
  - Minimum relevance score threshold = 0.42
  - Section-aware deduplication: nodes with the same entry id are merged
    before being passed to the LLM (max 3 merged contexts)

Prompt synthesis (Part 6):
  - System prompt: EduMentor mentor voice with section-aware tone hints
  - Retrieved path: knowledge context + student query
  - No-retrieval path: LLM from weights + off-domain redirect
"""

import asyncio
import logging
import re

import chromadb
from sentence_transformers import SentenceTransformer

from edmentor.safety_filter import edumentor_filter

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

SIMILARITY_TOP_K   = 1          # RAG-only mode: fetch single best chunk
SCORE_THRESHOLD    = 0.42
MAX_MERGED_CONTEXTS = 1

# ── Caches ────────────────────────────────────────────────────────────────────

_client     = None
_collection = None
_embedder   = None


def get_chroma_resources():
    """Initialize (once) and return the ChromaDB collection + SentenceTransformer."""
    global _client, _collection, _embedder
    if _client is None:
        from rag.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
        _client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _collection, _embedder


# ── Section-aware tone hints (Part 6) ────────────────────────────────────────

_SECTION_TONE_HINTS: dict[str, str] = {
    "mindset_support": (
        "The student is emotionally struggling. Lead with validation (1 sentence), "
        "then reframe, then one action. Be human, not motivational-poster."
    ),
    "placement_timelines": (
        "Be specific about timing. Name actual topics or platforms. "
        "Don't say 'study hard' — say what to study and when."
    ),
    "company_patterns": (
        "Be specific to the company they asked about. Name the actual round structure, "
        "what they test, what gets people rejected."
    ),
    "career_roadmaps": (
        "Don't give the full roadmap in voice. Give the single most important starting "
        "point and one milestone. Offer to go deeper on any phase if they want."
    ),
    "higher_studies": (
        "Be honest about the financial reality and ROI. Don't romanticise MS abroad or "
        "dismiss it. Give your honest mentor take."
    ),
}

# ── Base system prompt ────────────────────────────────────────────────────────

_BASE_SYSTEM = """\
You are EduMentor — a senior software engineer and mentor for Indian engineering \
students (2nd to 4th year). You speak exactly like a real person: casual, direct, \
occasionally blunt, always warm. You have lived through placements, internships, \
DSA grind, burnout — all of it.

STRICT VOICE RULES:
- Reply in 2-3 sentences maximum. Never more.
- No bullet points. No numbered lists. No markdown. No headers.
- Write exactly how you would speak out loud.
- Use casual English — contractions, "yeah", "look", "honestly" are fine.
- Never start your reply with "I", "Sure", "Great question", "Absolutely".
- If the student sounds anxious or burnt out, acknowledge it in one sentence before answering.
- Give ONE concrete next step, not a full plan.
- Do NOT copy the knowledge context verbatim. Use it to inform your answer, \
then say it in your own mentor voice.\
"""


def _build_system_prompt(top_section: str | None) -> str:
    """Append section-specific tone hint if available."""
    system = _BASE_SYSTEM
    if top_section and top_section in _SECTION_TONE_HINTS:
        system += "\n\n" + _SECTION_TONE_HINTS[top_section]
    return system


# ── Retrieval ─────────────────────────────────────────────────────────────────

async def retrieve_chunks(query: str, k: int = SIMILARITY_TOP_K) -> tuple[list[dict], str | None]:
    """
    Retrieve top-k nodes from ChromaDB, filter by threshold 0.42,
    merge nodes sharing the same entry id, return:
      - list of merged context strings (max MAX_MERGED_CONTEXTS)
      - section of the top retrieved node (for tone hints) or None
    """
    try:
        collection, embedder = get_chroma_resources()
    except Exception as e:
        logger.error(f"Error loading Chroma or embedder: {e}")
        return [], None

    # Embed query
    loop = asyncio.get_running_loop()
    embedding = await loop.run_in_executor(
        None,
        lambda: embedder.encode(query, normalize_embeddings=True),
    )
    q_embedding = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    if not results or not results["documents"] or not results["documents"][0]:
        return [], None

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    # Filter by threshold (cosine distance → similarity = 1 - dist)
    filtered = []
    for doc, meta, dist in zip(docs, metas, distances):
        similarity = 1.0 - dist
        if similarity >= SCORE_THRESHOLD:
            filtered.append({"text": doc, "meta": meta, "score": similarity})

    if not filtered:
        return [], None

    # Identify top section
    top_section: str | None = filtered[0]["meta"].get("section") if filtered else None

    # Section-aware deduplication: merge nodes with the same entry id
    grouped: dict[str, str] = {}
    for node in filtered:
        eid = node["meta"].get("id", node["text"][:40])  # fallback key
        if eid in grouped:
            grouped[eid] += " " + node["text"]
        else:
            grouped[eid] = node["text"]

    context_chunks = list(grouped.values())[:MAX_MERGED_CONTEXTS]
    return context_chunks, top_section


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_retrieved_prompt(context_chunks: list[str], query: str, profile_line: str = "") -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the retrieved path."""
    merged_context = "\n\n".join(context_chunks)
    system = _build_system_prompt(None)  # section hint added separately by caller
    user = (
        f"Knowledge context (use this to inform your answer, do not quote it directly):\n"
        f"{merged_context}\n\n"
        f"Student says: \"{query}\"\n"
        f"{profile_line}\n\n"
        f"Reply as EduMentor in 2-3 spoken sentences."
    ).strip()
    return system, user


def build_no_retrieval_prompt(query: str, profile_line: str = "") -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the no-retrieval path."""
    system = _build_system_prompt(None)
    user = (
        f"Student says: \"{query}\"\n"
        f"{profile_line}\n\n"
        "You don't have specific reference material for this. Answer from your experience "
        "as a senior engineer mentoring Indian CS students. Keep it to 2-3 spoken sentences. "
        "If it's completely off-domain (not about DSA, placements, internships, resume, career, "
        "projects, or engineering student life), say exactly:\n"
        "\"That's outside what I focus on. Ask me about DSA, placements, internships, resume, "
        "or your career and I'll give you a real answer.\"\n"
        "Do not say anything else for off-domain queries."
    ).strip()
    return system, user


# ── Mentor sentence extractor ─────────────────────────────────────────────────

# Signals that a sentence sounds like real mentor advice
_ACTION_VERBS = {
    "start", "use", "focus", "build", "do", "go", "pick", "try", "take",
    "skip", "avoid", "learn", "practice", "make", "get", "push", "read",
    "write", "deploy", "contribute", "aim", "treat", "keep", "show", "solve",
    "crack", "prepare", "work", "target", "master", "nail"
}
_NOISE_PATTERNS = re.compile(
    r"^(topic|phase|section|part|note|source|entry|category|mentor explanation)\s*[:–—]",
    re.IGNORECASE
)

def _sentence_score(sentence: str) -> float:
    """Score a sentence on how 'mentor-like' it is (higher = better)."""
    s = sentence.strip()
    if not s or len(s) < 20:
        return 0.0
    if _NOISE_PATTERNS.match(s):
        return 0.0

    score = 0.0
    words = s.lower().split()

    # Second-person address is very mentor-like
    if any(w in words for w in ("you", "your", "you're", "you'll")):
        score += 2.5

    # Starts with an action verb — direct instruction
    first = words[0].rstrip(".,;:")
    if first in _ACTION_VERBS:
        score += 2.0

    # Contains action verb anywhere
    if any(w in _ACTION_VERBS for w in words):
        score += 1.0

    # Mentions concrete CS topics
    if any(t in s.lower() for t in ("dsa", "leetcode", "github", "project", "resume",
                                      "internship", "placement", "interview", "api",
                                      "system design", "open source", "contribute")):
        score += 1.5

    # Shorter sentences feel more mentor-like (under 25 words ideal)
    word_count = len(words)
    if word_count <= 20:
        score += 1.0
    elif word_count <= 30:
        score += 0.5
    elif word_count > 50:
        score -= 1.0

    # Penalise pure metadata/label lines
    if ":" in s[:20]:
        score -= 1.5

    return score


def _extract_mentor_lines(raw_chunk: str, max_words: int = 75) -> str:
    """
    Split chunk into sentences, score each for mentor quality,
    return the top 2 most advice-like sentences joined naturally.
    Caps total output at max_words.
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', raw_chunk.strip())

    scored = sorted(
        [(s.strip(), _sentence_score(s)) for s in sentences if s.strip()],
        key=lambda x: x[1],
        reverse=True,
    )

    # Take up to top 2 scoring sentences (in original order for flow)
    top_texts = set(s for s, sc in scored[:2] if sc > 0)
    ordered = [s.strip() for s in sentences if s.strip() in top_texts]

    if not ordered:
        # Fallback: just trim the raw chunk to max_words
        words = raw_chunk.split()
        trimmed = " ".join(words[:max_words])
        if len(words) > max_words and "." in trimmed:
            trimmed = trimmed[:trimmed.rfind(".")+1]
        return trimmed.strip()

    result = " ".join(ordered)
    # Final word cap
    words = result.split()
    if len(words) > max_words:
        result = " ".join(words[:max_words])
        if "." in result:
            result = result[:result.rfind(".")+1]

    return result.strip()


# ── Semantic Mentor Tone Templates ──────────────────────────────────────────

_SECTION_INTROS = {
    "mindset_support": [
        "Hey, I hear you. Stress is completely normal. ",
        "Look, don't beat yourself up over this. ",
        "Honestly, we've all felt this pressure. ",
    ],
    "placement_timelines": [
        "Alright, let's talk about the placement timeline. ",
        "Here is what you need to align with. ",
        "Timing is everything here. ",
    ],
    "company_patterns": [
        "Here is the actual pattern they look for. ",
        "Let's break down how they recruit. ",
        "Listen, this is their selection structure: ",
    ],
    "career_roadmaps": [
        "Here is the starting point I recommend. ",
        "Let's map out your roadmap: ",
        "Alright, focus on this path: ",
    ],
    "higher_studies": [
        "Let's look at the ROI and reality here. ",
        "Honestly, higher studies is a major commitment. ",
        "Here is my honest take on this: ",
    ],
}

_SECTION_OUTROS = {
    "mindset_support": [
        " Take a deep breath. You've got this.",
        " Focus on just one small step today.",
        " Don't let the pressure get to you.",
    ],
    "placement_timelines": [
        " Start tracking these windows now.",
        " Keep these milestones in mind.",
        " Take action early so you're not rushing later.",
    ],
    "company_patterns": [
        " Target their core pattern first.",
        " Solve their past papers and you'll be fine.",
        " Focus on these rounds to stand out.",
    ],
    "career_roadmaps": [
        " Take it one phase at a time.",
        " Master this first, then look ahead.",
        " Keep building and keep iterating.",
    ],
    "higher_studies": [
        " Weigh the costs and ROI before deciding.",
        " Make sure it aligns with your long-term goals.",
        " Don't just jump in without a clear plan.",
    ],
}

_GENERAL_INTROS = [
    "Look, here is the deal: ",
    "Alright, let's break this down. ",
    "Here is the approach I always recommend. ",
    "Listen, if I were in your shoes, here is what I'd do. ",
    "Okay, let's get into it: ",
]

_GENERAL_OUTROS = [
    " Try that first and see how it goes.",
    " Don't overthink it, just take the first step.",
    " Let me know if that makes sense.",
    " Put this into action today.",
]


# ── Main entry point ──────────────────────────────────────────────────────────

async def rag_retrieve_and_respond(
    query: str,
    llm_model=None,
    tokenizer=None,
    k: int = SIMILARITY_TOP_K,
    session=None,
) -> str:
    """
    Full RAG pipeline:
      1. Retrieve and filter chunks
      2. Merge by entry id
      3. Build section-aware system prompt + user prompt
      4. Call LLM
      5. Run safety filter
    """
    context_chunks, top_section = await retrieve_chunks(query, k=k)

    # Profile line from session if available
    profile_line = ""
    if session is not None and hasattr(session, "profile_string"):
        ps = session.profile_string()
        if ps:
            profile_line = f"Student context: {ps}"

    # ── RAG-only mode: extract best mentor sentences from top chunk ──────────
    if context_chunks:
        raw = context_chunks[0]
        mentor_text = _extract_mentor_lines(raw)
        
        import random
        intros = _SECTION_INTROS.get(top_section, _GENERAL_INTROS) if top_section else _GENERAL_INTROS
        outros = _SECTION_OUTROS.get(top_section, _GENERAL_OUTROS) if top_section else _GENERAL_OUTROS
        
        intro = random.choice(intros)
        outro = random.choice(outros)
        
        if any(greet in mentor_text.lower()[:15] for greet in ["hey", "hello", "hi", "look", "listen", "alright"]):
            return f"{mentor_text}{outro}"
        return f"{intro}{mentor_text}{outro}"

    # No relevant context found
    return (
        "Honestly, I don't have solid notes on that one. "
        "Ask me about DSA, placements, internships, resume, or your career — that's my lane."
    )

    # ── LLM synthesis (disabled) ──────────────────────────────────────────────
    # Uncomment the block below to re-enable LLM-based response generation.
    #
    # if context_chunks:
    #     system, user = build_retrieved_prompt(context_chunks, query, profile_line)
    #     system = _build_system_prompt(top_section)
    # else:
    #     system, user = build_no_retrieval_prompt(query, profile_line)
    #
    # from edmentor.qwen_client import qwen_client
    # if qwen_client.is_available():
    #     try:
    #         response_raw = await qwen_client.generate(user, system_prompt=system)
    #         return edumentor_filter(response_raw)
    #     except Exception as e:
    #         logger.error(f"Error during LLM generation: {e}")
    #
    # return edumentor_filter(
    #     "Looks like my LLM brain is offline right now — make sure Ollama is running and try again."
    # )


# ── Legacy helpers (kept for backward compatibility) ──────────────────────────

def is_ambiguous_or_context_free(query: str) -> bool:
    """Check if query is genuinely ambiguous, one-word, or context-free."""
    q_clean = query.strip().lower().rstrip("?.!")
    words = q_clean.split()
    if len(words) <= 1:
        return True
    from edmentor.intent_router import ON_DOMAIN_KEYWORDS
    if len(words) == 2:
        if not any(kw in q_clean for kw in ON_DOMAIN_KEYWORDS):
            return True
    return False


def build_qwen_prompt(chunks: list[str], query: str) -> str:
    """Legacy single-string prompt builder (used by some older call sites)."""
    context_str = "\n\n".join(chunks)
    system = _BASE_SYSTEM
    return (
        f"{system}\n\n"
        f"Knowledge context (use this to inform your answer, do not quote it directly):\n"
        f"{context_str}\n\n"
        f"Student says: \"{query}\"\n\n"
        f"Reply as EduMentor in 2-3 spoken sentences."
    )
