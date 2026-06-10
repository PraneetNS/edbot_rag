import os
import sys
import logging
import asyncio
import re
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from edmentor.safety_filter import edumentor_filter

logger = logging.getLogger(__name__)

# Cache collections and embedder models to avoid reloading overhead
_client = None
_collection = None
_embedder = None

QUERY_EXPANSIONS = {
    'arrays': 'arrays data structure practice problems two pointer sliding window progression array next step',
    'placement': 'placement preparation interview coding resume 60 days plan timeline prep roadmap',
    'recursion': 'recursion base case stack call explanation beginner',
    'resume': 'resume projects skills engineering student format',
    'backlog': 'backlog placement eligibility impact career',
    'dp': 'dynamic programming memoization tabulation practice',
    'linked list': 'linked list operations reversal cycle detection',
    'trees': 'binary tree traversal bst problems',
    # Explicit time-based placement prep
    '60 days': 'placement preparation 60 days plan timeline prep roadmap study schedule',
    '30 days': 'placement preparation 30 days plan timeline prep roadmap study schedule',
    'placement prep': 'placement preparation roadmap timeline prep plan schedule',
    # Explicit DSA progression
    'do next': 'dsa progression data structures arrays learning path next topic',
    'next step': 'dsa progression data structures arrays learning path next topic',
    'what next': 'dsa progression data structures arrays learning path next topic',
}

def expand_query(query: str) -> str:
    # Check if query has under 15 words
    if len(query.split()) < 15:
        q_lower = query.lower()
        for keyword, expansion in QUERY_EXPANSIONS.items():
            if keyword in q_lower:
                return f"{query} {expansion}"
    return query

def get_chroma_resources():
    """Initializes and returns Chroma DB collection and embedding model."""
    global _client, _collection, _embedder
    if _client is None:
        from rag.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
        _client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    if _embedder is None:
        # all-MiniLM-L6-v2 — free, tiny (80MB), fast, good semantic quality
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _collection, _embedder

async def retrieve_chunks(query: str, k: int = 3) -> list[str]:
    """
    Retrieves top-k chunk texts (mentor answers) from ChromaDB collection.
    Filters out chunks with similarity below 0.45 (cosine distance > 0.55).
    If zero chunks pass the threshold, returns empty list.
    """
    try:
        collection, embedder = get_chroma_resources()
    except Exception as e:
        logger.error(f"Error loading Chroma or embedder model: {e}")
        return []

    # Embed the query directly (no expansion, no Student: prefix)
    loop = asyncio.get_running_loop()
    embedding = await loop.run_in_executor(
        None, 
        lambda: embedder.encode(query, normalize_embeddings=True)
    )
    if hasattr(embedding, "tolist"):
        q_embedding = embedding.tolist()
    else:
        q_embedding = list(embedding)

    # Retrieve top-k
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=k,
        where={"source": "Edumentor Dataset"},
        include=["documents", "metadatas", "distances"]
    )

    if not results or not results["documents"] or len(results["documents"][0]) == 0:
        return []

    retrieved = results["documents"][0]
    distances = results["distances"][0]

    # Filter by minimum relevance score threshold 0.45 (similarity = 1.0 - dist >= 0.45)
    valid_chunks = []
    for doc, dist in zip(retrieved, distances):
        similarity = 1.0 - dist
        if similarity >= 0.45:
            valid_chunks.append(doc)
    return valid_chunks

def build_qwen_prompt(chunks: list[str], query: str) -> str:
    """
    Build prompt for Qwen using retrieved context.
    """
    context_str = "\n\n".join(chunks)
    return (
        f"You are Edmentor, a senior engineering mentor.\n"
        f"Use the retrieved reference information if helpful, but answer in your mentor voice.\n"
        f"Do not use markdown, bullet points, or list formatting.\n\n"
        f"Reference Context:\n{context_str}\n\n"
        f"Student Question: {query}\n"
        f"Mentor Response:"
    )

def is_ambiguous_or_context_free(query: str) -> bool:
    """Check if query is genuinely ambiguous, one-word, or context-free."""
    q_clean = query.strip().lower().rstrip("?.!")
    words = q_clean.split()
    if len(words) <= 1:
        return True
    
    # If 2 words, check if it contains any on-domain keyword
    from edmentor.intent_router import ON_DOMAIN_KEYWORDS
    if len(words) == 2:
        if not any(kw in q_clean for kw in ON_DOMAIN_KEYWORDS):
            return True
            
    return False

async def rag_retrieve_and_respond(query: str, llm_model=None, tokenizer=None, k: int = 3) -> str:
    """
    RAG retrieval and response generation.
    Always synthesizes the response using the LLM.
    """
    chunks = await retrieve_chunks(query, k=k)
    
    # Build prompt exactly as instructed
    if chunks:
        # Take up to top 2 retrieved chunks as specified in prompt template:
        # {chunk_1_text}
        # {chunk_2_text}
        chunk_texts = [f"{c}" for c in chunks[:2]]
        knowledge_str = "\n".join(chunk_texts)
        prompt = (
            "You are EduMentor, a direct senior engineer mentoring an Indian "
            "engineering student. Speak casually, 2-3 sentences max, no markdown.\n\n"
            "Relevant knowledge:\n"
            f"{knowledge_str}\n\n"
            f"Student asks: {query}\n\n"
            "Answer by synthesising the knowledge above into a natural spoken reply. "
            "Do NOT copy the knowledge verbatim. Do NOT use bullet points or lists."
        )
    else:
        prompt = (
            "You are EduMentor, a direct senior engineer mentoring an Indian "
            "engineering student. Speak casually, 2-3 sentences max, no markdown.\n\n"
            f"Student asks: {query}\n\n"
            "Answer from your general knowledge as a mentor. Stay on-domain "
            "(DSA, placements, internships, resume, career, projects). "
            "If completely off-domain, say: \"That's outside what I focus on. "
            "Ask me about DSA, placements, or your career instead.\""
        )
        
    from edmentor.qwen_client import qwen_client
    if qwen_client.is_available():
        try:
            response_raw = await qwen_client.generate(prompt)
            # Run the new post-processing safety filter
            return edumentor_filter(response_raw)
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            
    # If LLM is offline, return a friendly synthesized message in the mentor's voice
    offline_response = (
        "I am EduMentor. My LLM brain seems to be offline right now. "
        "Double check that Ollama is running on your system so I can synthesize a reply."
    )
    return edumentor_filter(offline_response)

