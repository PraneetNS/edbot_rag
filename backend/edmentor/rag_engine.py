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
        backend_dir = Path(__file__).resolve().parent.parent
        chroma_path = backend_dir.parent / "edumentor_chroma"
        _client = chromadb.PersistentClient(path=str(chroma_path))
        _collection = _client.get_or_create_collection(
            name="edumentor_mentor",
            metadata={"hnsw:space": "cosine"}
        )
    if _embedder is None:
        # all-MiniLM-L6-v2 — free, tiny (80MB), fast, good semantic quality
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _collection, _embedder

async def retrieve_chunks(query: str, k: int = 3) -> list[str]:
    """
    Retrieves top-k chunk texts (mentor answers) from ChromaDB collection.
    Applies query expansion first.
    Filters out chunks with similarity below 0.15 (cosine distance > 0.85).
    """
    try:
        collection, embedder = get_chroma_resources()
    except Exception as e:
        logger.error(f"Error loading Chroma or embedder model: {e}")
        return []

    # Apply query expansion
    expanded = expand_query(query)
    query_prefix = f"Student: {expanded}"

    # Embed the query
    loop = asyncio.get_running_loop()
    embedding = await loop.run_in_executor(
        None, 
        lambda: embedder.encode(query_prefix, normalize_embeddings=True)
    )
    if hasattr(embedding, "tolist"):
        q_embedding = embedding.tolist()
    else:
        q_embedding = list(embedding)

    # Retrieve top-k (overfetch a bit so we can filter by threshold)
    # Filter only for Edumentor Dataset source
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=k * 2,
        where={"source": "Edumentor Dataset"},
        include=["documents", "metadatas", "distances"]
    )

    if not results or not results["documents"] or len(results["documents"][0]) == 0:
        return []

    retrieved = results["documents"][0]
    distances = results["distances"][0]

    # Filter by similarity threshold 0.15 (dist <= 0.85)
    valid_chunks = [doc for doc, dist in zip(retrieved, distances) if dist <= 0.85]
    return valid_chunks[:k]

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
    Bypasses external LLMs. Fallback to direct RAG answer if Qwen is offline.
    """
    chunks = await retrieve_chunks(query, k=k)
    
    # Check if Qwen is available
    from edmentor.qwen_client import qwen_client
    if qwen_client.is_available():
        if not chunks:
            # Fallback path when RAG fails to find relevant chunks but LLM is available
            if is_ambiguous_or_context_free(query):
                return "That's a bit outside what I've seen most. Can you give me more context on where you're at?"
                
            system_prompt = "You are Edmentor, a senior engineering mentor."
            prompt = (
                "Answer the student's question using your own knowledge as a senior engineering mentor. "
                "Do not use markdown, bullet points, or list formatting. Keep the response natural, spoken, and under 75 words.\n\n"
                f"Student Question: {query}"
            )
            response_raw = await qwen_client.generate(prompt, system_prompt=system_prompt)
        else:
            prompt = build_qwen_prompt(chunks, query)
            response_raw = await qwen_client.generate(prompt)
        return edumentor_filter(response_raw, max_words=75)

    # Fallback to direct top chunk answer when Qwen is offline
    if not chunks:
        if is_ambiguous_or_context_free(query):
            return "That's a bit outside what I've seen most. Can you give me more context on where you're at?"
            
        offline_fallback = (
            "I am EduMentor. I currently do not have enough verified context in my knowledge base to answer this question. "
            "However, as a senior mentor, I suggest looking into official documentation and academic roadmaps to guide your research."
        )
        return edumentor_filter(offline_fallback, max_words=75)

    top_chunk = chunks[0]
    
    # Strip any potential prefix if it has one
    if "Mentor:" in top_chunk:
        response = top_chunk.split("Mentor:", 1)[1].strip()
    else:
        response = top_chunk.strip()
        
    return edumentor_filter(response, max_words=75)
