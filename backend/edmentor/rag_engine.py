import os
import sys
import logging
import asyncio
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Cache collections and embedder models to avoid reloading overhead
_client = None
_collection = None
_embedder = None

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

async def rag_retrieve_and_respond(query: str, llm_model=None, tokenizer=None, k: int = 3) -> str:
    """
    RAG retrieval and response generation.
    Used when local model confidence is low, or as fallback in interim mode.
    """
    try:
        collection, embedder = get_chroma_resources()
    except Exception as e:
        logger.error(f"Error loading Chroma or embedder model: {e}")
        return "I am having a connection issue accessing my database. Give me a moment and try again."

    # CRITICAL CONSTRAINTS (Requirements Three):
    # The query embedding prefix at retrieval time is exactly 'Student: {query}'
    # which is identical to the prefix used during Chroma indexing.
    query_prefix = f"Student: {query}"

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

    # Retrieve top-k
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    if not results or not results["documents"] or len(results["documents"][0]) == 0:
        return "That's a bit outside what I've seen most. Can you give me more context on where you're at?"

    retrieved = results["documents"][0]  # list of k answers
    distances = results["distances"][0]

    # Only keep retrieved chunks that clear the cosine distance threshold of 0.15 (0.85 similarity)
    valid_retrieved = [doc for doc, dist in zip(retrieved, distances) if dist <= 0.15]
    if not valid_retrieved:
        return "That's a bit outside what I've seen most. Can you give me more context on where you're at?"

    # Extract the mentor response from the highest rank document directly
    top_doc = valid_retrieved[0]
    if "Mentor:" in top_doc:
        response = top_doc.split("Mentor:", 1)[1].strip()
    else:
        response = top_doc.strip()

    return response
