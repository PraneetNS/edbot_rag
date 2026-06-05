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

    # If best match is too distant, use generic fallback (cosine distance threshold of 0.6)
    if distances[0] > 0.6:
        return "That's a bit outside what I've seen most. Can you give me more context on where you're at?"

    # Build context string from retrieved mentor answers (top-2 is enough)
    context = "\n---\n".join(retrieved[:2])

    # Wrap in Edmentor system prompt
    prompt = f"""You are Edmentor — a senior engineering mentor. A student asked:
"{query}"

Here are relevant mentor insights on this topic:
{context}

Respond as Edmentor. Speak naturally, no markdown, 60-160 words. Be direct."""

    # Generate response
    if llm_model is not None and tokenizer is not None:
        # Local model mode (USE_LOCAL_MODEL=True)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        def local_generate():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = llm_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True
                )
            gen_ids = output[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        response = await loop.run_in_executor(None, local_generate)
        return response
    else:
        # Interim Mode: use Groq API to generate response
        # Import groq client singleton
        from edmentor.groq_client import llm as groq_llm
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = await groq_llm.chat(messages)
        return response
