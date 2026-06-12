import os
import logging
import asyncio
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Force offline mode for Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"

_embeddings = None
_db = None
_retriever = None

def get_retriever():
    global _embeddings, _db, _retriever
    if _retriever is None:
        from rag.config import CHROMA_PERSIST_DIR
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _db = Chroma(
            collection_name="edumentor_v3",
            embedding_function=_embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR)
        )
        _retriever = _db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.42}
        )
    return _retriever

async def retrieve(query: str) -> list[Document]:
    """
    Retrieve relevant documents from the edumentor_v3 ChromaDB collection.
    Appends "Student: " prefix to query to match the indexed text format.
    """
    try:
        retriever = get_retriever()
        
        # Ensure query prefix matches the Student/Mentor format used during indexing
        formatted_query = query
        if not query.lower().startswith("student:"):
            formatted_query = f"Student: {query}"
            
        logger.info(f"LangChain retrieve called with query: {formatted_query}")
        
        # Invoke LangChain retriever asynchronously
        docs = await retriever.ainvoke(formatted_query)
        logger.info(f"Retrieved {len(docs)} documents.")
        return docs
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}", exc_info=True)
        return []
