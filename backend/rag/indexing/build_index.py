import os
import sys
import logging
import json
from pathlib import Path

# Ensure backend directory is in the sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from rag.config import (
    WORKSPACE_DIR,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME
)
from rag.database.chroma_manager import ChromaManager

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_new_index():
    logger.info("==================================================")
    logger.info("      BUILDING NEW PURE ENGINEERING RAG INDEX")
    logger.info("==================================================")

    # 1. Load data
    json_path = WORKSPACE_DIR / "edumentor_rag_chunks.json"
    logger.info(f"Loading mentoring JSON dataset from: {json_path}")
    if not json_path.exists():
        logger.error(f"Dataset path does not exist: {json_path}")
        sys.exit(1)
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    chunks = data.get("chunks", [])
    if not chunks:
        logger.error("No valid chunks were loaded. Exiting.")
        sys.exit(1)

    # Convert chunks directly to LlamaIndex TextNode objects
    nodes = []
    for chunk in chunks:
        # Preserve full metadata dict and inject source = "Edumentor Dataset"
        metadata = chunk.get("metadata", {}).copy()
        metadata["source"] = "Edumentor Dataset"
        
        node = TextNode(
            text=chunk.get("text", ""),
            metadata=metadata
        )
        nodes.append(node)

    logger.info(f"Loaded {len(nodes)} chunks as individual nodes (no further splitting applied).")

    # 2. Configure embedding model
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME
    )

    # 3. Initialize ChromaDB (rebuild fresh)
    logger.info(f"Rebuilding database at: {CHROMA_PERSIST_DIR}")
    chroma_manager = ChromaManager(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME
    )
    chroma_manager.initialize_db(rebuild_fresh=True)
    vector_store = chroma_manager.get_vector_store()

    # 4. Store in Vector Store Index
    logger.info("Populating vector store and computing semantic embeddings...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    logger.info("==================================================")
    logger.info("       INGESTION AND INDEXING COMPLETE!")
    logger.info("==================================================")

if __name__ == "__main__":
    build_new_index()

