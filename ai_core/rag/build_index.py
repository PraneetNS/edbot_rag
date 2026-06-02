import os
import sys
import time
import pickle
import logging
from pathlib import Path

# Add workspace, backend and ai_core directories to system path
AI_CORE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = AI_CORE_DIR.parent
BACKEND_DIR = WORKSPACE_DIR / "backend"
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))
if str(AI_CORE_DIR) not in sys.path:
    sys.path.append(str(AI_CORE_DIR))



from ai_core.config import (
    DATASET_PATH,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PARENT_DOCS_PATH
)
from rag.loaders.jsonl_loader import load_jsonl_conversations
from rag.database.chroma_manager import ChromaManager
from ai_core.rag.bm25 import BM25Searcher

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def map_metadata(doc_text: str, dataset_topic: str, dataset_source: str) -> dict:
    """Dynamically parses and maps rich enterprise RAG metadata."""
    topic = str(dataset_topic).lower().strip()
    source = str(dataset_source).lower().strip()
    text = doc_text.lower()
    
    # 1. Base Meta
    meta = {
        "source": dataset_source,
        "title": f"{dataset_topic} Mentoring Knowledge Base",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "confidence": "high"
    }
    
    # 2. Domain classification
    if any(k in topic for k in ["placement", "job", "career", "interview", "resume", "recruit", "internship"]):
        meta["domain"] = "career"
        meta["audience"] = "final_year"
        meta["year"] = "4"
    elif any(k in topic for k in ["support", "help", "ticket", "login", "portal", "contact", "mail", "vtu", "certif", "verify", "stamp"]):
        meta["domain"] = "portal"
        meta["audience"] = "2nd_year"
        meta["year"] = "2"
    else:
        meta["domain"] = "course"
        meta["audience"] = "2nd_year"
        meta["year"] = "2"
        
    meta["topic"] = topic
    
    # 3. Content Type classification
    if "syllabus" in text or "curriculum" in text or "duration" in text or "schedule" in text:
        meta["content_type"] = "policy"
    elif "example" in text or "projects" in text or "sample" in text:
        meta["content_type"] = "example"
    elif "how to" in text or "guide" in text or "step" in text or "download" in text:
        meta["content_type"] = "tutorial"
    else:
        meta["content_type"] = "concept"
        
    # 4. Difficulty classification
    if any(k in text for k in ["advanced", "complex", "performance", "architecture", "system design", "rerank", "enterprise"]):
        meta["difficulty"] = "advanced"
        meta["year"] = "4"
    elif any(k in text for k in ["beginner", "intro", "basics", "simple", "what is"]):
        meta["difficulty"] = "beginner"
    else:
        meta["difficulty"] = "intermediate"
        
    # 5. Document Type
    meta["type"] = "faq" if "?" in doc_text else "qa"
    
    return meta

def build_index_v3():
    logger.info("==================================================")
    logger.info("      BUILDING EDUMENTOR RAG INDEX V3 (Modular)")
    logger.info("==================================================")

    # 1. Load Data
    if not DATASET_PATH.exists():
        logger.error(f"Dataset path does not exist: {DATASET_PATH}")
        sys.exit(1)
        
    documents = load_jsonl_conversations(str(DATASET_PATH))
    if not documents:
        logger.error("No valid conversations were loaded. Exiting.")
        sys.exit(1)
        
    # 2. Setup Chunk Splitter
    logger.info(f"Setting up SentenceSplitter (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    parent_docs_store = {}
    chunk_nodes = []
    
    # 3. Parse into chunks with rich metadata & parent mapping
    logger.info("Parsing documents and mapping parent lookups...")
    for idx, doc in enumerate(documents):
        doc_id = f"parent_doc_{idx}"
        # Save full document text as the parent context
        parent_docs_store[doc_id] = doc.text
        
        # Segment full conversation into chunks
        chunks = splitter.split_text(doc.text)
        
        # Apply rich metadata mapping
        meta = map_metadata(doc.text, doc.metadata.get("topic", "General"), doc.metadata.get("source", "Unknown"))
        
        for c_idx, chunk_text in enumerate(chunks):
            # Clone metadata to avoid ref leaks, then add parent hook
            chunk_meta = meta.copy()
            chunk_meta["parent_doc_id"] = doc_id
            chunk_meta["chunk_index"] = c_idx
            
            node = TextNode(
                text=chunk_text,
                metadata=chunk_meta
            )
            chunk_nodes.append(node)
            
    logger.info(f"Generated {len(chunk_nodes)} text chunks from {len(documents)} source conversations.")

    # 4. Initialize and Rebuild ChromaDB Collection
    logger.info(f"Initializing ChromaDB vector database collection '{CHROMA_COLLECTION_NAME}'...")
    chroma_manager = ChromaManager(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME
    )
    chroma_manager.initialize_db(rebuild_fresh=True)
    vector_store = chroma_manager.get_vector_store()
    
    # 5. Populate ChromaDB Vector Index (BAAI/bge-base-en-v1.5)
    logger.info(f"Populating Vector Store using embedding model: {EMBEDDING_MODEL_NAME}...")
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex(
        chunk_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    # 6. Save Parent Documents Dictionary
    logger.info(f"Saving persistent parent documents map to: {PARENT_DOCS_PATH}")
    PARENT_DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(PARENT_DOCS_PATH, "wb") as f:
            pickle.dump(parent_docs_store, f)
        logger.info("Parent document mappings persisted successfully.")
    except Exception as e:
        logger.error(f"Failed to persist parent docs store: {e}")

    # 7. Pre-Index and Save Persistent BM25 Search
    logger.info("Serializing persistent BM25 index corpus...")
    bm25_searcher = BM25Searcher()
    bm25_searcher.build_and_save(chunk_nodes)

    logger.info("==================================================")
    logger.info("     EDUMENTOR RAG V3 INGESTION COMPLETE!")
    logger.info("==================================================")

if __name__ == "__main__":
    build_index_v3()
