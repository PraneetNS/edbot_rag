import os
from pathlib import Path

# Base Paths
AI_CORE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = AI_CORE_DIR.parent
DATASET_PATH = WORKSPACE_DIR / "edumentor_ultra_premium_final.jsonl"

# Persistence Directories
CHROMA_PERSIST_DIR = WORKSPACE_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "edumentor_knowledge_v3"
BM25_INDEX_PATH = CHROMA_PERSIST_DIR / "bm25.pkl"
PARENT_DOCS_PATH = CHROMA_PERSIST_DIR / "parent_docs.pkl"
LOGS_DIR = AI_CORE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
RAG_LOGS_PATH = LOGS_DIR / "rag_logs.json"

# Ingestion Chunk Config
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# Model Configurations
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

# Retrieval Settings
VECTOR_TOP_K = 30
BM25_TOP_K = 30
FINAL_TOP_K = 5

# Switchable RAG Modes
RAG_MODE = "balanced"  # strict, balanced, loose

MODES = {
    "strict": {"high": 0.75, "low": 0.55},
    "balanced": {"high": 0.65, "low": 0.45},
    "loose": {"high": 0.50, "low": 0.30}
}

# LLM (Ollama Client Config)
OLLAMA_MODEL = "mistral"  # or qwen if locally renamed
OLLAMA_BASE_URL = "http://localhost:11434"
