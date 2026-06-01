import sys
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BASE_DIR.parent

# Input Dataset Path
DATASET_PATH = WORKSPACE_DIR / "edumentor_ultra_premium_final.jsonl"

# Chroma Database Settings
CHROMA_PERSIST_DIR = WORKSPACE_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "edumentor_knowledge"

# Embedding Model (Sentence Transformers)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Semantic Splitter Config
BREAKPOINT_PERCENTILE_THRESHOLD = 90
CHUNK_SIZE_TARGET = 400

# Retrieval & Reranking Settings
SIMILARITY_TOP_K = 5
RERANK_TOP_N = 3
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM (Ollama configuration)
OLLAMA_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"
