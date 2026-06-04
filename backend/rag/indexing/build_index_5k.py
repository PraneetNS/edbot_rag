"""
build_index_5k.py
─────────────────
Ingestion pipeline for edumentor_synthetic_5k.jsonl → ChromaDB collection 'edumentor_5k'.

Pipeline steps
──────────────
  1. Load & deduplicate  – strip [NNNN] suffix, skip duplicate base questions
  2. Sentence-segment    – custom_sentence_splitter preserves code blocks, lists
  3. Batch embed         – all sentences in one batched HuggingFace pass (fast)
  4. Semantic chunk      – cosine-threshold breakpoint grouping (configurable %)
  5. Upsert to ChromaDB  – fresh rebuild of 'edumentor_5k' collection
  6. Index               – VectorStoreIndex over the ChromaDB vector store

Run
───
  cd backend
  python -m rag.indexing.build_index_5k
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# ── Ensure backend root is importable ─────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from rag.config import (
    DATASET_5K_PATH,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_5K,
    EMBEDDING_MODEL_NAME,
    BREAKPOINT_PERCENTILE_THRESHOLD,
)
from rag.loaders.jsonl_loader import load_jsonl_conversations
from rag.database.chroma_manager import ChromaManager
from rag.indexing.custom_splitter import custom_sentence_splitter

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def cosine_distance(a: list, b: list) -> float:
    """Cosine distance (0 = identical, 1 = orthogonal, 2 = opposite)."""
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def semantic_chunk_document(sentences: list[str], embeddings: list[list], threshold_pct: int) -> list[str]:
    """
    Split a document's sentences into semantic chunks using cosine-distance breakpoints.

    Returns a list of chunk strings (sentences joined by double newlines).
    """
    if len(sentences) <= 1:
        return [" ".join(sentences)] if sentences else []

    # Compute pairwise cosine distances between adjacent sentences
    distances = [cosine_distance(embeddings[i], embeddings[i + 1]) for i in range(len(sentences) - 1)]

    # Breakpoint = sentences where cosine distance exceeds the Nth percentile
    threshold = np.percentile(distances, threshold_pct) if distances else 1.0

    chunks, current = [], [sentences[0]]
    for i, dist in enumerate(distances):
        if dist > threshold:
            chunks.append("\n\n".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ── Main ingestion function ────────────────────────────────────────────────────

def build_5k_index():
    logger.info("══════════════════════════════════════════════════════")
    logger.info("   BUILDING edumentor_5k RAG INDEX")
    logger.info("   Dataset : %s", DATASET_5K_PATH)
    logger.info("   Collection: %s", CHROMA_COLLECTION_5K)
    logger.info("══════════════════════════════════════════════════════")

    # ── Step 1 · Load & deduplicate ──────────────────────────────────────────
    logger.info("Step 1 › Loading and deduplicating dataset …")
    if not DATASET_5K_PATH.exists():
        logger.error("Dataset not found: %s", DATASET_5K_PATH)
        sys.exit(1)

    documents = load_jsonl_conversations(str(DATASET_5K_PATH), deduplicate=True)
    if not documents:
        logger.error("No documents loaded — check dataset path and format.")
        sys.exit(1)
    logger.info("  ✓ %d unique Q&A documents loaded.", len(documents))

    # ── Step 2 · Sentence-segment every document ─────────────────────────────
    logger.info("Step 2 › Segmenting documents into sentence units …")
    doc_sentences: list[list[str]] = []
    for doc in documents:
        doc_sentences.append(custom_sentence_splitter(doc.text))

    flat_sentences: list[str] = []
    sentence_map: list[tuple[int, int]] = []   # (doc_idx, sentence_idx)
    for doc_idx, sents in enumerate(doc_sentences):
        for s_idx, s in enumerate(sents):
            flat_sentences.append(s)
            sentence_map.append((doc_idx, s_idx))

    if not flat_sentences:
        logger.error("No sentences produced — check custom_sentence_splitter output.")
        sys.exit(1)
    logger.info("  ✓ %d sentences across %d documents.", len(flat_sentences), len(documents))

    # ── Step 3 · Batch-embed all sentences ───────────────────────────────────
    logger.info("Step 3 › Initialising embedding model: %s …", EMBEDDING_MODEL_NAME)
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

    logger.info("Step 3 › Generating embeddings for %d sentences (batched) …", len(flat_sentences))
    flat_embeddings = embed_model.get_text_embedding_batch(flat_sentences, show_progress=True)
    logger.info("  ✓ Embedding complete.")

    # Re-group embeddings back to per-document lists
    doc_embeddings: list[list[list]] = [[] for _ in range(len(documents))]
    for idx, emb in enumerate(flat_embeddings):
        doc_idx, _ = sentence_map[idx]
        doc_embeddings[doc_idx].append(emb)

    # ── Step 4 · Semantic chunking ───────────────────────────────────────────
    logger.info("Step 4 › Applying cosine-threshold semantic chunking (p%d) …",
                BREAKPOINT_PERCENTILE_THRESHOLD)

    nodes: list[TextNode] = []
    for doc_idx, doc in enumerate(documents):
        sents = doc_sentences[doc_idx]
        embs  = doc_embeddings[doc_idx]
        chunks = semantic_chunk_document(sents, embs, BREAKPOINT_PERCENTILE_THRESHOLD)
        for chunk_text in chunks:
            nodes.append(TextNode(text=chunk_text, metadata=doc.metadata))

    logger.info("  ✓ %d semantic nodes produced from %d documents.", len(nodes), len(documents))

    # ── Step 5 · Upsert into ChromaDB ────────────────────────────────────────
    logger.info("Step 5 › Rebuilding ChromaDB collection '%s' at %s …",
                CHROMA_COLLECTION_5K, CHROMA_PERSIST_DIR)

    # Temporarily override collection name by subclassing ChromaManager
    chroma_manager = ChromaManager(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_5K,
    )
    chroma_manager.initialize_db(rebuild_fresh=True)
    vector_store = chroma_manager.get_vector_store()
    logger.info("  ✓ ChromaDB collection ready.")

    # ── Step 6 · Build VectorStoreIndex ─────────────────────────────────────
    logger.info("Step 6 › Building VectorStoreIndex (this computes final node embeddings) …")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    logger.info("══════════════════════════════════════════════════════")
    logger.info("   ✓  INGESTION COMPLETE")
    logger.info("   Collection '%s' → %d chunks stored.", CHROMA_COLLECTION_5K, len(nodes))
    logger.info("══════════════════════════════════════════════════════")


if __name__ == "__main__":
    build_5k_index()
