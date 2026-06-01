import os
import sys
import logging
from pathlib import Path

# Ensure backend directory is in the sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from rag.config import (
    DATASET_PATH,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    BREAKPOINT_PERCENTILE_THRESHOLD
)
from rag.loaders.jsonl_loader import load_jsonl_conversations
from rag.database.chroma_manager import ChromaManager
from rag.indexing.custom_splitter import custom_sentence_splitter

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_new_index():
    logger.info("==================================================")
    logger.info("      BUILDING NEW PURE ENGINEERING RAG INDEX")
    logger.info("==================================================")

    # 1. Load data
    logger.info(f"Loading mentoring dataset from: {DATASET_PATH}")
    if not DATASET_PATH.exists():
        logger.error(f"Dataset path does not exist: {DATASET_PATH}")
        sys.exit(1)
        
    documents = load_jsonl_conversations(str(DATASET_PATH))
    if not documents:
        logger.error("No valid conversations were loaded. Exiting.")
        sys.exit(1)

    # 2. Configure embedding model
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME
    )

    # 3. Fast batched semantic chunker
    logger.info("Setting up high-performance batch semantic splitter...")
    
    # 3.1. Split each document into sentences
    logger.info("Segmenting documents into sentence units...")
    doc_sentences = []
    for doc in documents:
        sentences = custom_sentence_splitter(doc.text)
        doc_sentences.append(sentences)
        
    # 3.2. Flatten sentences to embed in batch
    flat_sentences = []
    sentence_map = [] # stores (doc_idx, sentence_idx)
    for doc_idx, sentences in enumerate(doc_sentences):
        for s_idx, s in enumerate(sentences):
            flat_sentences.append(s)
            sentence_map.append((doc_idx, s_idx))
            
    if not flat_sentences:
        logger.error("No sentences found to split. Exiting.")
        sys.exit(1)
        
    # 3.3. Generate embeddings in large batches (extremely fast!)
    logger.info(f"Generating embeddings for {len(flat_sentences)} sentences in batched pipeline...")
    flat_embeddings = embed_model.get_text_embedding_batch(flat_sentences, show_progress=True)
    
    # 3.4. Map embeddings back to documents
    doc_embeddings = [[] for _ in range(len(documents))]
    for idx, emb in enumerate(flat_embeddings):
        doc_idx, s_idx = sentence_map[idx]
        doc_embeddings[doc_idx].append(emb)
        
    # 3.5. Perform semantic chunking with HNSW Cosine thresholding
    import numpy as np
    from llama_index.core.schema import TextNode
    
    def cosine_distance(a, b):
        a = np.array(a)
        b = np.array(b)
        if np.all(a == 0) or np.all(b == 0):
            return 0.0
        return 1.0 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        
    logger.info("Applying cosine thematic similarity boundaries...")
    nodes = []
    for doc_idx, doc in enumerate(documents):
        sentences = doc_sentences[doc_idx]
        embeddings = doc_embeddings[doc_idx]
        
        if len(sentences) <= 1:
            node = TextNode(
                text=doc.text,
                metadata=doc.metadata
            )
            nodes.append(node)
            continue
            
        distances = []
        for i in range(len(sentences) - 1):
            dist = cosine_distance(embeddings[i], embeddings[i+1])
            distances.append(dist)
            
        if distances:
            threshold = np.percentile(distances, BREAKPOINT_PERCENTILE_THRESHOLD)
        else:
            threshold = 1.0
            
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(len(sentences) - 1):
            if distances[i] > threshold:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [sentences[i+1]]
            else:
                current_chunk.append(sentences[i+1])
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        for chunk in chunks:
            node = TextNode(
                text=chunk,
                metadata=doc.metadata
            )
            nodes.append(node)
            
    logger.info(f"Successfully generated {len(nodes)} high-quality semantic chunks.")

    # 5. Initialize ChromaDB (rebuild fresh)
    logger.info(f"Rebuilding database at: {CHROMA_PERSIST_DIR}")
    chroma_manager = ChromaManager(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME
    )
    chroma_manager.initialize_db(rebuild_fresh=True)
    vector_store = chroma_manager.get_vector_store()

    # 6. Store in Vector Store Index
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
