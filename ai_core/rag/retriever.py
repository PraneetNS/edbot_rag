import os
import sys
import time
import pickle
import asyncio
import logging
from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Allow imports from backend root
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
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    VECTOR_TOP_K,
    BM25_TOP_K,
    FINAL_TOP_K,
    RAG_MODE,
    MODES,
    PARENT_DOCS_PATH,
    RAG_LOGS_PATH,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL
)
from rag.database.chroma_manager import ChromaManager
from ai_core.rag.bm25 import BM25Searcher
from ai_core.rag.reranker import BGEReranker
from ai_core.rag.validator import ContextValidator
from ai_core.rag.compressor import ContextCompressor

logger = logging.getLogger(__name__)

class EduMentorAsyncRetriever:
    """
    Async Hybrid search-and-rerank retriever with dynamic confidence gating.
    Integrates BAAI BGE vector search + rank-bm25 text search, deduplication,
    FlagReranker, validation checks, and full parent document mappings.
    """
    def __init__(self, index: VectorStoreIndex = None):
        self.index = index
        self.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        
        # Load hybrid searches & rankings
        self.bm25_searcher = BM25Searcher()
        self.reranker = BGEReranker()
        self.validator = ContextValidator()
        self.compressor = ContextCompressor()
        
        # Load Parent Documents mappings
        self.parent_docs = {}
        self.load_parent_docs()

    def load_parent_docs(self):
        if not PARENT_DOCS_PATH.exists():
            logger.warning(f"Parent documents map not found at: {PARENT_DOCS_PATH}")
            return
        try:
            with open(PARENT_DOCS_PATH, "rb") as f:
                self.parent_docs = pickle.load(f)
            logger.info(f"Parent documents successfully loaded from: {PARENT_DOCS_PATH} ({len(self.parent_docs)} docs mapped)")
        except Exception as e:
            logger.error(f"Error loading parent docs store: {e}")

    async def _async_vector_search(self, query: str) -> list[NodeWithScore]:
        """Runs BGE vector search asynchronously in a thread pool."""
        try:
            loop = asyncio.get_running_loop()
            retriever = self.index.as_retriever(similarity_top_k=VECTOR_TOP_K)
            # Retrieve synchronously inside executor to keep it async-safe
            results = await loop.run_in_executor(None, retriever.retrieve, query)
            logger.info(f"Vector search retrieved {len(results)} candidate chunks.")
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _async_bm25_search(self, query: str) -> list[NodeWithScore]:
        """Runs BM25 keyword search asynchronously in a thread pool."""
        try:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, self.bm25_searcher.search, query, BM25_TOP_K)
            logger.info(f"BM25 search retrieved {len(results)} candidate chunks.")
            return results
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _deduplicate_and_merge(self, vector_hits: list[NodeWithScore], bm25_hits: list[NodeWithScore]) -> list[NodeWithScore]:
        """Merges two rankings, removing exact duplicate node IDs."""
        seen_nodes = set()
        merged = []
        
        # Prioritize BGE vector matches
        for hit in vector_hits:
            node_id = hit.node.node_id
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                merged.append(hit)
                
        # Fill in unmatched BM25 text matches
        for hit in bm25_hits:
            node_id = hit.node.node_id
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                merged.append(hit)
                
        logger.info(f"Merged hybrid search result sets: {len(vector_hits)} Vec + {len(bm25_hits)} BM25 -> {len(merged)} deduplicated chunks.")
        return merged

    async def retrieve_pipeline(self, query: str, session_state: dict = None) -> tuple[bool, str, dict]:
        """
        Executes the async hybrid retrieval pipeline:
        1. Async Vector Search + BM25 Search
        2. Deduplication Merge
        3. Cross-Encoder Reranking
        4. Dynamic Confidence Gate checking (Accept/LLM Judge/Reject)
        5. Parent Document Context Resolution
        6. Sentence de-duplication Context Compression
        7. Diagnostics debug logging
        """
        start_time = time.time()
        
        # Setup session/memory profile variables for logs
        session_id = session_state.get("session_id", "default") if session_state else "default"
        intent = session_state.get("active_intent", "knowledge_lookup") if session_state else "knowledge_lookup"
        
        # 1. Dispatch Vector and BM25 search concurrently
        logger.info(f"Initiating Hybrid Async retrieval for query: '{query}'")
        vector_task = asyncio.create_task(self._async_vector_search(query))
        bm25_task = asyncio.create_task(self._async_bm25_search(query))
        
        vector_hits, bm25_hits = await asyncio.gather(vector_task, bm25_task)
        
        # 2. Merge deduplicated results
        merged_hits = self._deduplicate_and_merge(vector_hits, bm25_hits)
        if not merged_hits:
            logger.info("Hybrid search returned zero results. Bypassing RAG.")
            return False, "", self._build_debug_log(query, intent, 0, 0, -100.0, False, start_time)
            
        # 3. Compute Rerank scores
        logger.info("Computing Cross-Encoder alignment rankings...")
        reranked_hits = self.reranker.rerank(query, merged_hits, top_n=FINAL_TOP_K)
        best_score = reranked_hits[0].score if reranked_hits else -100.0
        
        # 4. Confidence Gate Logic
        mode_thresholds = MODES.get(RAG_MODE, MODES["balanced"])
        high_threshold = mode_thresholds["high"]
        low_threshold = mode_thresholds["low"]
        
        use_rag = False
        decision = "reject"
        
        logger.info(f"Confidence Gate [Mode: {RAG_MODE}] - Best Score: {best_score:.4f} | Bounds: [{low_threshold:.2f}, {high_threshold:.2f}]")
        
        if best_score >= high_threshold:
            use_rag = True
            decision = "accept"
            logger.info("Reranker confidence is HIGH. RAG context immediately ACCEPTED.")
        elif best_score < low_threshold:
            use_rag = False
            decision = "reject"
            logger.info("Reranker confidence is LOW. RAG context immediately REJECTED.")
        else:
            decision = "validator"
            logger.info("Reranker confidence is UNCERTAIN. Dispatched LLM validator...")
            # Combine the texts of top retrieved nodes for validation
            validation_context = "\n\n".join([h.node.text for h in reranked_hits[:3]])
            use_rag = self.validator.validate(query, validation_context)
            
        if not use_rag:
            logger.info("RAG context was rejected or bypassed. Routing directly to direct LLM response.")
            return False, "", self._build_debug_log(query, intent, len(bm25_hits), len(vector_hits), best_score, False, start_time)

        # 5. Parent Document Context Retrieval
        logger.info("Resolving granular chunks to parent document contexts...")
        context_chunks = []
        resolved_parents_count = 0
        
        for idx, hit in enumerate(reranked_hits):
            parent_id = hit.node.metadata.get("parent_doc_id")
            if parent_id and parent_id in self.parent_docs:
                parent_text = self.parent_docs[parent_id]
                context_chunks.append(parent_text)
                resolved_parents_count += 1
            else:
                # Fallback to chunk text if parent missing
                context_chunks.append(hit.node.text)
                
        raw_context = "\n\n".join(context_chunks)
        logger.info(f"Parent Document lookup completed. Resolved {resolved_parents_count}/{len(reranked_hits)} full parent documents.")
        
        # 6. Sentence Context Compression
        compressed_context = self.compressor.compress(raw_context)
        
        # 7. Write debug log
        debug_log = self._build_debug_log(query, intent, len(bm25_hits), len(vector_hits), best_score, True, start_time)
        self._write_logs_to_file(debug_log)
        
        return True, compressed_context, debug_log

    def _build_debug_log(self, query: str, intent: str, bm25_count: int, vector_count: int, score: float, use_rag: bool, start_time: float) -> dict:
        latency = int((time.time() - start_time) * 1000)
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "intent": intent,
            "bm25_results": bm25_count,
            "vector_results": vector_count,
            "rerank_score": round(score, 4),
            "use_rag": use_rag,
            "latency_ms": latency
        }

    def _write_logs_to_file(self, log_entry: dict):
        RAG_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        logs = []
        if RAG_LOGS_PATH.exists():
            try:
                with open(RAG_LOGS_PATH, "r", encoding="utf-8") as f:
                    logs = pickle.load(f) if RAG_LOGS_PATH.suffix == ".pkl" else json.load(f)
            except Exception:
                logs = []
        
        logs.append(log_entry)
        # Keep last 500 logs to prevent infinite log growth
        logs = logs[-500:]
        
        try:
            import json
            with open(RAG_LOGS_PATH, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write diagnostic logs: {e}")

# Helper loader for index v3
def load_rag_index_v3() -> VectorStoreIndex:
    chroma_manager = ChromaManager(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME
    )
    vector_store = chroma_manager.get_vector_store()
    
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )
    return index
