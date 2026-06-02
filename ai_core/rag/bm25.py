import re
import pickle
import logging
from pathlib import Path
from rank_bm25 import BM25Okapi
from llama_index.core.schema import TextNode, NodeWithScore
from ai_core.config import BM25_INDEX_PATH

logger = logging.getLogger(__name__)

def tokenize_text(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())

class BM25Searcher:
    """
    Pickle-persisted hybrid text search engine using rank-bm25.
    Instant launch via pre-indexed pickle serialization.
    """
    def __init__(self):
        self.nodes = []
        self.bm25 = None
        self.load_index()

    def build_and_save(self, nodes: list[TextNode]):
        logger.info(f"Building BM25 index with {len(nodes)} document nodes...")
        self.nodes = nodes
        corpus = [tokenize_text(node.text) for node in nodes]
        self.bm25 = BM25Okapi(corpus)
        
        # Serialize to pickle
        BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(BM25_INDEX_PATH, "wb") as f:
                pickle.dump({"bm25": self.bm25, "nodes": self.nodes}, f)
            logger.info(f"BM25 index successfully saved to: {BM25_INDEX_PATH}")
        except Exception as e:
            logger.error(f"Failed to serialize BM25 index: {e}")

    def load_index(self):
        if not BM25_INDEX_PATH.exists():
            logger.warning(f"BM25 index file not found at: {BM25_INDEX_PATH}. Index needs rebuilding.")
            return
            
        try:
            with open(BM25_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data.get("bm25")
            self.nodes = data.get("nodes", [])
            logger.info(f"BM25 index successfully loaded from: {BM25_INDEX_PATH} (Active nodes: {len(self.nodes)})")
        except Exception as e:
            logger.error(f"Error loading serialized BM25: {e}")

    def search(self, query: str, top_k: int = 30) -> list[NodeWithScore]:
        if self.bm25 is None or not self.nodes:
            logger.warning("BM25 index is not loaded. Skipping text search.")
            return []
            
        tokenized_query = tokenize_text(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort indices by score descending
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:  # Only return nodes with positive relevance matching
                node = self.nodes[idx]
                results.append(NodeWithScore(node=node, score=score))
                
        logger.info(f"BM25 Search complete for '{query[:30]}...' -> {len(results)} matches.")
        return results
