import logging
from llama_index.core.schema import NodeWithScore
from ai_core.config import RERANK_MODEL_NAME

logger = logging.getLogger(__name__)

class BGEReranker:
    """
    Modular Cross-Encoder Reranker using BAAI/bge-reranker-base.
    Calculates exact textual alignment between the user query and candidate segments.
    """
    def __init__(self):
        self.reranker = None
        self._initialize_model()

    def _initialize_model(self):
        logger.info(f"Loading Cross-Encoder Reranker: {RERANK_MODEL_NAME}...")
        try:
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker(RERANK_MODEL_NAME, use_fp16=True)
            logger.info("Reranker model successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to load FlagReranker: {e}. Falling back to default scoring.")

    def rerank(self, query: str, nodes: list[NodeWithScore], top_n: int = 5) -> list[NodeWithScore]:
        if not nodes:
            return []
            
        if self.reranker is None:
            logger.warning("Reranker is not active. Returning top retrieved items in original rank.")
            return nodes[:top_n]
            
        try:
            # Pair query with each candidate text
            pairs = [(query, n.node.text) for n in nodes]
            scores = self.reranker.compute_score(pairs)
            
            # Normalize single float vs array
            if isinstance(scores, float):
                scores = [scores]
                
            # Map scores to nodes
            for idx, score in enumerate(scores):
                nodes[idx].score = float(score)
                
            # Re-sort descending
            nodes.sort(key=lambda x: x.score or -9999.0, reverse=True)
            
            logger.info(f"Reranking complete. Top score: {nodes[0].score:.4f}")
            return nodes[:top_n]
        except Exception as e:
            logger.error(f"Error during Rerank: {e}")
            return nodes[:top_n]
