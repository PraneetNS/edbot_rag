import sys
import math
import re
from pathlib import Path
from sentence_transformers import CrossEncoder

# Add parent path to import config
sys.path.append(str(Path(__file__).resolve().parent))
import config
from embedding_engine import EmbeddingEngine
from vector_store import EducationalVectorStore

class EducationalQueryRouter:
    """
    Orchestration layer that performs intent-aware query routing,
    retrieves candidate chunks from specialized collections,
    applies Cross-Encoder reranking, and outputs a true confidence score.
    """
    def __init__(self):
        # 1. Load local embedding and vector database
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = EducationalVectorStore()
        
        # 2. Load reranker model locally
        print(f"Loading local Reranker: {config.RERANKER_MODEL}...")
        self.reranker = CrossEncoder(config.RERANKER_MODEL)

    def _classify_query_intent(self, query: str) -> str:
        """
        Heuristically resolves student query intent to route to the correct collection.
        """
        q_lower = query.lower()
        if any(w in q_lower for w in ["placement", "resume", "interview", "hire", "recruit"]):
            return "PLACEMENT_GUIDANCE"
        elif any(w in q_lower for w in ["internship", "project guidelines"]):
            return "INTERNSHIP_GUIDANCE"
        elif any(w in q_lower for w in ["password", "login", "portal", "account", "issue"]):
            return "LMS_SUPPORT"
        elif any(w in q_lower for w in ["certificate", "vtu", "stamp"]):
            return "CERTIFICATION_SUPPORT"
        elif any(w in q_lower for w in ["exam", "test", "quiz", "failed", "grades", "dbms", "sql", "dsa", "operating system"]):
            return "EXAM_ASSISTANCE"
        elif any(w in q_lower for w in ["assignment", "homework", "project"]):
            return "ASSIGNMENT_HELP"
        elif any(w in q_lower for w in ["roadmap", "backend", "frontend", "cybersecurity", "ai"]):
            return "COURSE_QUERY"
        return "MENTORING"

    def _sigmoid(self, x: float) -> float:
        """
        Converts raw logit scores to standard 0.0 - 1.0 probability.
        """
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def route_and_retrieve(self, query: str, top_k: int = 5) -> dict:
        """
        Two-stage retrieval pipeline:
        Stage 1: Intent Routing + Dense collection retrieval (gets top 20 candidate chunks)
        Stage 2: ms-marco Cross-Encoder reranking down to Top 5 + Confidence computation
        """
        # 1. Resolve Query Intent and Collection Mapping
        intent = self._classify_query_intent(query)
        collection_name = config.COLLECTIONS.get(intent, "mentoring_collection")
        
        print(f"\nQuery: '{query}'")
        print(f"Routed Intent: {intent} --> Collection: {collection_name}")
        
        collection = self.vector_store.collections[collection_name]
        
        if collection.count() == 0:
            print("Warning: Collection is empty! Attempting broad search in other collections...")
            # Search placements or roadmaps as a broad fallback
            collection = self.vector_store.collections["placements_collection"]
            if collection.count() == 0:
                collection = self.vector_store.collections["roadmap_collection"]
                
        if collection.count() == 0:
            return {
                "intent": intent,
                "collection": collection_name,
                "retrieved_context": [],
                "retrieval_confidence": 0.0
            }

        # 2. Stage 1: Dense Retrieval
        query_emb = self.embedding_engine.get_query_embedding(query)
        
        # Retrieve up to 20 raw candidate records
        n_candidates = min(20, collection.count())
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=n_candidates
        )
        
        if not results or not results["documents"] or not results["documents"][0]:
            return {
                "intent": intent,
                "collection": collection_name,
                "retrieved_context": [],
                "retrieval_confidence": 0.0
            }

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        ids = results["ids"][0]

        # 3. Stage 2: Cross-Encoder Reranking
        pairs = [[query, doc] for doc in documents]
        
        try:
            scores = self.reranker.predict(pairs)
        except Exception as e:
            print(f"Warning: Reranker failed: {e}")
            scores = [1.0] * len(documents)

        # Structure hits with normalized scores
        hits = []
        highest_score = -999.0
        
        for idx, doc in enumerate(documents):
            raw_score = float(scores[idx])
            norm_score = self._sigmoid(raw_score)
            
            hits.append({
                "id": ids[idx],
                "content": doc,
                "metadata": metadatas[idx],
                "raw_score": raw_score,
                "score": norm_score
            })
            
            if raw_score > highest_score:
                highest_score = raw_score

        # Re-sort list by rerank score descending
        hits.sort(key=lambda x: x["score"], reverse=True)
        
        # Output Top 5 matches
        top_hits = hits[:top_k]
        
        # Calculate retrieval confidence score based on the highest ranked candidate
        retrieval_confidence = self._sigmoid(highest_score) if highest_score != -999.0 else 0.0

        return {
            "intent": intent,
            "collection": collection_name,
            "retrieved_context": top_hits,
            "retrieval_confidence": retrieval_confidence
        }

if __name__ == "__main__":
    router = EducationalQueryRouter()
    # Simple query test
    res = router.route_and_retrieve("How do I fix my LMS login issue?")
    print(f"Results: {len(res['retrieved_context'])} chunks retrieved (Confidence: {res['retrieval_confidence']:.3f})")
