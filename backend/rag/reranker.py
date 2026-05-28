import math
import re
from sentence_transformers import CrossEncoder

class Reranker:
    """
    Implements a precise Cross-Encoder Reranker using cross-encoder/ms-marco-MiniLM-L-6-v2
    to re-score and re-order chunks, integrating a custom domain-specific
    Educational Prioritization Boost.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Initializing Reranker with {model_name}...")
        self.model = CrossEncoder(model_name)

    def _sigmoid(self, x: float) -> float:
        """
        Maps raw cross-encoder logits (typically -10 to +10) to a clean 0.0 - 1.0 probability.
        """
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def rerank(self, query: str, hits: list, top_n: int = 3) -> tuple[list, float]:
        """
        Reranks retrieved hits based on CrossEncoder true semantic matching,
        applying an Educational Prioritization Boost to mentoring and instructional chunks.
        Returns: (reranked_hits, retrieval_confidence)
        """
        if not hits:
            return [], 0.0

        # Create pairs of (query, document text)
        pairs = []
        for h in hits:
            text = h.node.text if hasattr(h, "node") else getattr(h, "text", "")
            pairs.append([query, text])

        # Compute cross-encoder relevance scores
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            print(f"Warning: Reranker prediction failed: {e}")
            # Fallback to current scores if model fails
            scores = [h.score if h.score is not None else 0.0 for h in hits]

        # Attach reranked score to each hit
        reranked_hits = []
        highest_score = -999.0
        
        for idx, h in enumerate(hits):
            score_logit = float(scores[idx])
            normalized_score = self._sigmoid(score_logit)
            
            # Apply Educational Prioritization Boost
            # Boost chunks containing step-by-step guidance, roadmaps, mentoring, or career prep
            text = h.node.text.lower() if hasattr(h, "node") else getattr(h, "text", "").lower()
            metadata = h.node.metadata if hasattr(h, "node") else {}
            
            educational_type = metadata.get("educational_type", "")
            
            # Boost checks:
            is_roadmap_or_mentoring = educational_type in ["roadmap", "mentoring_guidance", "placement_guide"]
            has_instructional_keywords = any(kw in text for kw in [
                "step 1", "step 2", "step 3", "roadmap", "curriculum", "syllabus", 
                "guide", "how to become", "prepare for", "career advice", "placement preparation"
            ])
            
            if is_roadmap_or_mentoring or has_instructional_keywords:
                # Add +0.12 educational boost, capped at 1.0
                normalized_score = min(1.0, normalized_score + 0.12)
            
            h.score = normalized_score
            reranked_hits.append(h)
            
            if score_logit > highest_score:
                highest_score = score_logit

        # Re-sort hits based on true reranked scores
        reranked_hits.sort(key=lambda x: x.score, reverse=True)

        # Calculate retrieval confidence based on the highest scored logit
        retrieval_confidence = self._sigmoid(highest_score) if highest_score != -999.0 else 0.0

        return reranked_hits[:top_n], retrieval_confidence
