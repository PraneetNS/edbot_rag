import logging
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ai_core.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

INTENT_TEMPLATES = {
    "rag": [
        "what courses are available on the portal?",
        "explain VTU co-branded certifications",
        "college placement preparation and internship opportunities",
        "how can I contact the academic support team?",
        "LMS login errors and help tickets",
        "React JS training syllabus and fee structure",
        "what is the placement preparation roadmap?",
        "how to download certificate from my profile?"
    ],
    "direct_llm": [
        "explain recursion in simple terms",
        "teach me dynamic programming patterns",
        "what is JavaScript closure?",
        "solve this programming algorithm",
        "how does an operating system manage memory?",
        "explain binary search tree time complexity",
        "what is object oriented programming?"
    ],
    "memory": [
        "what did I ask yesterday?",
        "continue my previous topic",
        "what is my target domain?",
        "what was my weak subject as discussed before?",
        "tell me about my active goal"
    ]
}

class IntentClassifier:
    """
    Lightning-fast, embedding-based Cosine Similarity classifier.
    Bypasses LLM generation latency by comparing BGE embeddings against intent templates.
    """
    def __init__(self, embed_model: HuggingFaceEmbedding = None):
        if embed_model is None:
            logger.info("Initializing HuggingFaceEmbedding in classifier...")
            self.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        else:
            self.embed_model = embed_model
            
        self.precomputed_embeddings = {}
        self._precompute_templates()

    def _precompute_templates(self):
        logger.info("Pre-computing Cosine templates for Intent Router...")
        for intent, templates in INTENT_TEMPLATES.items():
            embeddings = []
            for t in templates:
                try:
                    emb = self.embed_model.get_query_embedding(t)
                    embeddings.append(emb)
                except Exception as e:
                    logger.error(f"Failed to embed template '{t}': {e}")
            self.precomputed_embeddings[intent] = embeddings
        logger.info("Intent Router templates pre-computed successfully.")

    def classify(self, query: str) -> str:
        try:
            query_emb = np.array(self.embed_model.get_query_embedding(query))
            
            best_intent = "direct_llm"
            best_score = -1.0
            
            for intent, templates_embs in self.precomputed_embeddings.items():
                for t_emb in templates_embs:
                    t_emb_arr = np.array(t_emb)
                    # Compute cosine similarity
                    norm_q = np.linalg.norm(query_emb)
                    norm_t = np.linalg.norm(t_emb_arr)
                    if norm_q == 0 or norm_t == 0:
                        similarity = 0.0
                    else:
                        similarity = np.dot(query_emb, t_emb_arr) / (norm_q * norm_t)
                        
                    if similarity > best_score:
                        best_score = similarity
                        best_intent = intent
                        
            logger.info(f"Intent classified: '{query}' -> '{best_intent}' (Cosine score: {best_score:.4f})")
            return best_intent
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return "direct_llm"  # Safe default fallback
