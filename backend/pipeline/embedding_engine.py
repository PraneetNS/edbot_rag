import sys
from pathlib import Path
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Add parent path to import config
sys.path.append(str(Path(__file__).resolve().parent))
import config

class EmbeddingEngine:
    """
    Handles local query and text embeddings generation using BAAI/bge-small-en-v1.5
    under a cached workspace singleton to minimize memory load.
    """
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        print(f"Loading local Embedding Engine: {model_name}...")
        self.embed_model = HuggingFaceEmbedding(model_name=model_name)

    def get_text_embedding(self, text: str) -> list[float]:
        """
        Generates dense vector representation for educational text.
        """
        return self.embed_model.get_text_embedding(text)

    def get_query_embedding(self, query: str) -> list[float]:
        """
        Generates dense vector representation for student query text.
        """
        return self.embed_model.get_query_embedding(query)

if __name__ == "__main__":
    engine = EmbeddingEngine()
    emb = engine.get_text_embedding("Master arrays and hashing for DSA placements.")
    print(f"Embedding length: {len(emb)}") # Should print 384 for bge-small
