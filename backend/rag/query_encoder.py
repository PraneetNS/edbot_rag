from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class QueryEncoder:
    """
    Handles query embedding generation using the state-of-the-art BAAI/bge-large-en-v1.5 model (1024 dimensions).
    """
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        print(f"Initializing QueryEncoder with {model_name}...")
        self.embed_model = HuggingFaceEmbedding(model_name=model_name)

    def get_embedding(self, text: str) -> list[float]:
        """
        Generates dense vector representation for the query.
        """
        return self.embed_model.get_text_embedding(text)
