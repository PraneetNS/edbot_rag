import sys
from pathlib import Path
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    StorageContext
)

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Robust path setup relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_store"

def main():
    if not CHROMA_DIR.exists():
        print(f"Error: ChromaDB directory '{CHROMA_DIR}' does not exist. Please run ingest.py first.")
        return

    print("Initializing HuggingFaceEmbedding (sentence-transformers/all-MiniLM-L6-v2)...")
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print(f"Connecting to ChromaDB client at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    print("Loading collection 'edubot'...")
    collection = client.get_or_create_collection("edubot")

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    print("Loading index from vector store...")
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )

    retriever = index.as_retriever(similarity_top_k=2)

    print("\nEduBot Retriever is ready. Type 'exit' or 'quit' to end.")
    while True:
        try:
            query = input("\nAsk Question: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                break

            results = retriever.retrieve(query)

            print("\n================ RETRIEVED CHUNKS ================")
            if not results:
                print("No relevant chunks found.")
            for idx, r in enumerate(results):
                score_str = f" (Score: {r.score:.4f})" if r.score is not None else ""
                print(f"\n[{idx+1}] Chunk ID: {r.node.node_id}{score_str}")
                print("-" * 50)
                print(r.text.strip())
                print("-" * 50)
            print("==================================================")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
