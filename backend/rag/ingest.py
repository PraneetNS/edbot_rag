import sys
from pathlib import Path
import shutil
import chromadb

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Robust path setup relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "cleaned_docs"
CHROMA_DIR = BASE_DIR / "chroma_store"

def main():
    print(f"Loading cleaned documents from {CLEANED_DIR}...")
    if not CLEANED_DIR.exists() or not any(CLEANED_DIR.iterdir()):
        print(f"Error: Cleaned documents directory '{CLEANED_DIR}' is empty or does not exist.")
        print("Please run document_cleaner.py first.")
        return

    documents = SimpleDirectoryReader(str(CLEANED_DIR)).load_data()
    print(f"Loaded {len(documents)} document pages/files.")

    print("Initializing HuggingFaceEmbedding (sentence-transformers/all-MiniLM-L6-v2)...")
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print(f"Initializing ChromaDB PersistentClient at {CHROMA_DIR}...")
    # Delete old database collection files to rebuild completely fresh
    if CHROMA_DIR.exists():
        print("Removing existing Chroma DB store for a completely fresh rebuild...")
        try:
            # We try to connect and delete the collection first
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            if "edubot" in [c.name for c in client.list_collections()]:
                client.delete_collection("edubot")
                print("Deleted existing Chroma collection 'edubot'.")
        except Exception as e:
            print(f"Note: Could not cleanly delete collection via client ({e}). Cleaning folder...")
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    CHROMA_DIR.mkdir(exist_ok=True, parents=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    print("Creating fresh ChromaDB collection 'edubot'...")
    collection = client.create_collection("edubot")

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # Smart, sentence-aware semantic chunking
    print("Setting up SentenceSplitter (chunk_size=500, chunk_overlap=90)...")
    node_parser = SentenceSplitter(
        chunk_size=500,
        chunk_overlap=90
    )

    print("Building VectorStoreIndex from cleaned documents...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[node_parser],
        show_progress=True
    )

    print("RAG clean semantic knowledge base created successfully.")

if __name__ == "__main__":
    main()
