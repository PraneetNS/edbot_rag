import sys
import argparse
from pathlib import Path
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Define path
CHROMA_STORE_DIR = Path(__file__).parent.parent / "chroma_store"

def query_kb(query_text: str, n_results: int = 2):
    if not CHROMA_STORE_DIR.exists():
        print(f"ChromaDB store not found at {CHROMA_STORE_DIR}. Please run ingestion first.")
        return

    # 1. Initialize same embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Get Chroma Collection
    client = chromadb.PersistentClient(path=str(CHROMA_STORE_DIR))
    try:
        collection = client.get_collection(name="edubot")
    except Exception:
        print("Collection 'edubot' not found. Please run ingestion first.")
        return

    # 3. Build Vector Store Index
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )

    # 4. Create retriever
    retriever = index.as_retriever(similarity_top_k=n_results)
    
    print(f"Querying LlamaIndex: '{query_text}' (fetching top {n_results} results)...")
    results = retriever.retrieve(query_text)

    if not results:
        print("No results found.")
        return

    print("\n" + "="*50)
    print("RETRIEVED RESULTS:")
    print("="*50)
    
    for idx, r in enumerate(results):
        score_str = f"Similarity: {r.score:.4f}" if r.score is not None else "Similarity: N/A"
        source = r.node.metadata.get('file_name', 'Unknown')
        print(f"\n[{idx+1}] ID: {r.node.node_id} | Source: {source} | {score_str}")
        print("-" * 50)
        print(r.text.strip())
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Query Edutainer RAG Knowledge Base")
    parser.add_argument("query", type=str, nargs="?", help="The search query to retrieve information for")
    parser.add_argument("-n", "--num-results", type=int, default=2, help="Number of results to retrieve")
    args = parser.parse_args()

    if not args.query:
        # Interactive mode
        print("Edutainer RAG KB Query Tool. Type 'exit' to quit.\n")
        while True:
            try:
                q = input("\nEnter your query: ").strip()
                if not q:
                    continue
                if q.lower() == 'exit':
                    break
                query_kb(q, args.num_results)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        query_kb(args.query, args.num_results)

if __name__ == "__main__":
    main()

