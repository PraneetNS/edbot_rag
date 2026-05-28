import sys
import argparse
from pathlib import Path
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Define path
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_STORE_DIR = BASE_DIR / "chroma_store"

# Add backend directory to sys.path
sys.path.append(str(BASE_DIR))

from rag.hybrid_retriever import HybridRetriever
from rag.reranker import Reranker
from rag.preprocessor import preprocess_query

def query_kb(query_text: str, n_results: int = 3):
    if not CHROMA_STORE_DIR.exists():
        print(f"ChromaDB store not found at {CHROMA_STORE_DIR}. Please run ingestion first.")
        return

    # 1. Initialize upgraded embedding model
    print("Loading BAAI/bge-large-en-v1.5 embedding model...")
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5"
    )

    # 2. Get Chroma Collection
    print(f"Connecting to ChromaDB client at {CHROMA_STORE_DIR}...")
    client = chromadb.PersistentClient(path=str(CHROMA_STORE_DIR))
    COLLECTION_NAME = "educational_mentor_knowledgebase"
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' not found. Please run ingestion first.")
        return

    # 3. Build Vector Store Index
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )

    # 4. Initialize Modular Pipelines
    print("Initializing Hybrid Retriever and Cross-Encoder Reranker...")
    hybrid_retriever = HybridRetriever(index, collection, similarity_top_k=n_results * 2)
    reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # 5. Preprocess and Expand Query
    expanded_query = preprocess_query(query_text)
    print(f"\nOriginal Query: '{query_text}'")
    print(f"Expanded Query: '{expanded_query}'")

    # 6. Retrieve
    print("\nRetrieving candidates...")
    hybrid_hits = hybrid_retriever.retrieve(expanded_query)

    # 7. Rerank
    print("Reranking and applying Educational Boost...")
    final_hits, confidence = reranker.rerank(query_text, hybrid_hits, top_n=n_results)

    if not final_hits:
        print("No results found.")
        return

    print("\n" + "="*70)
    print(f"RETRIEVED RESULTS (Retrieval Confidence: {confidence:.4f}):")
    print("="*70)
    
    for idx, r in enumerate(final_hits):
        score_str = f"Score: {r.score:.4f}" if r.score is not None else "Score: N/A"
        metadata = r.node.metadata
        source = metadata.get('source', 'Unknown')
        topic = metadata.get('topic', 'General')
        edu_type = metadata.get('educational_type', 'N/A')
        domain = metadata.get('roadmap_domain', 'N/A')
        
        print(f"\n[{idx+1}] ID: {r.node.node_id} | Topic: {topic} | Type: {edu_type} | Domain: {domain} | {score_str}")
        print(f"Source: {source}")
        print("-" * 70)
        print(r.text.strip())
        print("-" * 70)

def main():
    parser = argparse.ArgumentParser(description="Query Edutainer RAG Knowledge Base")
    parser.add_argument("query", type=str, nargs="?", help="The search query to retrieve information for")
    parser.add_argument("-n", "--num-results", type=int, default=3, help="Number of results to retrieve")
    args = parser.parse_args()

    if not args.query:
        # Interactive mode
        print("Edutainer Upgraded RAG KB Query Tool. Type 'exit' to quit.\n")
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
