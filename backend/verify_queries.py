import sys
import os
import requests
import json

BASE_URL = "http://localhost:8000"

def get_rag_score_distribution():
    print("\n=== RAG Score Distribution for: 'I am doing arrays what should I do next' ===")
    
    # Setup path so we can import modules
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    
    from rag.retrieval.retriever import get_edmentor_retriever, PriorityTopicPostprocessor, get_cached_reranker
    from llama_index.core import QueryBundle
    
    query = "I am doing arrays what should I do next"
    
    # 1. Retrieve nodes
    retriever = get_edmentor_retriever(topic="Dsa", top_k=6)
    nodes = retriever.retrieve(query)
    print("\n--- 1. Raw Vector Retrieval Chunks & Cosine Scores ---")
    for idx, nws in enumerate(nodes, 1):
        print(f"[{idx}] Text: {nws.node.text[:80].strip()}...\n    Raw Score (Cosine): {nws.score}")
        
    # 2. Apply reranker
    reranker = get_cached_reranker(3)
    nodes = reranker.postprocess_nodes(nodes, QueryBundle(query))
    print("\n--- 2. After Reranking (Reranker Logits) ---")
    for idx, nws in enumerate(nodes, 1):
        print(f"[{idx}] Text: {nws.node.text[:80].strip()}...\n    Rerank Score (Logit): {nws.score}")
        
    # 3. Apply min-max normalization & priority boost
    priority_pp = PriorityTopicPostprocessor()
    nodes = priority_pp.postprocess_nodes(nodes, QueryBundle(query))
    print("\n--- 3. After Min-Max Normalization & Topic Boost ---")
    for idx, nws in enumerate(nodes, 1):
        print(f"[{idx}] Text: {nws.node.text[:80].strip()}...\n    Normalized & Boosted Score: {nws.score}")


def test_six_queries():
    queries = [
        "I am doing arrays what should I do next.",
        "I have 60 days for placements how should I prepare.",
        "my resume has no projects what do I do.",
        "explain recursion to me like I am a beginner.",
        "I have a backlog will it affect my placement.",
        "what should I learn after arrays for DSA."
    ]
    
    print("\n=== Testing Six Queries ===")
    
    for idx, q in enumerate(queries, 1):
        print(f"\nQuery {idx}: '{q}'")
        payload = {
            "question": q,
            "session_id": f"verify_session_{idx}"
        }
        
        try:
            r = requests.post(f"{BASE_URL}/edmentor/query", json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                response = data.get("response", "")
                word_count = data.get("word_count", 0)
                topic = data.get("topic", "unknown")
                print(f"Topic: {topic}")
                print(f"Word Count: {word_count}")
                print(f"Response:\n{response}")
            else:
                print(f"Status Error: {r.status_code}\n{r.text}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    get_rag_score_distribution()
    test_six_queries()
