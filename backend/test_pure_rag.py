import sys
import logging
from pathlib import Path

# Ensure backend directory is in the sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

# Quiet unnecessary logs
logging.basicConfig(level=logging.WARNING)
logging.getLogger("rag").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

from rag.retrieval.retriever import load_rag_index, get_edumentor_query_engine

def run_pure_rag_tests():
    print("==================================================")
    print("      RUNNING PURE ENGINEERING MENTOR RAG TESTS")
    print("==================================================")
    
    # 1. Load persisted index
    try:
        index = load_rag_index()
    except Exception as e:
        print(f"\nError loading database: {e}")
        print("Please make sure ingestion was completed by running Option 1 in main.py first.")
        sys.exit(1)
        
    query_engine = get_edumentor_query_engine(index)
    
    # 2. Define target test cases
    test_questions = [
        "How should I prepare for placements?",
        "Explain dynamic programming",
        "Suggest machine learning projects",
        "How do I get my first internship?",
        "How should I improve my resume?"
    ]
    
    # 3. Run and print results
    for idx, question in enumerate(test_questions, 1):
        print(f"\n[TEST CASE {idx}] Question: \"{question}\"")
        print("-" * 60)
        
        try:
            response = query_engine.query(question)
            
            # Print retrieved chunks & metadata
            source_nodes = getattr(response, "source_nodes", [])
            print("Retrieved Chunks & Reranked Cosine Similarity:")
            if not source_nodes:
                print("  No matching chunks retrieved.")
            for chunk_idx, nws in enumerate(source_nodes):
                score = nws.score if nws.score is not None else 0.0
                topic = nws.node.metadata.get("topic", "General")
                source = nws.node.metadata.get("source", "Unknown")
                print(f"  ({chunk_idx+1}) [Score: {score:.4f}] | Topic: {topic} | Source: {source}")
                snippet = nws.node.text.replace("\n", " ")[:160].strip()
                print(f"      Snippet: {snippet}...")
            
            # Print final synthesized response
            print("\nSynthesized EduMentor Mentor Response:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
        except Exception as e:
            print(f"Error during test case execution: {e}")
            
        print("=" * 60)

if __name__ == "__main__":
    run_pure_rag_tests()
