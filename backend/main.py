import sys
import logging
from pathlib import Path

# Fix Windows cp1252 console encoding (crashes on special chars without this)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Add backend directory to system path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from rag.indexing.build_index import build_new_index
from rag.retrieval.retriever import load_rag_index, get_edumentor_query_engine, check_ollama_active
from rag.config import DATASET_PATH, CHROMA_PERSIST_DIR

def show_menu():
    print("\n" + "═"*50)
    print("        EDUMENTOR AI RAG SYSTEM CONTROL")
    print("═"*50)
    print("  1. Rebuild Engineering Mentor Knowledge Base (Fresh Ingest)")
    print("  2. Run Semantic Query (Single Search & Rerank)")
    print("  3. Chat with EduMentor (Interactive Console)")
    print("  4. System Diagnostics (Check Collection & Ollama Status)")
    print("  5. Exit")
    print("═"*50)

def run_diagnostics():
    print("\n--- SYSTEM DIAGNOSTICS ---")
    print(f"Dataset path: {DATASET_PATH} [EXIST]" if DATASET_PATH.exists() else f"Dataset path: {DATASET_PATH} [MISSING]")
    print(f"Chroma persistence directory: {CHROMA_PERSIST_DIR} [EXIST]" if CHROMA_PERSIST_DIR.exists() else f"Chroma directory: {CHROMA_PERSIST_DIR} [NOT CREATED YET]")
    
    ollama_ok = check_ollama_active()
    print(f"Ollama Local Daemon: {'ONLINE (Ready for Mistral)' if ollama_ok else 'OFFLINE (Running in high-quality fallback retriever mode)'}")
    
    try:
        index = load_rag_index()
        # Verify collection size
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        from rag.config import CHROMA_COLLECTION_NAME
        col = client.get_collection(CHROMA_COLLECTION_NAME)
        print(f"Chroma Collection Count: {col.count()} active semantic nodes.")
    except Exception as e:
        print(f"Chroma Collection Status: Not initialized or empty yet. (Error: {e})")
    print("-------------------------\n")

def query_kb():
    try:
        index = load_rag_index()
    except Exception as e:
        print(f"\nError: Could not load index. Please run Option 1 (Fresh Ingest) first. Details: {e}")
        return

    q = input("\nEnter student question: ").strip()
    if not q:
        return
        
    print("\nSearching and synthesizing response...")
    query_engine = get_edumentor_query_engine(index)
    res = query_engine.query(q)
    
    print("\n" + "─"*50)
    print("                    EDUMENTOR RESPONSE")
    print("─"*50)
    if hasattr(res, 'response_gen') and res.response_gen:
        for chunk in res.response_gen:
            print(chunk, end="", flush=True)
        print()
    else:
        print(res.response)
    print("─"*50)
    
    # Print source nodes for clarity
    source_nodes = getattr(res, "source_nodes", [])
    if source_nodes:
        print("\n[Retrieved Chunks & Similarity Scores]")
        for idx, nws in enumerate(source_nodes):
            score_str = f"Score: {nws.score:.4f}" if nws.score is not None else "Score: N/A"
            topic = nws.node.metadata.get("topic", "General")
            source = nws.node.metadata.get("source", "Unknown")
            print(f"  {idx+1}. Topic: {topic} | Source: {source} | {score_str}")
            # Snippet of text
            snippet = nws.node.text.replace("\n", " ")[:150] + "..."
            print(f"     Snippet: {snippet}")
        print("─"*50)

def chat_console():
    try:
        index = load_rag_index()
    except Exception as e:
        print(f"\nError: Could not load index. Please run Option 1 (Fresh Ingest) first. Details: {e}")
        return

    print("\n" + "═"*50)
    print("   EduMentor Interactive Session Launched")
    print("   Type 'exit' or 'quit' to end the session.")
    print("═"*50)
    
    query_engine = get_edumentor_query_engine(index)
    
    while True:
        try:
            query = input("\nStudent: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("Ending session. Keep coding and growing!")
                break
                
            response = query_engine.query(query)
            print(f"\nEduMentor: ", end="")
            if hasattr(response, 'response_gen') and response.response_gen:
                for chunk in response.response_gen:
                    print(chunk, end="", flush=True)
                print()
            else:
                print(response.response)
            
        except (KeyboardInterrupt, EOFError):
            print("\nEnding session. Keep coding and growing!")
            break

def main():
    while True:
        show_menu()
        try:
            choice = input("Enter choice (1-5): ").strip()
            if choice == "1":
                print("\nInitiating fresh index build from mentoring conversations dataset...")
                build_new_index()
            elif choice == "2":
                query_kb()
            elif choice == "3":
                chat_console()
            elif choice == "4":
                run_diagnostics()
            elif choice == "5":
                print("\nExiting EduMentor Control Panel. Keep mentoring and learning!")
                break
            else:
                print("Invalid selection, please enter 1-5.")
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"An unexpected system error occurred: {e}")

if __name__ == "__main__":
    main()
