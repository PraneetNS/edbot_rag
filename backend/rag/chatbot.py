import sys
from pathlib import Path
import chromadb
import requests

from llama_index.core import (
    VectorStoreIndex,
    StorageContext
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Import system prompt
sys.path.append(str(Path(__file__).resolve().parent))
from prompt import SYSTEM_PROMPT

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_store"

def check_ollama_running() -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def main():
    if not CHROMA_DIR.exists():
        print(f"Error: ChromaDB store not found at {CHROMA_DIR}. Please run ingest.py first.")
        return

    # Check if Ollama is running
    ollama_running = check_ollama_running()
    if not ollama_running:
        print("\n" + "!"*60)
        print("WARNING: Ollama service is not running locally on http://localhost:11434.")
        print("Please start the Ollama desktop app and run 'ollama pull mistral'.")
        print("The chatbot will run in FALLBACK MODE (displaying retrieved chunks only).")
        print("!"*60 + "\n")

    print("Initializing HuggingFaceEmbedding (BAAI/bge-large-en-v1.5)...")
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5"
    )

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("educational_mentor_knowledgebase")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Loading index...")
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )

    if ollama_running:
        print("Initializing Ollama Mistral LLM...")
        llm = Ollama(model="mistral", system_prompt=SYSTEM_PROMPT, request_timeout=120.0)
        query_engine = index.as_query_engine(llm=llm, similarity_top_k=2)
        print("\nEduBot Chatbot is ready! Ask your questions (type 'exit' to quit).")
    else:
        # Fallback to retriever
        retriever = index.as_retriever(similarity_top_k=2)
        print("\nEduBot Retriever (Fallback Mode) is ready! Ask your questions (type 'exit' to quit).")

    while True:
        try:
            query = input("\nYou: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                break

            if ollama_running:
                response = query_engine.query(query)
                from formatter import clean_response
                cleaned_llm_response = clean_response(str(response))
                print(f"EduBot: {cleaned_llm_response}")
            else:
                from preprocessor import preprocess_query
                expanded_query = preprocess_query(query)
                from intent_router import classify_intent
                intent, score = classify_intent(expanded_query)
                
                if intent == "OUT_OF_SCOPE":
                    print("\nEduBot: I am designed specifically for educational and LMS-related assistance on the Edutainer platform. Please let me know how I can support you in these academic domains!")
                    continue
                    
                results = retriever.retrieve(expanded_query)
                from formatter import conversational_fallback
                fallback_msg = conversational_fallback(query, results)
                print(f"\nEduBot: {fallback_msg}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
