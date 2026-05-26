import sys
from pathlib import Path

# Add directories to system path
sys.path.append(str(Path(__file__).parent))

import scraper.scrape
import rag.ingest
import rag.query
import rag.chatbot

def show_menu():
    print("\n" + "="*40)
    print("      EDUTAINER RAG PIPELINE CONTROL")
    print("="*40)
    print("1. Scrape Website Pages (Homepage, About, Courses, Contact)")
    print("2. Ingest Documents into ChromaDB")
    print("3. Query the Knowledge Base (Semantic Search)")
    print("4. Run Full Pipeline (Scrape -> Ingest -> Query)")
    print("5. Chat with EduBot (Interactive Chatbot Interface)")
    print("6. Exit")
    print("="*40)

def main():
    while True:
        show_menu()
        try:
            choice = input("Enter your choice (1-6): ").strip()
            if choice == "1":
                mode = input("Select scrape mode [auto (default), selenium, requests]: ").strip() or "auto"
                print(f"Starting scraper in '{mode}' mode...")
                orig_argv = sys.argv
                sys.argv = [sys.argv[0], "--mode", mode]
                try:
                    scraper.scrape.main()
                finally:
                    sys.argv = orig_argv
            elif choice == "2":
                print("Starting document ingestion...")
                rag.ingest.main()
            elif choice == "3":
                q = input("Enter search query: ").strip()
                if q:
                    rag.query.query_kb(q)
            elif choice == "4":
                print("Running full pipeline...")
                # 1. Scrape
                orig_argv = sys.argv
                sys.argv = [sys.argv[0], "--mode", "auto"]
                try:
                    scraper.scrape.main()
                finally:
                    sys.argv = orig_argv
                # 2. Ingest
                rag.ingest.main()
                # 3. Test query
                print("\nRunning verification test query...")
                rag.query.query_kb("What does Edutainer provide?")
            elif choice == "5":
                print("Launching EduBot Interactive Chatbot Interface...")
                try:
                    rag.chatbot.main()
                except Exception as e:
                    print(f"Error launching chatbot: {e}")
            elif choice == "6":
                print("Exiting. Goodbye!")
                break
            else:
                print("Invalid choice, please select 1-6.")
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
