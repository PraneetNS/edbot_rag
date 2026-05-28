import sys
import json
import glob
from pathlib import Path

# Add parent path to import config
sys.path.append(str(Path(__file__).resolve().parent))
import config
from setup_dirs import initialize_directories
from dataset_generator import generate_synthetic_data
from scraper import EducationalScraper
from semantic_chunker import SemanticEducationalChunker
from metadata_tagger import MetadataTagger
from embedding_engine import EmbeddingEngine
from vector_store import EducationalVectorStore
from query_router import EducationalQueryRouter

def main():
    print("\n" + "="*60)
    # 1. Setup targeted directory folders
    initialize_directories()
    
    # 2. Generate structured synthetic JSON datasets
    generate_synthetic_data()
    
    # 3. Targeted Web Scraping
    scraper = EducationalScraper()
    print("\nScraping targeted educational sources...")
    
    # Let's scrape roadmap.sh AI guide and MDN JavaScript introduction guide
    # Fallback to high-quality offline documents if internet fails or times out
    targets = [
        ("roadmaps", "ai_roadmap", "https://roadmap.sh/ai"),
        ("courses", "javascript_guide", "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Introduction"),
        ("engineering_subjects", "dbms_basics", "https://www.geeksforgeeks.org/dbms/")
    ]
    
    for category, filename, url in targets:
        text, title = scraper.scrape_url(url)
        if text:
            cleaned = scraper.clean_text(text)
            scraper.save_educational_document(category, filename, cleaned, url)
        else:
            # High-fidelity fallback to bootstrap educational corpus
            print(f"URL {url} could not be resolved. Bootstrapping offline document...")
            fallback_corpus = ""
            if category == "roadmaps":
                fallback_corpus = (
                    "# AI & Machine Learning Career Roadmap\n\n"
                    "Step 1: Mathematics and Fundamentals. Master linear algebra, calculus, and probability.\n\n"
                    "Step 2: Programming. Learn Python or R and data handling libraries like Pandas and NumPy.\n\n"
                    "Step 3: Core Machine Learning. Explore regression, decision trees, random forests, and SVMs.\n\n"
                    "Step 4: Deep Learning and Neural Networks. Master PyTorch or TensorFlow, CNNs, and RNNs."
                )
            elif category == "courses":
                fallback_corpus = (
                    "# JavaScript Language Guide\n\n"
                    "JavaScript is a lightweight, interpreted, object-oriented programming language designed for web development.\n\n"
                    "Key concepts include variables (let, const), functions, loops, arrays, objects, and asynchronous promises."
                )
            else:
                fallback_corpus = (
                    "# Database Management Systems (DBMS) Fundamentals\n\n"
                    "A Database Management System (DBMS) is software designed to store, retrieve, and manage structured databases.\n\n"
                    "Key academic topics: Relational algebra, SQL query optimization, transaction ACID properties, and database indexing techniques."
                )
            scraper.save_educational_document(category, filename, fallback_corpus, url)

    # 4. Semantic Chunking, Tagging, and Embedding
    chunker = SemanticEducationalChunker(chunk_size=400, chunk_overlap=50)
    tagger = MetadataTagger()
    embedding_engine = EmbeddingEngine()
    vector_store = EducationalVectorStore()
    
    # Rebuild fresh database collections
    vector_store.reset_collections()
    
    print("\nProcessing, Chunking, Tagging, and Ingesting documents...")
    
    # A. Ingest scraped/cleaned text documents
    scraped_files = glob.glob(str(config.DATA_DIR / "**" / "*.txt"), recursive=True)
    for filepath in scraped_files:
        path = Path(filepath)
        category = path.parent.name
        source_name = path.name
        
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                continue
            source_url = lines[0].replace("Source URL: ", "").strip()
            content = "".join(lines[1:])
            
        print(f"Parsing semantic chunks for file: {source_name}...")
        chunks = chunker.split_text(content)
        
        for c in chunks:
            # Enforce 11-key schema
            tagged = tagger.tag_chunk(c, source_url, category)
            # Embed
            emb = embedding_engine.get_text_embedding(c)
            # Route to collection
            intent = tagged["intent"]
            collection_name = config.COLLECTIONS.get(intent, "roadmap_collection")
            vector_store.add_educational_chunk(collection_name, tagged, emb)

    # B. Ingest synthetic dialogs JSON
    dialogues_path = config.DATA_DIR / "synthetic_dialogues" / "conversations.json"
    if dialogues_path.exists():
        with open(dialogues_path, "r", encoding="utf-8") as f:
            conversations = json.load(f)
            
        print("Ingesting synthetic dialogues JSON...")
        for conv in conversations:
            formatted_text = f"Student Query: {conv['student_query']}\nMentor Response: {conv['mentor_response']}"
            tagged = tagger.tag_chunk(formatted_text, "conversations.json", "synthetic_dialogues")
            
            # Map custom properties
            tagged["difficulty"] = conv["difficulty"]
            tagged["domain"] = conv["domain"]
            tagged["intent"] = conv["intent"]
            tagged["topic"] = conv["topic"]
            tagged["tags"] = conv["tags"]
            
            emb = embedding_engine.get_text_embedding(formatted_text)
            collection_name = config.COLLECTIONS.get(conv["intent"], "mentoring_collection")
            vector_store.add_educational_chunk(collection_name, tagged, emb)

    # C. Ingest LMS workflows JSON
    workflows_path = config.DATA_DIR / "workflows" / "lms_workflows.json"
    if workflows_path.exists():
        with open(workflows_path, "r", encoding="utf-8") as f:
            workflows = json.load(f)
            
        print("Ingesting LMS workflows JSON...")
        for w in workflows:
            formatted_text = f"Intent: {w['intent']}\nExamples:\n" + "\n".join(f"- {ex}" for ex in w["examples"]) + "\nWorkflow:\n" + "\n".join(f"{idx+1}. {step}" for idx, step in enumerate(w["workflow"]))
            tagged = tagger.tag_chunk(formatted_text, "lms_workflows.json", "workflows")
            
            # Map custom properties
            tagged["difficulty"] = w["difficulty"]
            tagged["domain"] = w["domain"]
            tagged["topic"] = w["topic"]
            tagged["tags"] = w["tags"]
            
            emb = embedding_engine.get_text_embedding(formatted_text)
            vector_store.add_educational_chunk("workflow_collection", tagged, emb)

    # 5. Ingestion Verification & Collection Stats
    vector_store.print_statistics()
    print("Ingestion Pipeline indexing successfully completed!")
    
    # 6. Verification Queries using Reranked Query Routing Engine
    print("\nRunning RAG pipeline query verification routing...")
    query_router = EducationalQueryRouter()
    
    verification_queries = [
        "I am weak in DSA and need placement support.",
        "VTU certification missing",
        "What concepts are required in AI roadmap?"
    ]
    
    for q in verification_queries:
        res = query_router.route_and_retrieve(q, top_k=2)
        print(f"Results retrieved: {len(res['retrieved_context'])} (Retrieval Confidence: {res['retrieval_confidence']:.4f})")
        for idx, item in enumerate(res['retrieved_context']):
            print(f"  [{idx+1}] ID: {item['id']} | Score: {item['score']:.4f} | Category: {item['metadata'].get('category')}")
            print(f"      Content summary: {item['content'].strip().replace('\n', ' ')[:100]}...")

if __name__ == "__main__":
    main()
