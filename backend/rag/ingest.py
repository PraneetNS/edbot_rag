import sys
import json
import re
import shutil
from pathlib import Path
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    StorageContext
)
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Robust path setup relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_store"
DATA_DIR = BASE_DIR / "data"
CHUNKS_FILE = BASE_DIR.parent / "chunks.json"

# Import custom BM25 retriever
sys.path.append(str(BASE_DIR))
from rag.bm25 import BM25Retriever

def normalize_whitespace(text: str) -> str:
    """
    Cleans text by converting all whitespace characters to single spaces
    and validating UTF-8 compliance.
    """
    if not text:
        return ""
    # Replace carriage returns, tabs, multiple spaces, etc.
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned

def infer_metadata(chunk: dict) -> dict:
    """
    Intelligently infers academic metadata fields from raw content and category,
    ensuring all list values are converted to comma-separated strings for ChromaDB compatibility.
    """
    content = chunk.get("content", "")
    content_lower = content.lower()
    category = chunk.get("category", "general").lower()
    
    # 1. Infer Topic
    topic = "General Academics"
    topic_keywords = {
        "react": "React JS",
        "python": "Python",
        "java": "Java",
        "dbms": "DBMS",
        "database": "DBMS",
        "sql": "DBMS",
        "dsa": "DSA",
        "data structures": "DSA",
        "algorithms": "DSA",
        "devops": "DevOps",
        "docker": "Docker",
        "kubernetes": "Kubernetes",
        "ansible": "DevOps",
        "terraform": "DevOps",
        "jenkins": "DevOps",
        "data science": "Data Science",
        "data scientist": "Data Science",
        "cybersecurity": "Cybersecurity",
        "cyber security": "Cybersecurity",
        "hacking": "Cybersecurity",
        "ielts": "IELTS English",
        "interior design": "Interior Design",
        "system design": "System Design",
        "marketing": "Growth Marketing",
        "trading": "Stock Trading",
        "crypto": "Cryptocurrency",
        "finance": "Financial Planning"
    }
    
    for kw, topic_name in topic_keywords.items():
        if kw in content_lower:
            topic = topic_name
            break

    # 2. Generate Technical Tags
    tags_set = set()
    if category:
        tags_set.add(category)
        
    tag_keywords = ["components", "props", "state", "hooks", "variables", "functions", "loops", 
                    "classes", "inheritance", "tables", "queries", "indexing", "sorting", "searching",
                    "cloud", "ci/cd", "deployment", "pipelines", "statistics", "regression", 
                    "networks", "protocols", "firewalls", "exams", "passing", "vtu", "resumes", 
                    "interviews", "placements", "internships"]
                    
    for tk in tag_keywords:
        if tk in content_lower:
            tags_set.add(tk)
            
    tags_str = ",".join(sorted(list(tags_set)))

    # 3. Infer Educational Type
    educational_type = "technical_concept"
    if "roadmap.sh" in content_lower or "roadmap" in content_lower or "step by step guide" in content_lower:
        educational_type = "roadmap"
    elif any(k in content_lower for k in ["career", "journey", "advice", "motivate", "mentoring", "mentor", "mindset", "continuous learning", "marathon"]):
        educational_type = "mentoring_guidance"
    elif any(k in content_lower for k in ["resume", "mock interview", "placement preparation", "employability", "internship opportunity", "recruiting"]):
        educational_type = "placement_guide"
    elif any(k in content_lower for k in ["failed", "re-exam", "backlog", "passing requirements", "vtu stamp", "grade", "exam marks"]):
        educational_type = "exam_prep"

    # 4. Infer Roadmap Domain
    roadmap_domain = "general"
    if any(k in content_lower for k in ["data science", "data scientist", "machine learning", "ai", "deep learning", "neural"]):
        roadmap_domain = "data_science"
    elif any(k in content_lower for k in ["devops", "docker", "kubernetes", "ansible", "terraform", "jenkins"]):
        roadmap_domain = "devops"
    elif any(k in content_lower for k in ["frontend", "react", "html", "css", "javascript", "typescript"]):
        roadmap_domain = "frontend"
    elif any(k in content_lower for k in ["backend", "nodejs", "django", "database", "sql", "postgresql"]):
        roadmap_domain = "backend"
    elif any(k in content_lower for k in ["full stack", "fullstack"]):
        roadmap_domain = "fullstack"
    elif any(k in content_lower for k in ["cyber", "security", "hacking"]):
        roadmap_domain = "cybersecurity"

    # 5. Extract Semantic Keywords (simple noun-phrase/word extractor)
    keywords_found = []
    semantic_triggers = ["react", "props", "jsx", "python", "interpreter", "list comprehension", "object-oriented", 
                         "inheritance", "databases", "primary key", "foreign key", "joins", "normalization", 
                         "dsa", "complexity", "arrays", "trees", "graphs", "sorting", "binary search", 
                         "devops", "containers", "orchestration", "ci/cd", "pipelines", "machine learning", 
                         "statistics", "regression", "neural networks", "placement", "resume", "mock interview", 
                         "vtu", "certificates", "lms", "portal", "login"]
                         
    for trigger in semantic_triggers:
        if trigger in content_lower:
            keywords_found.append(trigger)
            
    keywords_str = ",".join(keywords_found)

    return {
        "source": chunk.get("source", "https://edutainer.in"),
        "category": chunk.get("category", "general"),
        "intent": chunk.get("intent", "general"),
        "difficulty": chunk.get("difficulty", "beginner"),
        "topic": topic,
        "tags": tags_str,
        "chunk_id": str(chunk.get("id", "")),
        "educational_type": educational_type,
        "roadmap_domain": roadmap_domain,
        "semantic_keywords": keywords_str
    }

def main():
    print(f"Loading raw chunks from {CHUNKS_FILE}...")
    if not CHUNKS_FILE.exists():
        print(f"Error: Chunks file '{CHUNKS_FILE}' is missing.")
        return

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)
    print(f"Loaded {len(raw_chunks)} raw chunks.")

    # 1. Cleaning & Deduplication
    print("Performing whitespace normalization and encoding validation...")
    cleaned_chunks = []
    seen_contents = set()
    duplicates_count = 0

    for chunk in raw_chunks:
        # Validate UTF-8 by re-encoding/decoding if needed
        try:
            raw_content = chunk.get("content", "")
            raw_content.encode("utf-8").decode("utf-8")
        except Exception:
            continue
            
        content_normalized = normalize_whitespace(raw_content)
        if not content_normalized:
            continue
            
        # Deduplication based on content hash/string
        content_hash = content_normalized.lower()
        if content_hash in seen_contents:
            duplicates_count += 1
            continue
            
        seen_contents.add(content_hash)
        
        # Preserve original fields and update content
        new_chunk = dict(chunk)
        new_chunk["content"] = content_normalized
        cleaned_chunks.append(new_chunk)

    print(f"Deduplication summary: Removed {duplicates_count} exact duplicates. Remaining chunks: {len(cleaned_chunks)}")

    # 2. Smart Metadata Enrichment & Preparation of Nodes
    print("Enriching metadata and creating LlamaIndex nodes...")
    nodes = []
    for idx, chunk in enumerate(cleaned_chunks):
        metadata = infer_metadata(chunk)
        
        # Ensure ChromaDB primary IDs and chunk_ids are completely unique
        unique_id = f"chunk_{idx}"
        metadata["chunk_id"] = unique_id
        metadata["original_id"] = str(chunk.get("id", ""))
        
        node = TextNode(
            id_=unique_id,
            text=chunk["content"],
            metadata=metadata
        )
        nodes.append(node)

    # 3. Setup Persistent ChromaDB Vector Store
    COLLECTION_NAME = "educational_mentor_knowledgebase"
    print(f"Initializing ChromaDB PersistentClient at {CHROMA_DIR}...")
    
    # Safely clear old collection files if it exists, to rebuild fresh
    if CHROMA_DIR.exists():
        print("Recreating Chroma DB store to ensure clean versioning...")
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            if COLLECTION_NAME in [c.name for c in client.list_collections()]:
                client.delete_collection(COLLECTION_NAME)
                print(f"Deleted old collection '{COLLECTION_NAME}' successfully.")
        except Exception as e:
            print(f"Chroma clean warning: {e}. Clearing folder...")
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    CHROMA_DIR.mkdir(exist_ok=True, parents=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    print(f"Creating fresh collection '{COLLECTION_NAME}'...")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Optimal space for semantic matching
    )

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load high-capacity BAAI embedding model
    print("Loading HuggingFaceEmbedding (BAAI/bge-large-en-v1.5) for 1024-dim vectors...")
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5"
    )

    # Build Vector Store Index
    print("Generating dense semantic embeddings and building vector database...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    print(f"ChromaDB Vector store populated successfully with {collection.count()} chunks!")

    # 4. Construct & Persist Custom BM25 Index
    print("Compiling high-performance BM25 sparse index...")
    bm25_retriever = BM25Retriever()
    bm25_docs = [{"id": n.node_id, "content": n.text} for n in nodes]
    bm25_retriever.fit(bm25_docs)

    bm25_index_path = DATA_DIR / "bm25_index.json"
    print(f"Saving serialized BM25 index state to {bm25_index_path}...")
    bm25_retriever.save(str(bm25_index_path))

    print("=== INGESTION SUCCESSFUL AND COMPLETE ===")

if __name__ == "__main__":
    main()
