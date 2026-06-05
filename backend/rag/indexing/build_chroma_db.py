import os
import json
import uuid
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# Resolve paths
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
WORKSPACE_DIR = BACKEND_DIR.parent
INPUT_PATH = WORKSPACE_DIR / "rag_docs.json"
CHROMA_PATH = WORKSPACE_DIR / "edumentor_chroma"

# CRITICAL CONSTRAINTS (Requirements Three):
# 1. During indexing, we embed the combined text format: f"Student: {question}\nMentor: {answer}"
# 2. At retrieval time, the query embedding prefix MUST be exactly "Student: {query}" to match the indexed context format.
# This comment confirms the query prefix design constraint.

print(f"Loading cleaned docs from: {INPUT_PATH}")
if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Cleaned dataset not found at {INPUT_PATH}. Run clean_dataset.py first.")

docs = json.load(open(INPUT_PATH, "r", encoding="utf-8"))
print(f"Total documents to index: {len(docs)}")

# Load embedding model (downloads to cache if not present)
print("Loading sentence-transformers/all-MiniLM-L6-v2 model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print(f"Initializing persistent ChromaDB client at: {CHROMA_PATH}")
client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# Create or clear collection
print("Creating collection 'edumentor_mentor' with cosine distance...")
collection = client.get_or_create_collection(
    name="edumentor_mentor",
    metadata={"hnsw:space": "cosine"}
)

# Batch indexing
BATCH = 256
print(f"Starting indexing in batches of {BATCH}...")
for i in range(0, len(docs), BATCH):
    batch = docs[i:i+BATCH]
    texts = [d["text"] for d in batch]
    embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()
    
    collection.add(
        documents=[d["answer"] for d in batch],  # Store only the mentor's answer for retrieval
        embeddings=embeddings,
        metadatas=[{"question": d["question"]} for d in batch],
        ids=[str(uuid.uuid4()) for _ in batch]
    )
    if i % 1000 == 0 or (i + BATCH) >= len(docs):
        print(f"Indexed {min(i + len(batch), len(docs))}/{len(docs)} documents.")

print(f"ChromaDB indexing complete. Total vectors in collection: {collection.count()}")
