import sys
from pathlib import Path
import chromadb

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.append(str(BACKEND_DIR))

from rag.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_5K

# Check edumentor_mentor collection
chroma_path_mentor = BACKEND_DIR.parent / "edumentor_chroma"
print("Checking edumentor_mentor:")
if chroma_path_mentor.exists():
    client_mentor = chromadb.PersistentClient(path=str(chroma_path_mentor))
    col = client_mentor.get_collection("edumentor_mentor")
    count = col.count()
    res = col.get(limit=count, include=["documents", "metadatas"])
    docs = res.get("documents", [])
    metadatas = res.get("metadatas", [])
    
    found_count = 0
    for doc, meta in zip(docs, metadatas):
        if "Edutainer" in doc or "Edutainer" in str(meta):
            found_count += 1
            print(f"  Found in edumentor_mentor: {doc[:100]}... | Meta: {meta}")
            if found_count > 5:
                print("  (more than 5 matches, stopping mentor print)")
                break
    print(f"Total found in edumentor_mentor: {found_count}")

# Check edumentor_5k collection
print("\nChecking edumentor_5k:")
client_5k = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
col_5k = client_5k.get_collection(CHROMA_COLLECTION_5K)
count_5k = col_5k.count()
res_5k = col_5k.get(limit=count_5k, include=["documents", "metadatas"])
docs_5k = res_5k.get("documents", [])
metadatas_5k = res_5k.get("metadatas", [])

found_count_5k = 0
for doc, meta in zip(docs_5k, metadatas_5k):
    if "Edutainer" in doc or "Edutainer" in str(meta):
        found_count_5k += 1
        print(f"  Found in edumentor_5k: {doc[:100]}... | Meta: {meta}")
        if found_count_5k > 5:
            print("  (more than 5 matches, stopping 5k print)")
            break
print(f"Total found in edumentor_5k: {found_count_5k}")
