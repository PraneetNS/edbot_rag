import sys
from pathlib import Path
import chromadb

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.append(str(BACKEND_DIR))

from rag.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, CHROMA_COLLECTION_5K

print("Chroma persist dir:", CHROMA_PERSIST_DIR)

client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
print("Collections:", [c.name for c in client.list_collections()])

for col_name in [CHROMA_COLLECTION_NAME, CHROMA_COLLECTION_5K]:
    try:
        col = client.get_collection(col_name)
        count = col.count()
        print(f"\nCollection '{col_name}' has {count} documents.")
        
        # Get a sample of metadatas
        if count > 0:
            res = col.get(limit=count, include=["metadatas"])
            metadatas = res.get("metadatas", [])
            sources = set()
            topics = set()
            for meta in metadatas:
                if meta:
                    if "source" in meta:
                        sources.add(meta["source"])
                    if "topic" in meta:
                        topics.add(meta["topic"])
            print("  Sources:", list(sources))
            print("  Topics:", list(topics))
    except Exception as e:
        print(f"Error reading collection '{col_name}': {e}")

# Also check edumentor_mentor collection in edumentor_chroma
chroma_path_mentor = BACKEND_DIR.parent / "edumentor_chroma"
print("\nChroma mentor path:", chroma_path_mentor)
if chroma_path_mentor.exists():
    try:
        client_mentor = chromadb.PersistentClient(path=str(chroma_path_mentor))
        print("Mentor Collections:", [c.name for c in client_mentor.list_collections()])
        for c in client_mentor.list_collections():
            col = client_mentor.get_collection(c.name)
            count = col.count()
            print(f"  Collection '{c.name}' has {count} documents.")
            if count > 0:
                res = col.get(limit=count, include=["metadatas"])
                metadatas = res.get("metadatas", [])
                sources = set()
                for meta in metadatas:
                    if meta:
                        if "source" in meta:
                            sources.add(meta["source"])
                        elif "question" in meta:
                            # print some keys
                            pass
                print("    Sources found:", list(sources))
    except Exception as e:
        print(f"Error checking mentor db: {e}")
