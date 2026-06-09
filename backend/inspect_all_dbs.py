import sys
from pathlib import Path
import chromadb

WORKSPACE_DIR = Path(__file__).resolve().parent.parent

paths_to_check = [
    WORKSPACE_DIR / "chroma_db",
    WORKSPACE_DIR / "edumentor_chroma",
    WORKSPACE_DIR / "backend" / "chroma_store",
    WORKSPACE_DIR / "backend" / "vectordb",
]

for p in paths_to_check:
    if p.exists():
        print("\nChecking path:", p)
        try:
            client = chromadb.PersistentClient(path=str(p))
            collections = client.list_collections()
            print("  Collections:", [c.name for c in collections])
            for c in collections:
                col = client.get_collection(c.name)
                count = col.count()
                print(f"    Collection '{c.name}' has {count} documents.")
                if count > 0:
                    res = col.get(limit=min(count, 1000), include=["documents", "metadatas"])
                    docs = res.get("documents", [])
                    metadatas = res.get("metadatas", [])
                    sources = set()
                    edutainer_count = 0
                    for d, m in zip(docs, metadatas):
                        if m and "source" in m:
                            sources.add(m["source"])
                        if "Edutainer" in d or (m and "Edutainer" in str(m)):
                            edutainer_count += 1
                    print("      Sources:", list(sources))
                    print("      Edutainer count:", edutainer_count)
        except Exception as e:
            print("  Error:", e)
