import sys
import json
from pathlib import Path
import chromadb

# Add parent path to import config
sys.path.append(str(Path(__file__).resolve().parent))
import config

class EducationalVectorStore:
    """
    Manages 7 specialized, domain-isolated collections in ChromaDB
    to ensure precise context isolation and metadata routing.
    """
    def __init__(self, db_dir: Path = config.VECTORDB_DIR):
        print(f"Connecting to ChromaDB PersistentClient at {db_dir}...")
        self.client = chromadb.PersistentClient(path=str(db_dir))
        self.collections = {}
        
        # Provision the 7 collections
        self._initialize_collections()

    def _initialize_collections(self):
        """
        Creates and stores collection references for the 7 designated sub-domains.
        """
        target_collections = [
            "placements_collection",
            "support_collection",
            "engineering_collection",
            "roadmap_collection",
            "internship_collection",
            "mentoring_collection",
            "workflow_collection"
        ]
        
        for c in target_collections:
            self.collections[c] = self.client.get_or_create_collection(name=c)
            print(f"ChromaDB collection initialized: {c} (count: {self.collections[c].count()})")

    def reset_collections(self):
        """
        Deletes and rebuilds fresh collections for clean rebuilds.
        """
        print("Resetting all ChromaDB collections for clean rebuild...")
        for name in list(self.collections.keys()):
            try:
                self.client.delete_collection(name)
                print(f"Deleted collection: {name}")
            except Exception as e:
                print(f"Warning: Could not delete collection {name}: {e}")
        self._initialize_collections()

    def add_educational_chunk(self, collection_name: str, chunk_schema: dict, embedding: list[float]):
        """
        Ingests a standardized chunk conforming to the 11-key metadata schema.
        Maps list fields like tags to CSV strings for ChromaDB compatibility.
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} is not one of the 7 designated collections.")
            
        collection = self.collections[collection_name]
        
        # Format list variables into simple primitives for ChromaDB metadata compatibility
        metadata_clean = {}
        for k, v in chunk_schema.items():
            if k in ["content", "id"]:
                continue
            if isinstance(v, list):
                metadata_clean[k] = ",".join(v)
            else:
                metadata_clean[k] = v if v is not None else ""

        collection.add(
            ids=[chunk_schema["id"]],
            embeddings=[embedding],
            metadatas=[metadata_clean],
            documents=[chunk_schema["content"]]
        )

    def print_statistics(self):
        """
        Outputs document count of all collections.
        """
        print("\n=== VECTORSTORE COLLECTION STATISTICS ===")
        for name, col in self.collections.items():
            print(f"Collection: {name:<25} | Documents Indexed: {col.count()}")
        print("=========================================\n")

if __name__ == "__main__":
    store = EducationalVectorStore()
    store.print_statistics()
