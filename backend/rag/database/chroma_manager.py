import shutil
import logging
from pathlib import Path
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)

class ChromaManager:
    """
    Manages database connection and persistence directories for ChromaDB.
    Supports rebuilding databases from scratch.
    """
    def __init__(self, persist_dir: Path, collection_name: str):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    def initialize_db(self, rebuild_fresh: bool = False):
        """
        Initializes the Chroma persistent client.
        If rebuild_fresh is True, clears the folder completely first.
        """
        if rebuild_fresh:
            logger.info(f"Clearing old Chroma DB persist folder at {self.persist_dir} to rebuild fresh.")
            try:
                # We do a robust directory tree cleanup
                if self.persist_dir.exists():
                    shutil.rmtree(self.persist_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to delete directory {self.persist_dir}: {e}")

        # Ensure directory structure exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        if rebuild_fresh:
            try:
                collections = [c.name for c in self.client.list_collections()]
                if self.collection_name in collections:
                    self.client.delete_collection(self.collection_name)
                    logger.info(f"Collection '{self.collection_name}' deleted successfully.")
            except Exception as e:
                logger.warning(f"Error checking/deleting collection {self.collection_name}: {e}")

        # Recreate collection using cosine distance metric
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialized ChromaDB collection '{self.collection_name}' with Cosine metric.")

    def get_vector_store(self) -> ChromaVectorStore:
        """
        Returns a LlamaIndex-compatible ChromaVectorStore instance.
        """
        if self.collection is None:
            self.initialize_db()
        return ChromaVectorStore(chroma_collection=self.collection)

    def get_collection(self):
        """
        Returns the raw ChromaDB collection object.
        Useful for administrative operations like count(), get(), etc.
        """
        if self.collection is None:
            self.initialize_db()
        return self.collection

