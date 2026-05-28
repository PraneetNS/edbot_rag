import sys
from pathlib import Path

# Add pipeline root to system path to load config
sys.path.append(str(Path(__file__).resolve().parent))
import config

def initialize_directories():
    """
    Creates the entire target directories structure for Phase 3 ingestion.
    """
    print("Initializing RAG pipeline folder structures...")
    
    # 1. Base directories
    base_dirs = [
        config.DATA_DIR,
        config.PROCESSED_DIR,
        config.CHUNKS_DIR,
        config.EMBEDDINGS_DIR,
        config.VECTORDB_DIR,
        config.LOGS_DIR
    ]
    
    for d in base_dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created base directory: {d}")
        
    # 2. Data Subdirectories
    for sub in config.DATA_SUBDIRS:
        subdir_path = config.DATA_DIR / sub
        subdir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created data subcategory directory: {subdir_path}")
        
    print("RAG Directory Ingestion infrastructure created successfully.")

if __name__ == "__main__":
    initialize_directories()
