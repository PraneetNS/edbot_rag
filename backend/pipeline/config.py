import os
from pathlib import Path

# Base Paths
PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent

# Data Folders
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed"
CHUNKS_DIR = PROJECT_ROOT / "chunks"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
VECTORDB_DIR = PROJECT_ROOT / "vectordb"
LOGS_DIR = PROJECT_ROOT / "logs"

# Subfolders under data/
DATA_SUBDIRS = [
    "placements",
    "internships",
    "roadmaps",
    "support",
    "courses",
    "engineering_subjects",
    "mentoring",
    "workflows",
    "certifications",
    "synthetic_dialogues"
]

# Logging configuration
LOG_FILE = LOGS_DIR / "pipeline.log"

# Model Configurations
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ChromaDB collections
COLLECTIONS = {
    "PLACEMENT_GUIDANCE": "placements_collection",
    "INTERNSHIP_GUIDANCE": "internship_collection",
    "LMS_SUPPORT": "support_collection",
    "CERTIFICATION_SUPPORT": "support_collection",
    "COURSE_QUERY": "roadmap_collection",
    "EXAM_ASSISTANCE": "engineering_collection",
    "ASSIGNMENT_HELP": "engineering_collection",
    "MENTORING": "mentoring_collection",
    "WORKFLOW": "workflow_collection"
}

# Targeted Educational Sources to Scrape (Subset for indexing demo)
TARGET_SOURCES = {
    "roadmaps": [
        "https://roadmap.sh/ai",
        "https://roadmap.sh/backend",
        "https://roadmap.sh/frontend"
    ],
    "programming": [
        "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Introduction"
    ],
    "engineering": [
        "https://www.geeksforgeeks.org/dbms/"
    ]
}
