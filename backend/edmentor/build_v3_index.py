import os
import json
import uuid
import sys
from pathlib import Path

# Setup paths to ensure imports from backend work
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BACKEND_DIR))

# Force offline mode for Hugging Face
os.environ["HF_HUB_OFFLINE"] = "1"

from rag.config import CHROMA_PERSIST_DIR
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def build_v3_index():
    print("Initializing build_v3_index.py...")
    
    # Paths to source data files
    kb_path = BACKEND_DIR.parent / "edumentor_knowledge_base.json"
    rag_path = BACKEND_DIR.parent / "rag_docs.json"
    
    print(f"Knowledge Base Path: {kb_path}")
    print(f"RAG Docs Path: {rag_path}")
    print(f"Persist Directory: {CHROMA_PERSIST_DIR}")
    
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found at {kb_path}")
    if not rag_path.exists():
        raise FileNotFoundError(f"RAG docs file not found at {rag_path}")
        
    # 1. Load Knowledge Base and parse
    print("Loading and parsing edumentor_knowledge_base.json...")
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_data = json.load(f)
        
    kb_documents = []
    
    # Parse dsa_concepts
    for item in kb_data.get("dsa_concepts", []):
        concept_name = item["topic"]
        content_fields_joined = (
            f"Category: {item['category']}. Difficulty: {item['difficulty']}. "
            f"When to use: {item['when_to_use']}. Core idea: {item['core_idea']}. "
            f"Common mistakes: {item['common_mistakes']}. Interview tip: {item['interview_tip']}"
        )
        text = f"DSA Concept: {concept_name}. {content_fields_joined}"
        kb_documents.append(Document(
            page_content=text,
            metadata={"id": item["id"], "section": "mindset_support" if item['category'] == "mindset" else "dsa_concepts"}
        ))
        
    # Parse company_patterns
    for item in kb_data.get("company_patterns", []):
        company = item["company"]
        content_fields_joined = (
            f"Tier: {item['tier']}. Rounds: {item['rounds']}. Focus Areas: {item['focus_areas']}. "
            f"OA Pattern: {item['oa_pattern']}. Red Flags: {item['red_flags']}. Prep Advice: {item['prep_advice']}"
        )
        text = f"Company Profile: {company}. {content_fields_joined}"
        kb_documents.append(Document(
            page_content=text,
            metadata={"id": item["id"], "section": "company_patterns"}
        ))
        
    # Parse placement_timelines
    for item in kb_data.get("placement_timelines", []):
        phase = item["phase"]
        content_fields_joined = (
            f"Student Profile: {item['student_profile']}. Goal: {item['goal']}. "
            f"What to do: {item['what_to_do']}. What to avoid: {item['what_to_avoid']}. Checkpoint: {item['checkpoint']}"
        )
        text = f"Placement Timeline: {phase}. {content_fields_joined}"
        kb_documents.append(Document(
            page_content=text,
            metadata={"id": item["id"], "section": "placement_timelines"}
        ))
        
    # Parse resume_guidance
    for item in kb_data.get("resume_guidance", []):
        subtopic = item["subtopic"]
        content_fields_joined = (
            f"Context: {item['context']}. Guidance: {item['guidance']}. Example: {item['example']}"
        )
        text = f"Resume Guidance: {subtopic}. {content_fields_joined}"
        kb_documents.append(Document(
            page_content=text,
            metadata={"id": item["id"], "section": "resume_guidance"}
        ))
        
    # Parse internship_strategy
    for item in kb_data.get("internship_strategy", []):
        subtopic = item["subtopic"]
        content_fields_joined = (
            f"Applicable Year: {item['year_applicable']}. Guidance: {item['guidance']}. "
            f"Common mistake: {item['common_mistake']}. Success signal: {item['success_signal']}"
        )
        text = f"Internship Strategy: {subtopic}. {content_fields_joined}"
        kb_documents.append(Document(
            page_content=text,
            metadata={"id": item["id"], "section": "internship_strategy"}
        ))
        
    # Parse career_roadmaps
    for item in kb_data.get("career_roadmaps", []):
        career_path = item["career_path"]
        content_fields_joined = (
            f"Time Horizon: {item['time_horizon']}. Starting Point: {item['starting_point']}. "
            f"Phase 1: {item['phase_1']}. Phase 2: {item['phase_2']}. Phase 3: {item['phase_3']}. "
            f"Projects to build: {item['projects_to_build']}. Skills that matter: {item['skills_that_matter']}"
        )
        text = f"Career Roadmap: {career_path}. {content_fields_joined}"
        kb_documents.append(Document(
            page_content=text,
            metadata={"id": item["id"], "section": "career_roadmaps"}
        ))
        
    # Parse mindset_support
    for item in kb_data.get("mindset_support", []):
        situation = item["situation"]
        content_fields_joined = (
            f"Validation: {item['validation']}. Reframe: {item['reframe']}. Action: {item['action']}"
        )
        text = f"Mindset Support: {situation}. {content_fields_joined}"
        kb_documents.append(Document(
            page_content=text,
            metadata={"id": item["id"], "section": "mindset_support"}
        ))
        
    # Parse higher_studies
    for item in kb_data.get("higher_studies", []):
        subtopic = item["subtopic"]
        content_fields_joined = (
            f"Guidance: {item['guidance']}. Common myths: {item['common_myths']}. Decision factor: {item['decision_factor']}"
        )
        text = f"Higher Studies: {subtopic}. {content_fields_joined}"
        kb_documents.append(Document(
            page_content=text,
            metadata={"id": item["id"], "section": "higher_studies"}
        ))
        
    print(f"Total structured knowledge base documents parsed: {len(kb_documents)}")
    
    # 2. Load and filter Q&A pairs from rag_docs.json
    print("Loading and filtering rag_docs.json...")
    with open(rag_path, "r", encoding="utf-8") as f:
        rag_data = json.load(f)
        
    bad_signals = ["write a python", "write code", "dll", "go build", "tailwind css", "modify the skill map"]
    
    rag_documents = []
    for item in rag_data:
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        # Word count calculation
        word_count = len(answer.split())
        
        # Filters:
        # - Word count limit: < 15 or > 130
        if word_count < 15 or word_count > 130:
            continue
            
        # - Off-domain phrases in question
        if any(sig in question.lower() for sig in bad_signals):
            continue
            
        # Format
        text = f"Student: {question}\nMentor: {answer}"
        
        # Generate stable UUID or random UUID for doc
        doc_id = str(uuid.uuid4())
        rag_documents.append(Document(
            page_content=text,
            metadata={"id": doc_id, "section": "rag_docs"}
        ))
        
    print(f"Total RAG docs after filtering: {len(rag_documents)}")
    
    all_documents = kb_documents + rag_documents
    print(f"Total combined documents to index: {len(all_documents)}")
    
    # 3. Create HuggingFaceEmbeddings model
    print("Initializing HuggingFaceEmbeddings using 'all-MiniLM-L6-v2'...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Initialize LangChain Chroma wrapper and build index
    # We clear the collection if it exists by deleting it first.
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    
    collection_name = "edumentor_v3"
    print(f"Clearing collection '{collection_name}' if it already exists...")
    try:
        client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted.")
    except Exception:
        print(f"Collection '{collection_name}' did not exist or could not be deleted.")
        
    print(f"Creating new LangChain Chroma collection '{collection_name}'...")
    
    # We will index in batches to avoid overhead and track progress
    batch_size = 256
    db = None
    
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i+batch_size]
        if db is None:
            db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=str(CHROMA_PERSIST_DIR)
            )
        else:
            db.add_documents(batch)
            
        print(f"Indexed {min(i + batch_size, len(all_documents))}/{len(all_documents)} documents.")
        
    print(f"Index built successfully! Total documents in '{collection_name}': {len(all_documents)}")

if __name__ == "__main__":
    build_v3_index()
