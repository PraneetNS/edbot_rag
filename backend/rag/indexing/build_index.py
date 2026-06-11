import os
import sys
import logging
import json
from pathlib import Path

# Ensure backend directory is in the sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from rag.config import (
    WORKSPACE_DIR,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME
)
from rag.database.chroma_manager import ChromaManager

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _cat(*parts) -> str:
    """Concatenate non-empty string parts with a single space."""
    return " ".join(p for p in parts if p)


def _meta(section: str, entry_id: str, topic: str, category: str, chunk_part: str) -> dict:
    return {
        "section":    section,
        "id":         entry_id,
        "topic":      topic,
        "category":   category,
        "chunk_part": chunk_part,
    }


# ── Per-section chunkers ──────────────────────────────────────────────────────

def chunk_dsa_concepts(entries: list) -> list:
    nodes = []
    for e in entries:
        prefix = f"Topic: {e.get('topic', '')}. Category: {e.get('category', '')}. "
        content = _cat(
            e.get("when_to_use", ""),
            e.get("core_idea", ""),
            e.get("common_mistakes", ""),
            e.get("interview_tip", ""),
        )
        nodes.append(TextNode(
            text=prefix + content,
            metadata=_meta(
                section="dsa_concepts",
                entry_id=e["id"],
                topic=e.get("topic", ""),
                category=e.get("category", e.get("difficulty", "")),
                chunk_part="full",
            )
        ))
    return nodes


def chunk_company_patterns(entries: list) -> list:
    nodes = []
    for e in entries:
        prefix = f"Company: {e.get('company', '')}. Tier: {e.get('tier', '')}. "
        content = _cat(
            e.get("rounds", ""),
            e.get("focus_areas", ""),
            e.get("oa_pattern", ""),
            e.get("red_flags", ""),
            e.get("prep_advice", ""),
        )
        nodes.append(TextNode(
            text=prefix + content,
            metadata=_meta(
                section="company_patterns",
                entry_id=e["id"],
                topic=e.get("company", ""),
                category=e.get("tier", ""),
                chunk_part="full",
            )
        ))
    return nodes


def chunk_placement_timelines(entries: list) -> list:
    nodes = []
    for e in entries:
        prefix = f"Phase: {e.get('phase', '')}. Profile: {e.get('student_profile', '')}. "
        content = _cat(
            e.get("goal", ""),
            e.get("what_to_do", ""),
            e.get("what_to_avoid", ""),
            e.get("checkpoint", ""),
        )
        nodes.append(TextNode(
            text=prefix + content,
            metadata=_meta(
                section="placement_timelines",
                entry_id=e["id"],
                topic=e.get("phase", ""),
                category="",
                chunk_part="full",
            )
        ))
    return nodes


def chunk_resume_guidance(entries: list) -> list:
    nodes = []
    for e in entries:
        prefix = f"Subtopic: {e.get('subtopic', '')}. Context: {e.get('context', '')}. "
        content = _cat(
            e.get("guidance", ""),
            e.get("example", ""),
        )
        nodes.append(TextNode(
            text=prefix + content,
            metadata=_meta(
                section="resume_guidance",
                entry_id=e["id"],
                topic=e.get("subtopic", ""),
                category="",
                chunk_part="full",
            )
        ))
    return nodes


def chunk_internship_strategy(entries: list) -> list:
    nodes = []
    for e in entries:
        prefix = f"Subtopic: {e.get('subtopic', '')}. Year: {e.get('year_applicable', '')}. "
        content = _cat(
            e.get("guidance", ""),
            e.get("common_mistake", ""),
            e.get("success_signal", ""),
        )
        nodes.append(TextNode(
            text=prefix + content,
            metadata=_meta(
                section="internship_strategy",
                entry_id=e["id"],
                topic=e.get("subtopic", ""),
                category="",
                chunk_part="full",
            )
        ))
    return nodes


def chunk_career_roadmaps(entries: list) -> list:
    """Split each entry into 3 chunks: roadmap / projects / skills."""
    nodes = []
    for e in entries:
        prefix = f"Career path: {e.get('career_path', '')}. "
        meta_base = dict(
            section="career_roadmaps",
            entry_id=e["id"],
            topic=e.get("career_path", ""),
            category="",
        )
        # Carry these in all 3 chunk metadata
        extra = {
            "career_path":    e.get("career_path", ""),
            "time_horizon":   e.get("time_horizon", ""),
            "starting_point": e.get("starting_point", ""),
        }

        # Chunk A — roadmap sequence
        roadmap_content = _cat(
            e.get("phase_1", ""),
            e.get("phase_2", ""),
            e.get("phase_3", ""),
        )
        m_a = _meta(**meta_base, chunk_part="roadmap")
        m_a.update(extra)
        nodes.append(TextNode(text=prefix + roadmap_content, metadata=m_a))

        # Chunk B — projects
        projects_content = e.get("projects_to_build", "")
        m_b = _meta(**meta_base, chunk_part="projects")
        m_b.update(extra)
        nodes.append(TextNode(text=prefix + projects_content, metadata=m_b))

        # Chunk C — skills
        skills_content = e.get("skills_that_matter", "")
        m_c = _meta(**meta_base, chunk_part="skills")
        m_c.update(extra)
        nodes.append(TextNode(text=prefix + skills_content, metadata=m_c))

    return nodes


def chunk_mindset_support(entries: list) -> list:
    nodes = []
    for e in entries:
        prefix = f"Situation: {e.get('situation', '')}. "
        content = _cat(
            e.get("validation", ""),
            e.get("reframe", ""),
            e.get("action", ""),
        )
        nodes.append(TextNode(
            text=prefix + content,
            metadata=_meta(
                section="mindset_support",
                entry_id=e["id"],
                topic=e.get("situation", ""),
                category="",
                chunk_part="full",
            )
        ))
    return nodes


def chunk_higher_studies(entries: list) -> list:
    """Split each entry into 2 chunks: guidance / myths_decision."""
    nodes = []
    for e in entries:
        prefix = f"Subtopic: {e.get('subtopic', '')}. "
        meta_base = dict(
            section="higher_studies",
            entry_id=e["id"],
            topic=e.get("subtopic", ""),
            category="",
        )
        extra = {"subtopic": e.get("subtopic", "")}

        # Chunk A — guidance
        m_a = _meta(**meta_base, chunk_part="guidance")
        m_a.update(extra)
        nodes.append(TextNode(
            text=prefix + e.get("guidance", ""),
            metadata=m_a,
        ))

        # Chunk B — common_myths + decision_factor
        myths_content = _cat(
            e.get("common_myths", ""),
            e.get("decision_factor", ""),
        )
        m_b = _meta(**meta_base, chunk_part="myths_decision")
        m_b.update(extra)
        nodes.append(TextNode(
            text=prefix + myths_content,
            metadata=m_b,
        ))

    return nodes


# ── Main ─────────────────────────────────────────────────────────────────────

SECTION_CHUNKERS = {
    "dsa_concepts":        chunk_dsa_concepts,
    "company_patterns":    chunk_company_patterns,
    "placement_timelines": chunk_placement_timelines,
    "resume_guidance":     chunk_resume_guidance,
    "internship_strategy": chunk_internship_strategy,
    "career_roadmaps":     chunk_career_roadmaps,
    "mindset_support":     chunk_mindset_support,
    "higher_studies":      chunk_higher_studies,
}

EXPECTED_NODE_COUNT = 330  # Actual: 80+40+20+30+25+(25*3)+20+(20*2) = 330


def build_new_index():
    logger.info("==================================================")
    logger.info("   BUILDING EDUMENTOR KNOWLEDGE BASE RAG INDEX")
    logger.info("==================================================")

    # 1. Load knowledge base
    json_path = WORKSPACE_DIR / "edumentor_knowledge_base.json"
    logger.info(f"Loading knowledge base from: {json_path}")
    if not json_path.exists():
        logger.error(f"Knowledge base not found: {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Build nodes with section-aware chunking
    all_nodes = []
    for section, chunker in SECTION_CHUNKERS.items():
        entries = data.get(section, [])
        if not entries:
            logger.warning(f"Section '{section}' is empty or missing — skipping.")
            continue
        section_nodes = chunker(entries)
        logger.info(f"  {section}: {len(entries)} entries → {len(section_nodes)} nodes")
        all_nodes.extend(section_nodes)

    # 3. Verify total node count
    total = len(all_nodes)
    logger.info(f"\nTotal nodes constructed: {total} (expected {EXPECTED_NODE_COUNT})")
    if total != EXPECTED_NODE_COUNT:
        logger.error(
            f"NODE COUNT MISMATCH — got {total}, expected {EXPECTED_NODE_COUNT}. "
            f"Difference: {total - EXPECTED_NODE_COUNT:+d}. Stopping."
        )
        sys.exit(1)

    # 4. Configure embedding model
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

    # 5. Initialize ChromaDB — rebuild fresh, collection: "edumentor_knowledge"
    logger.info(f"Rebuilding ChromaDB at: {CHROMA_PERSIST_DIR}")
    logger.info(f"Collection name: {CHROMA_COLLECTION_NAME}")
    chroma_manager = ChromaManager(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME
    )
    chroma_manager.initialize_db(rebuild_fresh=True)
    vector_store = chroma_manager.get_vector_store()

    # 6. Store in Vector Store Index
    logger.info("Computing semantic embeddings and populating vector store...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        all_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    logger.info("==================================================")
    logger.info(f"  INGESTION COMPLETE — {total} nodes indexed.")
    logger.info("==================================================")
    return index


if __name__ == "__main__":
    build_new_index()
