import os
os.environ["HF_HUB_OFFLINE"] = "1"
import logging
import requests
from pathlib import Path
from typing import List, Optional

from rag.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    CHROMA_COLLECTION_5K,
    EMBEDDING_MODEL_NAME,
    SIMILARITY_TOP_K,
    RERANK_TOP_N,
    RERANK_MODEL_NAME,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)
from rag.database.chroma_manager import ChromaManager

from llama_index.core import VectorStoreIndex, QueryBundle, PromptTemplate
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

logger = logging.getLogger(__name__)

# --- Cached models for offline performance and reuse ---
_EMBED_MODEL_CACHE = None
_RERANKER_CACHE = {}

def get_cached_embed_model():
    global _EMBED_MODEL_CACHE
    if _EMBED_MODEL_CACHE is None:
        _EMBED_MODEL_CACHE = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return _EMBED_MODEL_CACHE

def get_cached_reranker(top_n: int):
    global _RERANKER_CACHE
    if top_n not in _RERANKER_CACHE:
        _RERANKER_CACHE[top_n] = SentenceTransformerRerank(
            model=RERANK_MODEL_NAME,
            top_n=top_n
        )
    return _RERANKER_CACHE[top_n]

# --- 1. Custom Priority Topic Postprocessor ---
class PriorityTopicPostprocessor(BaseNodePostprocessor):
    """
    Custom LlamaIndex Node Postprocessor that adjusts relevance scores of nodes 
    based on the prioritized engineering domains in the user guidelines:
    1. Exact engineering concepts / Programming (dsa, programming, academics)
    2. Projects (project)
    3. Internships / Placements (placement, internship)
    4. Career guidance (career)
    """
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if not nodes:
            return nodes

        # Extract all raw scores to calculate min/max
        scores = [nws.score if nws.score is not None else 0.0 for nws in nodes]
        min_score = min(scores)
        max_score = max(scores)
        denom = max_score - min_score

        priority_weights = {
            "dsa": 1.5,
            "programming": 1.4,
            "academics": 1.35,
            "project": 1.3,
            "internship": 1.25,
            "placement": 1.2,
            "career": 1.1
        }

        boosted_nodes = []
        for nws in nodes:
            node = nws.node
            raw_score = nws.score if nws.score is not None else 0.0
            
            # Normalize to 0-1
            if denom == 0:
                normalized = 1.0
            else:
                normalized = (raw_score - min_score) / (denom + 1e-8)
            
            topic = str(node.metadata.get("topic", "")).lower()
            text = node.text.lower()
            
            boost = 1.0
            # Boost based on metadata topic matches
            for kw, weight in priority_weights.items():
                if kw in topic:
                    boost = max(boost, weight)
            
            # Boost based on body text matches
            for kw, weight in priority_weights.items():
                if kw in text:
                    boost = max(boost, weight * 0.9)  # Lower weight for text body match
            
            # Apply topic boost multiplier on the normalized score
            nws.score = normalized * boost
            boosted_nodes.append(nws)
            
        # Re-sort descending based on updated boosted score
        boosted_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        return boosted_nodes

# --- 2. Custom Query Template ---
QA_PROMPT_TEMPLATE = PromptTemplate(
    "You are EduMentor.\n\n"
    "Use the retrieved engineering knowledge as reference:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Do not copy answers directly.\n"
    "Explain like a senior engineering mentor.\n"
    "Give:\n"
    "- clear explanation\n"
    "- practical steps\n"
    "- examples\n"
    "- mistakes to avoid\n"
    "- learning path when useful\n\n"
    "If context is insufficient, use your own reasoning.\n\n"
    "Student Question: {query_str}\n"
    "EduMentor Answer:"
)

# --- 3. Mock Response / Offline Fallback Engine ---
class QueryResponse:
    def __init__(self, response: str, source_nodes: List[NodeWithScore]):
        self.response = response
        self.source_nodes = source_nodes
        
    def __str__(self):
        return self.response

class StreamingQueryResponse:
    def __init__(self, response_gen, source_nodes: List[NodeWithScore]):
        self.response_gen = response_gen
        self.source_nodes = source_nodes
        self.response = "<Streaming response>"
        
    def __str__(self):
        return self.response

class FallbackEduMentorQueryEngine:
    """
    Offline fallback query engine that activates if Ollama is not running.
    Formats the top-ranked retrieved segments into a beautiful mentor explanation.
    """
    def __init__(self, retriever, postprocessors: list):
        self.retriever = retriever
        self.postprocessors = postprocessors

    def query(self, query_str: str) -> QueryResponse:
        nodes = self.retriever.retrieve(query_str)
        # Apply rerank and priority boosters
        for pp in self.postprocessors:
            nodes = pp.postprocess_nodes(nodes, QueryBundle(query_str))

        from edmentor.safety_filter import edumentor_filter

        # Try to call local LLM if it's active
        if check_ollama_active():
            import requests
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": f"Answer the student's question using your own knowledge as a senior engineering mentor. Do not use markdown, bullet points, or list formatting. Keep the response natural, spoken, and under 200 words.\n\nStudent Question: {query_str}",
                "system": "You are Edmentor, a senior engineering mentor. You answer student questions in plain natural spoken sentences only. Never use markdown, bold, lists, or formatting.",
                "stream": False,
                "options": {
                    "temperature": 0.3
                }
            }
            try:
                r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=10)
                if r.status_code == 200:
                    response_text = r.json().get("response", "").strip()
                    cleaned_response = edumentor_filter(response_text)
                    return QueryResponse(cleaned_response, nodes)
            except Exception as e:
                logger.error(f"Error calling LLM in fallback query engine: {e}")

        if not nodes:
            fallback_text = (
                "I am EduMentor. I currently do not have enough verified context in my knowledge base to answer this question. "
                "However, as a senior mentor, I suggest looking into official documentation and academic roadmaps "
                "to guide your research."
            )
            return QueryResponse(edumentor_filter(fallback_text), [])

        top_nws = nodes[0]
        text = top_nws.node.text

        # Parse out pure Mentor Explanation from formatted document
        explanation = text
        if "Mentor Explanation:" in text:
            parts = text.split("Mentor Explanation:")
            if len(parts) > 1:
                explanation = parts[1].split("Topic:")[0].strip()

        # Clean the explanation through the safety filter to remove all formatting and templates
        clean_explanation = edumentor_filter(explanation)
        return QueryResponse(clean_explanation, nodes)

# --- 4. Loading Index & Retrieving Engine ---
def load_rag_index() -> VectorStoreIndex:
    """Loads the persisted ChromaDB VectorStoreIndex."""
    chroma_manager = ChromaManager(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME
    )
    vector_store = chroma_manager.get_vector_store()
    
    embed_model = get_cached_embed_model()
    
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )
    return index

def check_ollama_active() -> bool:
    """Checks if local Ollama daemon is running."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

# ── Section-aware tone hints ──────────────────────────────────────────────────
_SECTION_TONE_HINTS = {
    "mindset_support": (
        "The student is emotionally struggling. Lead with validation (1 sentence), "
        "then reframe, then one action. Be human, not motivational-poster."
    ),
    "placement_timelines": (
        "Be specific about timing. Name actual topics or platforms. "
        "Don't say 'study hard' — say what to study and when."
    ),
    "company_patterns": (
        "Be specific to the company they asked about. Name the actual round structure, "
        "what they test, what gets people rejected."
    ),
    "career_roadmaps": (
        "Don't give the full roadmap in voice. Give the single most important starting "
        "point and one milestone. Offer to go deeper on any phase if they want."
    ),
    "higher_studies": (
        "Be honest about the financial reality and ROI. Don't romanticise MS abroad or "
        "dismiss it. Give your honest mentor take."
    ),
}

_BASE_SYSTEM = """\
You are EduMentor — a senior software engineer and mentor for Indian engineering \
students (2nd to 4th year). You speak exactly like a real person: casual, direct, \
occasionally blunt, always warm. You have lived through placements, internships, \
DSA grind, burnout — all of it.

STRICT VOICE RULES:
- Reply in 5-8 sentences. Give comprehensive guidance.
- No bullet points. No numbered lists. No markdown. No headers.
- Write exactly how you would speak out loud.
- Use casual English — contractions, "yeah", "look", "honestly" are fine.
- Never start your reply with "I", "Sure", "Great question", "Absolutely".
- If the student sounds anxious or burnt out, acknowledge it in one sentence before answering.
- Give ONE concrete next step, not a full plan.
- Do NOT copy the knowledge context verbatim. Use it to inform your answer, \
then say it in your own mentor voice.\
"""


def _merge_by_id(nodes: list) -> list[str]:
    """
    Section-aware deduplication: merge nodes sharing the same entry id.
    Returns merged context strings (max 3).
    """
    grouped: dict = {}
    for nws in nodes:
        eid = nws.node.metadata.get("id", nws.node.node_id)
        if eid in grouped:
            grouped[eid] += " " + nws.node.text
        else:
            grouped[eid] = nws.node.text
    return list(grouped.values())[:3]  # max 3 merged contexts


class EduMentorRAGQueryEngine:
    """
    Unified query engine for EduMentor.
    - similarity_top_k = 4
    - Relevance score threshold = 0.42
    - Section-aware merge_by_id deduplication before LLM
    """
    def __init__(self, index: VectorStoreIndex, is_5k: bool = False):
        self.index = index
        self.is_5k = is_5k

    def query(self, query_str: str) -> QueryResponse:
        # Fetch best matching chunks
        retriever = self.index.as_retriever(similarity_top_k=4)
        nodes = retriever.retrieve(query_str)

        # Filter by threshold 0.42
        filtered = []
        for nws in nodes:
            dist = nws.score if nws.score is not None else 1.0
            similarity = 1.0 - dist
            if similarity >= 0.42:
                nws.score = similarity
                filtered.append(nws)

        # ── Ollama LLM synthesis ─────────
        top_section: str | None = filtered[0].node.metadata.get("section") if filtered else None
        system = _BASE_SYSTEM
        if top_section and top_section in _SECTION_TONE_HINTS:
            system += "\n\n" + _SECTION_TONE_HINTS[top_section]
        if filtered:
            merged_context = "\n\n".join(_merge_by_id(filtered))
            user_prompt = (
                f"Knowledge context (use this to inform your answer, do not quote it directly):\n{merged_context}\n\n"
                f"Student says: \"{query_str}\"\n\nReply as EduMentor in 2-3 spoken sentences."
            )
        else:
            user_prompt = (
                f"Student says: \"{query_str}\"\n\n"
                "You don't have specific reference material. Answer from your experience as a senior engineer."
            )
        if check_ollama_active():
            import requests
            import json
            try:
                r = requests.post(f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": user_prompt,
                          "system": system, "stream": True, "options": {"temperature": 0.3}},
                    timeout=60, stream=True)
                
                if r.status_code == 200:
                    def generate_chunks():
                        for line in r.iter_lines():
                            if line:
                                data = json.loads(line)
                                if "response" in data:
                                    # Basic inline filtering for markdown
                                    chunk = data["response"]
                                    chunk = chunk.replace("*", "").replace("#", "").replace("`", "")
                                    yield chunk
                                    
                    return StreamingQueryResponse(generate_chunks(), filtered)
            except Exception as e:
                logger.error(f"Error calling Ollama: {e}")
                
        response_text = "Looks like my LLM brain is offline — make sure Ollama is running and try again."
        from edmentor.safety_filter import edumentor_filter
        return QueryResponse(edumentor_filter(response_text), filtered)

def get_edumentor_query_engine(index: VectorStoreIndex):
    """Returns the custom EduMentor Query Engine."""
    return EduMentorRAGQueryEngine(index, is_5k=False)

# ── 5k Dataset Variants ──────────────────────────────────────────────────────

def load_rag_index_5k() -> VectorStoreIndex:
    """
    Loads the persisted ChromaDB VectorStoreIndex for the 'edumentor_5k' collection.
    Must be populated first by running rag/indexing/build_index_5k.py.
    """
    chroma_manager = ChromaManager(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_5K,
    )
    vector_store = chroma_manager.get_vector_store()
    embed_model  = get_cached_embed_model()
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    return index


def get_edmentor_retriever(topic: str = "General", top_k: int = 4):
    """
    Returns a retriever for the edumentor_knowledge collection.
    top_k defaults to 4 (increased for split-entry retrieval headroom).
    """
    chroma_manager = ChromaManager(
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    vector_store = chroma_manager.get_vector_store()
    embed_model  = get_cached_embed_model()
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )

    retriever = index.as_retriever(similarity_top_k=top_k)
    logger.info(f"EdmentorRetriever: top_k={top_k}")
    return retriever


def retrieve_chunks_for_edmentor(query: str, topic: str = "General", top_k: int = 4) -> List[str]:
    """
    Retrieves top-k chunk texts for a query + topic.
    Applies the 0.42 similarity threshold.
    """
    retriever = get_edmentor_retriever(topic=topic, top_k=top_k)
    nodes = retriever.retrieve(query)

    valid_nodes = []
    for nws in nodes:
        dist = nws.score if nws.score is not None else 1.0
        similarity = 1.0 - dist
        if similarity >= 0.42:
            valid_nodes.append(nws)

    return [nws.node.text for nws in valid_nodes[:top_k]]


def get_edumentor_query_engine_5k(index: VectorStoreIndex):
    """Returns a query engine targeting the 'edumentor_5k' collection."""
    return EduMentorRAGQueryEngine(index, is_5k=True)

