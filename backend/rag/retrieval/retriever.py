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
        if not query_bundle:
            return nodes

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
            score = nws.score or 0.0
            
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
            
            nws.score = score * boost
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

        if not nodes:
            fallback_text = (
                "Hello! I am EduMentor.\n\n"
                "I currently do not have enough verified context in my knowledge base to answer this question. "
                "However, as a senior mentor, I suggest looking into official documentation and academic roadmaps "
                "to guide your research!"
            )
            return QueryResponse(fallback_text, [])

        top_nws = nodes[0]
        text = top_nws.node.text
        topic = top_nws.node.metadata.get("topic", "General")
        source = top_nws.node.metadata.get("source", "Unknown")

        # Parse out pure Mentor Explanation from formatted document
        explanation = text
        if "Mentor Explanation:" in text:
            parts = text.split("Mentor Explanation:")
            if len(parts) > 1:
                explanation = parts[1].split("Topic:")[0].strip()

        fallback_text = (
            f"*(EduMentor - Offline Fallback Mode)*\n\n"
            f"### Senior Mentor Explanation:\n"
            f"{explanation}\n\n"
            f"### Actionable Mentoring Steps:\n"
            f"1. Focus your study on **{topic}** from verified references (Source: {source}).\n"
            f"2. Apply these concepts directly in hands-on programming projects.\n"
            f"3. Make mistakes early, debug extensively, and analyze worst-case time complexities.\n\n"
            f"*(Retrieved from verified engineering knowledge base)*"
        )
        return QueryResponse(fallback_text, nodes)

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

def get_edumentor_query_engine(index: VectorStoreIndex):
    """
    Returns the custom EduMentor Query Engine.
    Leverages SentenceTransformerRerank and PriorityTopicPostprocessor.
    Automatically handles offline fallback.
    """
    retriever = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
    
    reranker = get_cached_reranker(RERANK_TOP_N)
    priority_pp = PriorityTopicPostprocessor()
    
    if check_ollama_active():
        logger.info("Local Ollama service detected. Activating full Mistral LLM Query Engine...")
        llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            request_timeout=120.0
        )
        
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=QA_PROMPT_TEMPLATE,
            llm=llm
        )
        
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[reranker, priority_pp],
            response_synthesizer=response_synthesizer
        )
        return query_engine
    else:
        logger.info("Ollama service not running. Initializing Fallback EduMentor Query Engine...")
        return FallbackEduMentorQueryEngine(
            retriever=retriever,
            postprocessors=[reranker, priority_pp]
        )


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


def get_edmentor_retriever(topic: str = "General", top_k: int = 3):
    """
    Returns a retriever for the edumentor_5k collection with optional
    ChromaDB topic metadata pre-filter.

    When topic != 'General', only chunks tagged with that topic are
    searched — DSA queries only retrieve DSA chunks, etc.
    When topic == 'General', no filter is applied (broad retrieval).

    Args:
        topic:  Topic label from EdmentorTopicClassifier.classify().
                Must match the 'topic' metadata field in ChromaDB.
        top_k:  Number of chunks to retrieve before reranking.

    Returns:
        A LlamaIndex retriever with optional ChromaDB where filter.
    """
    from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator

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

    # Apply topic metadata filter when not General
    if topic and topic.lower() != "general":
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="topic",
                    value=topic,
                    operator=FilterOperator.EQ,
                )
            ]
        )
        retriever = index.as_retriever(
            similarity_top_k=top_k,
            filters=filters,
        )
        logger.info(f"EdmentorRetriever: topic filter='{topic}', top_k={top_k}")
    else:
        retriever = index.as_retriever(similarity_top_k=top_k)
        logger.info(f"EdmentorRetriever: no topic filter (General), top_k={top_k}")

    return retriever


def retrieve_chunks_for_edmentor(query: str, topic: str = "General", top_k: int = 3) -> List[str]:
    """
    Convenience function: retrieve top-k chunk texts for a query + topic.
    Applies reranking and priority boost. Returns plain text list for prompt injection.

    Args:
        query:  The student's question.
        topic:  Classified topic for metadata pre-filter.
        top_k:  Number of final chunks to return.

    Returns:
        List of chunk text strings (already in Student:/Mentor: format).
    """
    retriever = get_edmentor_retriever(topic=topic, top_k=top_k * 2)  # over-fetch then rerank
    nodes = retriever.retrieve(query)

    # Apply reranker
    try:
        reranker = get_cached_reranker(top_k)
        priority_pp = PriorityTopicPostprocessor()
        from llama_index.core import QueryBundle
        nodes = reranker.postprocess_nodes(nodes, QueryBundle(query))
        nodes = priority_pp.postprocess_nodes(nodes, QueryBundle(query))
    except Exception as e:
        logger.warning(f"Reranking failed, using raw retrieval: {e}")

    # Tighten similarity threshold to 0.85
    # nws.score represents cosine similarity. If it is less than 0.85, we filter it out.
    valid_nodes = [nws for nws in nodes if nws.score is not None and nws.score >= 0.85]

    return [nws.node.text for nws in valid_nodes[:top_k]]


def get_edumentor_query_engine_5k(index: VectorStoreIndex):
    """
    Returns a query engine targeting the 'edumentor_5k' collection.
    Identical pipeline to the main engine (reranker + priority postprocessor),
    but backed by the deduplicated synthetic 5k dataset.
    """
    retriever    = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
    reranker     = get_cached_reranker(RERANK_TOP_N)
    priority_pp  = PriorityTopicPostprocessor()

    if check_ollama_active():
        logger.info("[5k] Ollama detected — activating Mistral LLM query engine.")
        llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            request_timeout=120.0,
        )
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=QA_PROMPT_TEMPLATE,
            llm=llm,
        )
        return RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[reranker, priority_pp],
            response_synthesizer=response_synthesizer,
        )
    else:
        logger.info("[5k] Ollama offline — using Fallback EduMentor Query Engine.")
        return FallbackEduMentorQueryEngine(
            retriever=retriever,
            postprocessors=[reranker, priority_pp],
        )
