"""
rag_pipeline_5k.py
══════════════════════════════════════════════════════════════════════════════
Full end-to-end RAG pipeline for the Edumentor Synthetic 5k Dataset.

This self-contained module exposes four high-level functions:

  ingest()          — Load, chunk, embed, and upsert into ChromaDB
  search(query)     — Embed query → retrieve top-K semantically similar chunks
  answer(query)     — Retrieve + generate a grounded LLM response
  interactive_cli() — REPL-style question / answer loop

Usage
─────
  # 1. Ingest the dataset (run once, or whenever the dataset changes)
  python rag_pipeline_5k.py --ingest

  # 2. Ask a single question and get an answer
  python rag_pipeline_5k.py --query "How do I prepare for placements?"

  # 3. Launch interactive Q&A REPL
  python rag_pipeline_5k.py --chat

  # 4. Run all built-in smoke-test queries
  python rag_pipeline_5k.py --test

Architecture
────────────
  Dataset  ──→  Dedup & Load  ──→  Sentence Segment  ──→  Batch Embed
                                                               │
  ChromaDB ←── VectorStoreIndex ←── Semantic Chunk  ←─────────┘
      │
      ├──→ Embed Query  ──→  ANN Search (cosine) ──→  Top-K chunks
      │
      └──→ Cross-Encoder Rerank ──→ Priority Boost ──→ LLM Synthesis
                                                           │
                                                      Final Answer
"""

from __future__ import annotations

import io
import sys
import logging
import argparse
import textwrap
from pathlib import Path
from typing import List, Tuple

# ── Windows UTF-8 stdout fix (avoids cp1252 UnicodeEncodeError) ───────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Bootstrap: ensure backend root is importable ──────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,           # suppress noisy library output by default
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("rag").setLevel(logging.INFO)
logger = logging.getLogger("rag_pipeline_5k")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — INGESTION                                                      ║
# ║  Load → Deduplicate → Chunk → Embed → Upsert to ChromaDB                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def ingest(force_rebuild: bool = True) -> int:
    """
    Full ingestion pipeline for edumentor_synthetic_5k.jsonl.

    Args:
        force_rebuild: If True (default), wipes and rebuilds the 'edumentor_5k'
                       ChromaDB collection from scratch.

    Returns:
        Number of semantic chunks stored in the vector database.
    """
    import numpy as np
    from rag.config import (
        DATASET_5K_PATH, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_5K,
        EMBEDDING_MODEL_NAME, BREAKPOINT_PERCENTILE_THRESHOLD,
    )
    from rag.loaders.jsonl_loader import load_jsonl_conversations
    from rag.indexing.custom_splitter import custom_sentence_splitter
    from rag.database.chroma_manager import ChromaManager
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    print("\n" + "=" * 60)
    print("  EDUMENTOR 5K -- RAG INGESTION PIPELINE")
    print("=" * 60)

    # ── 1. Load & deduplicate ─────────────────────────────────────────────────
    print(f"\n[1/5] Loading dataset from:\n      {DATASET_5K_PATH}")
    if not DATASET_5K_PATH.exists():
        print(f"  [FAIL] File not found: {DATASET_5K_PATH}")
        sys.exit(1)

    docs = load_jsonl_conversations(str(DATASET_5K_PATH), deduplicate=True)
    print(f"  [OK] {len(docs)} unique Q&A pairs loaded (duplicates stripped).")

    # ── 2. Sentence-segment ───────────────────────────────────────────────────
    print("\n[2/5] Segmenting documents into sentence units ...")
    doc_sentences: List[List[str]] = [custom_sentence_splitter(d.text) for d in docs]
    flat_sentences: List[str] = []
    sentence_map: List[Tuple[int, int]] = []
    for doc_idx, sents in enumerate(doc_sentences):
        for s_idx, s in enumerate(sents):
            flat_sentences.append(s)
            sentence_map.append((doc_idx, s_idx))
    print(f"  [OK] {len(flat_sentences)} total sentence segments across {len(docs)} docs.")

    # ── 3. Batch embed ────────────────────────────────────────────────────────
    print(f"\n[3/5] Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    print(f"  Generating embeddings for {len(flat_sentences)} sentences ...")
    flat_embeddings = embed_model.get_text_embedding_batch(flat_sentences, show_progress=True)
    print("  [OK] Embeddings generated.")

    # Re-group per document
    doc_embeddings: List[List[List[float]]] = [[] for _ in docs]
    for idx, emb in enumerate(flat_embeddings):
        doc_embeddings[sentence_map[idx][0]].append(emb)

    # ── 4. Semantic chunking ──────────────────────────────────────────────────
    print(f"\n[4/5] Applying semantic chunking (p{BREAKPOINT_PERCENTILE_THRESHOLD} threshold) ...")

    def _cosine_dist(a, b):
        a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        if np.all(a == 0) or np.all(b == 0): return 0.0
        return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    nodes: List[TextNode] = []
    for doc_idx, doc in enumerate(docs):
        sents = doc_sentences[doc_idx]
        embs  = doc_embeddings[doc_idx]
        if len(sents) <= 1:
            nodes.append(TextNode(text=doc.text, metadata=doc.metadata))
            continue
        dists = [_cosine_dist(embs[i], embs[i+1]) for i in range(len(embs) - 1)]
        threshold = float(np.percentile(dists, BREAKPOINT_PERCENTILE_THRESHOLD)) if dists else 1.0
        current_chunk, chunks = [sents[0]], []
        for i, dist in enumerate(dists):
            if dist > threshold:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [sents[i + 1]]
            else:
                current_chunk.append(sents[i + 1])
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        for chunk_text in chunks:
            nodes.append(TextNode(text=chunk_text, metadata=doc.metadata))

    print(f"  [OK] {len(nodes)} semantic chunks produced.")

    # ── 5. Upsert into ChromaDB ───────────────────────────────────────────────
    print(f"\n[5/5] Upserting into ChromaDB collection '{CHROMA_COLLECTION_5K}' ...")
    chroma_mgr = ChromaManager(persist_dir=CHROMA_PERSIST_DIR, collection_name=CHROMA_COLLECTION_5K)
    chroma_mgr.initialize_db(rebuild_fresh=force_rebuild)
    vector_store    = chroma_mgr.get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model, show_progress=True)

    print("\n" + "=" * 60)
    print(f"  [OK] INGESTION COMPLETE  --  {len(nodes)} chunks stored")
    print(f"    Collection  : {CHROMA_COLLECTION_5K}")
    print(f"    Persist dir : {CHROMA_PERSIST_DIR}")
    print("=" * 60 + "\n")
    return len(nodes)


# ==============================================================================
#   SECTION 2 -- SEMANTIC SEARCH
#   Embed query --> ANN lookup --> Rerank --> Return top-K chunks with scores
# ==============================================================================

def search(query: str, top_k: int = 5) -> List[dict]:
    """
    Embed a query and retrieve the top-K most semantically similar chunks.

    Args:
        query: Free-text question or search string.
        top_k: Number of results to return (before reranking).

    Returns:
        List of dicts with keys: 'text', 'score', 'topic', 'source', 'question'
    """
    from rag.retrieval.retriever import load_rag_index_5k
    from llama_index.core import QueryBundle

    index    = load_rag_index_5k()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes    = retriever.retrieve(query)

    results = []
    for nws in nodes:
        results.append({
            "text":     nws.node.text,
            "score":    round(nws.score or 0.0, 4),
            "topic":    nws.node.metadata.get("topic",    "General"),
            "source":   nws.node.metadata.get("source",   "Unknown"),
            "question": nws.node.metadata.get("original_question", ""),
        })
    return results


# ==============================================================================
#   SECTION 3 -- RAG ANSWER GENERATION
#   Retrieve --> Rerank --> Synthesize grounded LLM response
# ==============================================================================

def answer(query: str) -> Tuple[str, List[dict]]:
    """
    Full RAG pipeline: retrieve relevant chunks then synthesize a grounded answer.

    If Ollama is running, uses Mistral for generation.
    If Ollama is offline, formats the top retrieved chunk into a structured
    mentor-style response (FallbackEduMentorQueryEngine).

    Args:
        query: The student's question.

    Returns:
        Tuple of (answer_text, list_of_retrieved_chunk_dicts)
    """
    from rag.retrieval.retriever import load_rag_index_5k, get_edumentor_query_engine_5k

    index        = load_rag_index_5k()
    query_engine = get_edumentor_query_engine_5k(index)
    response_obj = query_engine.query(query)
    response_str = str(response_obj)

    # Collect source chunks for transparency
    source_nodes = getattr(response_obj, "source_nodes", [])
    sources = [
        {
            "text":     nws.node.text[:300] + "..." if len(nws.node.text) > 300 else nws.node.text,
            "score":    round(nws.score or 0.0, 4),
            "topic":    nws.node.metadata.get("topic",    "General"),
            "source":   nws.node.metadata.get("source",   "Unknown"),
            "question": nws.node.metadata.get("original_question", ""),
        }
        for nws in source_nodes
    ]
    return response_str, sources


# ==============================================================================
#   SECTION 4 -- PRETTY PRINTING HELPERS
# ==============================================================================

def _print_chunks(chunks: List[dict], header: str = "Retrieved Chunks") -> None:
    sep = "-" * 60
    print(f"\n  +-- {header} " + "-" * max(1, 52 - len(header)) + "+")
    for i, c in enumerate(chunks, 1):
        print(f"  |  [{i}] Score: {c['score']:.4f}  |  Topic: {c['topic']}")
        q = c.get("question", "")
        if q:
            print(f"  |      Question: {q[:80]}{'...' if len(q) > 80 else ''}")
        snippet = c["text"].replace("\n", " ")[:120]
        print(f"  |      Text: {snippet}...")
    print("  +" + "-" * 58 + "+")


def _print_answer(query: str, ans: str, sources: List[dict]) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  Q: {textwrap.fill(query, width - 5)}")
    print("-" * width)
    for line in textwrap.wrap(ans, width - 2):
        print(f"  {line}")
    if sources:
        _print_chunks(sources, "Grounding Sources")
    print("=" * width)


# ==============================================================================
#   SECTION 5 -- SMOKE TESTS
# ==============================================================================

_SMOKE_TEST_QUERIES = [
    "How do I prepare for placements as a final year student?",
    "Should I learn DSA even for frontend jobs?",
    "What is the difference between frontend and backend development?",
    "How do I stop procrastinating on DSA practice?",
    "Should I learn cloud computing during college?",
]


def run_smoke_tests() -> None:
    """Run built-in smoke-test queries against the 5k index and print results."""
    print("\n" + "=" * 60)
    print("  SMOKE TESTS -- Edumentor 5k RAG Pipeline")
    print("=" * 60)

    passed = 0
    for i, q in enumerate(_SMOKE_TEST_QUERIES, 1):
        print(f"\n[Test {i}/{len(_SMOKE_TEST_QUERIES)}] {q}")
        try:
            ans, sources = answer(q)
            _print_answer(q, ans, sources[:2])   # show top-2 sources
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] ERROR: {exc}")

    print(f"\n  Tests passed: {passed}/{len(_SMOKE_TEST_QUERIES)}")
    print("=" * 60 + "\n")


# ==============================================================================
#   SECTION 6 -- INTERACTIVE CLI
# ==============================================================================

def interactive_cli() -> None:
    """Interactive Q&A REPL. Type 'exit' or Ctrl-C to quit."""
    print("\n" + "=" * 60)
    print("  EduMentor 5k  -  Interactive RAG Chat")
    print("  Type 'exit' to quit, 'search <query>' for raw retrieval")
    print("=" * 60)

    while True:
        try:
            raw = input("\n  You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        if not raw:
            continue
        if raw.lower() in {"exit", "quit", "q"}:
            print("  Bye!")
            break

        if raw.lower().startswith("search "):
            q = raw[7:].strip()
            if not q:
                print("  Usage: search <your question>")
                continue
            try:
                chunks = search(q, top_k=5)
                _print_chunks(chunks, f"Top {len(chunks)} results for: '{q}'")
            except Exception as exc:
                print(f"  ✗ Search error: {exc}")
        else:
            try:
                ans, sources = answer(raw)
                _print_answer(raw, ans, sources)
            except Exception as exc:
                print(f"  ✗ Error: {exc}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag_pipeline_5k",
        description="Edumentor Synthetic 5k — Full RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python rag_pipeline_5k.py --ingest
              python rag_pipeline_5k.py --query "How do I get an internship?"
              python rag_pipeline_5k.py --search "DSA practice tips" --top-k 3
              python rag_pipeline_5k.py --chat
              python rag_pipeline_5k.py --test
        """),
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--ingest",  action="store_true", help="Run full ingestion (build index from dataset)")
    group.add_argument("--query",   metavar="QUESTION",  help="Ask a single question and get a grounded answer")
    group.add_argument("--search",  metavar="QUERY",     help="Raw semantic search — returns top-K chunks")
    group.add_argument("--chat",    action="store_true", help="Launch interactive Q&A REPL")
    group.add_argument("--test",    action="store_true", help="Run built-in smoke-test queries")
    p.add_argument("--top-k", type=int, default=5, metavar="K", help="Number of chunks to retrieve (default: 5)")
    p.add_argument("--no-rebuild", action="store_true",          help="Skip fresh rebuild during ingestion (append mode)")
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.ingest:
        ingest(force_rebuild=not args.no_rebuild)

    elif args.query:
        ans, sources = answer(args.query)
        _print_answer(args.query, ans, sources)

    elif args.search:
        chunks = search(args.search, top_k=args.top_k)
        _print_chunks(chunks, f"Top {len(chunks)} chunks for: '{args.search}'")

    elif args.chat:
        interactive_cli()

    elif args.test:
        run_smoke_tests()


if __name__ == "__main__":
    main()
