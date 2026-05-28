import re
from pathlib import Path
from llama_index.core.schema import NodeWithScore, TextNode
from rag.bm25 import BM25Retriever

class HybridRetriever:
    """
    Implements a production-grade Hybrid Retriever.
    Combines dense semantic representations (BAAI/bge-large-en-v1.5)
    with sparse keyword relevance (BM25) using Min-Max score normalization
    and linear score fusion (0.6 * Dense + 0.4 * Sparse).
    """
    def __init__(self, index, collection, similarity_top_k: int = 6):
        self.index = index
        self.collection = collection
        self.similarity_top_k = similarity_top_k
        self.vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k * 3)
        
        # Load BM25 Retriever
        self.bm25 = BM25Retriever()
        bm25_path = Path(__file__).resolve().parent.parent / "data" / "bm25_index.json"
        if bm25_path.exists():
            print(f"Loading BM25 index from {bm25_path}...")
            try:
                self.bm25.load(str(bm25_path))
                print("BM25 index loaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to load BM25 index ({e}). It will be re-initialized if needed.")
        else:
            print("Warning: BM25 index file not found. Sparse search will be disabled until ingest.py is run.")

    def retrieve(self, query_text: str) -> list[NodeWithScore]:
        """
        Retrieves top nodes by fusing normalized Dense and Sparse BM25 scores.
        """
        # 1. Fetch Dense Candidates
        dense_hits = self.vector_retriever.retrieve(query_text)
        
        # If BM25 index is not loaded, fallback to pure dense search
        if not self.bm25 or self.bm25.corpus_size == 0:
            print("Warning: BM25 index not initialized. Falling back to pure Dense Vector search.")
            return dense_hits[:self.similarity_top_k]

        # 2. Fetch Sparse BM25 Candidate Scores
        bm25_scores = self.bm25.get_scores(query_text)
        
        # Sort documents by BM25 score and get top candidates
        top_sparse_ids = sorted(bm25_scores.keys(), key=lambda k: bm25_scores[k], reverse=True)[:self.similarity_top_k * 3]
        
        # Merge candidate pools
        dense_ids = {h.node.node_id for h in dense_hits}
        sparse_ids = set(top_sparse_ids)
        
        # Find sparse candidates that are not in dense candidates, and fetch them from ChromaDB
        missing_ids = list(sparse_ids - dense_ids)
        missing_hits = []
        if missing_ids:
            try:
                # Query ChromaDB collection directly for missing nodes by ID
                chroma_results = self.collection.get(ids=missing_ids)
                if chroma_results and "documents" in chroma_results:
                    for idx, doc_text in enumerate(chroma_results["documents"]):
                        doc_id = chroma_results["ids"][idx]
                        doc_metadata = chroma_results["metadatas"][idx] if chroma_results["metadatas"] else {}
                        
                        node = TextNode(
                            id_=doc_id,
                            text=normalize_text(doc_text),
                            metadata=doc_metadata
                        )
                        # We use 0.0 as default dense score for dense-missed nodes
                        missing_hits.append(NodeWithScore(node=node, score=0.0))
            except Exception as e:
                print(f"Warning: Failed to fetch missing sparse nodes from ChromaDB ({e})")

        # Compile total pool of candidates (Dense Hits + Missing Sparse Hits)
        candidate_pool = dense_hits + missing_hits
        if not candidate_pool:
            return []

        # 3. Score Mapping
        # Build dictionaries for dense and sparse scores
        dense_scores_dict = {}
        for h in dense_hits:
            dense_scores_dict[h.node.node_id] = h.score if h.score is not None else 0.0
            
        for h in missing_hits:
            dense_scores_dict[h.node.node_id] = 0.0  # Dense miss

        sparse_scores_dict = {}
        for h in candidate_pool:
            sparse_scores_dict[h.node.node_id] = bm25_scores.get(h.node.node_id, 0.0)

        # 4. Min-Max Normalization
        # Normalize Dense Scores
        dense_vals = list(dense_scores_dict.values())
        min_dense, max_dense = min(dense_vals), max(dense_vals)
        dense_range = max_dense - min_dense
        
        # Normalize Sparse Scores
        sparse_vals = list(sparse_scores_dict.values())
        min_sparse, max_sparse = min(sparse_vals), max(sparse_vals)
        sparse_range = max_sparse - min_sparse

        # 5. Hybrid Score Fusion (Linear Weighted)
        fused_hits = []
        for h in candidate_pool:
            node_id = h.node.node_id
            
            # Min-Max normalize dense score to [0, 1]
            raw_dense = dense_scores_dict[node_id]
            norm_dense = (raw_dense - min_dense) / dense_range if dense_range > 0 else 0.5
            
            # Min-Max normalize sparse score to [0, 1]
            raw_sparse = sparse_scores_dict[node_id]
            norm_sparse = (raw_sparse - min_sparse) / sparse_range if sparse_range > 0 else 0.5
            
            # Compute linearly fused score (0.6 * Dense + 0.4 * Sparse)
            hybrid_score = (0.6 * norm_dense) + (0.4 * norm_sparse)
            
            # Update hit's score
            h.score = hybrid_score
            fused_hits.append(h)

        # Sort combined list by fused score descending
        fused_hits.sort(key=lambda x: x.score, reverse=True)

        return fused_hits[:self.similarity_top_k]

def normalize_text(text: str) -> str:
    """Helper to clean whitespace from database text inputs."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()
