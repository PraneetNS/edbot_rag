import math
import re
import json
from pathlib import Path

class BM25Retriever:
    """
    An optimized, pure-Python BM25 keyword search engine.
    Computes TF-IDF term weights and length normalization for high-precision
    academic and technical term matching.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens = []
        self.doc_ids = []
        self.term_freqs = []  # List of dicts: [{term: count}, ...]
        self.doc_freq = {}    # Dict: {term: doc_count}
        self.idf = {}         # Dict: {term: idf_score}

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenizes text into a clean list of lowercased alphanumeric words.
        """
        if not text:
            return []
        # Extract alphanumeric words >= 2 chars, ignore basic punctuation
        return re.findall(r'\b\w{2,}\b', text.lower())

    def fit(self, documents: list[dict]):
        """
        Fits the BM25 model on a list of document chunks.
        Each doc in documents must have 'id' and 'content' keys.
        """
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return

        self.doc_ids = []
        self.doc_lens = []
        self.term_freqs = []
        self.doc_freq = {}
        total_len = 0

        for doc in documents:
            doc_id = doc["id"]
            content = doc["content"]
            tokens = self._tokenize(content)
            
            self.doc_ids.append(doc_id)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len
            
            # Compute term frequencies for this document
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)
            
            # Increment document frequencies for distinct terms
            for token in tf.keys():
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0.0

        # Compute Inverse Document Frequency (IDF) for all terms
        for term, df in self.doc_freq.items():
            # Standard BM25 IDF formulation with a floor to prevent negative IDFs
            idf_val = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))
            self.idf[term] = max(0.0001, idf_val)

    def get_scores(self, query: str) -> dict[str, float]:
        """
        Computes BM25 similarity scores for all documents given a query string.
        Returns a dictionary mapping document ID -> BM25 score.
        """
        query_tokens = self._tokenize(query)
        scores = {doc_id: 0.0 for doc_id in self.doc_ids}
        
        if not query_tokens or self.corpus_size == 0:
            return scores

        for idx, doc_id in enumerate(self.doc_ids):
            tf_dict = self.term_freqs[idx]
            doc_len = self.doc_lens[idx]
            
            # Avoid divide-by-zero if document is empty
            denom_boost = self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len)) if self.avg_doc_len > 0 else 1.0
            
            doc_score = 0.0
            for token in query_tokens:
                if token in tf_dict:
                    tf = tf_dict[token]
                    idf_val = self.idf.get(token, 0.0)
                    
                    # BM25 numerator and denominator term
                    term_score = idf_val * (tf * (self.k1 + 1.0)) / (tf + denom_boost)
                    doc_score += term_score
            
            scores[doc_id] = doc_score

        return scores

    def save(self, filepath: str):
        """
        Serializes the BM25 model's states to a JSON file.
        """
        state = {
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self.corpus_size,
            "avg_doc_len": self.avg_doc_len,
            "doc_lens": self.doc_lens,
            "doc_ids": self.doc_ids,
            "term_freqs": self.term_freqs,
            "doc_freq": self.doc_freq,
            "idf": self.idf
        }
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """
        Deserializes BM25 parameters from a JSON file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        self.k1 = state["k1"]
        self.b = state["b"]
        self.corpus_size = state["corpus_size"]
        self.avg_doc_len = state["avg_doc_len"]
        self.doc_lens = state["doc_lens"]
        self.doc_ids = state["doc_ids"]
        self.term_freqs = state["term_freqs"]
        self.doc_freq = state["doc_freq"]
        self.idf = state["idf"]
