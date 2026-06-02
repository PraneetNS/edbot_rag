import re
import logging

logger = logging.getLogger(__name__)

class ContextCompressor:
    """
    Cleans and compresses retrieved context documents by removing duplicate sentences,
    excess whitespace, and structural boilerplate to minimize token usage.
    """
    def __init__(self):
        pass

    def compress(self, context_str: str) -> str:
        if not context_str.strip():
            return ""

        # 1. Normalize line endings and double newlines
        text = re.sub(r'\n+', '\n', context_str)
        
        # 2. Split into sentences (simple dot-space sentence boundary segmenter)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        seen_sentences = set()
        unique_sentences = []
        
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
                
            # Normalize sentence for comparison (lowercase, extract alpha)
            s_norm = "".join(re.findall(r'\w+', s_clean.lower()))
            
            # Avoid too short sentences and duplicates
            if len(s_norm) < 5:
                continue
                
            if s_norm not in seen_sentences:
                seen_sentences.add(s_norm)
                unique_sentences.append(s_clean)
                
        compressed = " ".join(unique_sentences)
        
        original_len = len(context_str)
        compressed_len = len(compressed)
        logger.info(f"Context compressed: {original_len} chars -> {compressed_len} chars (Reduced by {(1 - compressed_len/original_len)*100:.1f}%)" if original_len > 0 else "Context compressed.")
        
        return compressed
