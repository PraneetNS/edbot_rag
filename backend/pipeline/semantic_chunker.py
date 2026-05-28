import sys
import re
from pathlib import Path

# Add parent path to import config
sys.path.append(str(Path(__file__).resolve().parent))
import config

class SemanticEducationalChunker:
    """
    Sentence and heading-aware semantic chunker that splits educational content
    preserving topic continuity and structural hierarchy.
    """
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _approximate_tokens(self, text: str) -> int:
        """
        Simple, robust token count approximation based on words.
        """
        words = text.split()
        return len(words)

    def split_text(self, text: str) -> list[str]:
        """
        Splits text based on section boundaries, heading indicators, and paragraph blank lines,
        ensuring chunks stay between 300-500 tokens with proper overlaps.
        """
        if not text:
            return []

        # Split text into paragraphs based on double newlines (common in scraped layouts)
        raw_paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for p in raw_paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue
                
            p_tokens = self._approximate_tokens(p_clean)
            
            # Check if this paragraph is a heading (typically short, bold, or markdown-formatted)
            is_heading = len(p_clean) < 100 and (p_clean.startswith("#") or p_clean.isupper() or any(p_clean.startswith(x) for x in ["Course:", "Syllabus:", "FAQ:", "Q:", "A:", "Step:"]))
            
            if is_heading and current_chunk:
                # Heading marks a new logical section. Finalize the current chunk if it has substantial size.
                if current_tokens >= 200:
                    chunks.append("\n\n".join(current_chunk))
                    # Handle overlap by taking last elements
                    overlap_size = 0
                    overlap_chunk = []
                    for item in reversed(current_chunk):
                        item_tokens = self._approximate_tokens(item)
                        if overlap_size + item_tokens <= self.chunk_overlap:
                            overlap_chunk.insert(0, item)
                            overlap_size += item_tokens
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_tokens = overlap_size
            
            # Append paragraph
            if current_tokens + p_tokens <= self.chunk_size:
                current_chunk.append(p_clean)
                current_tokens += p_tokens
            else:
                # Chunk size exceeded. Save current chunk.
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                
                # Handle overlap
                overlap_size = 0
                overlap_chunk = []
                for item in reversed(current_chunk):
                    item_tokens = self._approximate_tokens(item)
                    if overlap_size + item_tokens <= self.chunk_overlap:
                        overlap_chunk.insert(0, item)
                        overlap_size += item_tokens
                    else:
                        break
                        
                # Start new chunk with overlaps and current paragraph
                current_chunk = overlap_chunk + [p_clean]
                current_tokens = overlap_size + p_tokens

        # Save remaining chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        # Clean empty chunks
        chunks = [c.strip() for c in chunks if c.strip()]
        
        # Verify sizes and split very long paragraphs if necessary
        verified_chunks = []
        for c in chunks:
            if self._approximate_tokens(c) > self.chunk_size * 1.5:
                # Split roughly in half by sentences if paragraph was extremely huge
                sentences = re.split(r'(?<=[.!?])\s+', c)
                mid = len(sentences) // 2
                verified_chunks.append(" ".join(sentences[:mid]))
                verified_chunks.append(" ".join(sentences[mid:]))
            else:
                verified_chunks.append(c)

        return verified_chunks

if __name__ == "__main__":
    chunker = SemanticEducationalChunker()
    test_text = "\n\n".join([f"Paragraph {i} is a block of text discussing general syllabus topics. " * 30 for i in range(5)])
    split_chunks = chunker.split_text(test_text)
    print(f"Chunks split: {len(split_chunks)}")
    for idx, c in enumerate(split_chunks):
        print(f"\nChunk {idx+1} ({chunker._approximate_tokens(c)} tokens):\n{c[:100]}...")
