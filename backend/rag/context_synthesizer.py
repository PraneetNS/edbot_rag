import re

class ContextSynthesizer:
    """
    Synthesizes multiple educational text chunks into a unified, coherent
    summary designed for students, avoiding literal-only fragmentation.
    """
    def __init__(self):
        pass

    def synthesize(self, query: str, hits: list) -> str:
        """
        Synthesizes multiple retrieved chunks into a single student-friendly summary.
        """
        if not hits:
            return ""

        # Gather distinct substantive sentences/points from chunks
        all_sentences = []
        seen = set()
        
        # Simple regex for sentence splitting
        sentence_endings = re.compile(r'(?<=[.!?])\s+')

        for h in hits:
            text = h.node.text if hasattr(h, "node") else getattr(h, "text", "")
            if not text:
                continue
                
            lines = text.split("\n")
            for line in lines:
                cleaned_line = line.strip()
                if not cleaned_line or len(cleaned_line) < 15:
                    continue
                    
                # Split line into sentences
                sentences = sentence_endings.split(cleaned_line)
                for s in sentences:
                    s_clean = s.strip()
                    if len(s_clean) < 15 or s_clean.endswith('?'):
                        continue
                        
                    # Normalize spacing
                    s_norm = re.sub(r'\s+', ' ', s_clean)
                    s_lower = s_norm.lower()
                    
                    if s_lower not in seen:
                        seen.add(s_lower)
                        all_sentences.append(s_norm)

        if not all_sentences:
            return ""

        # Identify query keywords to prioritize highly relevant sentences
        query_words = [w.lower() for w in re.findall(r'\b\w{4,}\b', query)]
        scored_sentences = []
        for s in all_sentences:
            match_count = sum(1 for qw in query_words if qw in s.lower())
            scored_sentences.append((s, match_count))
            
        # Sort by relevance to query keywords
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top 3-4 sentences and join them cohesively
        top_sentences = [item[0] for item in scored_sentences[:4]]
        
        # Format the synthesis into a unified conversational paragraph
        synthesized_text = " ".join(top_sentences)
        
        # Clean double punctuation
        synthesized_text = re.sub(r'\.+', '.', synthesized_text)
        
        return synthesized_text.strip()
