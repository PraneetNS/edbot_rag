import re

def edumentor_filter(text: str) -> str:
    """
    Cleans response text for natural voice synthesis:
    - Strips markdown formatting (headers, lists, bold).
    - Removes common AI filler words.
    - Limits response length to 250 words, cutting strictly at the last complete sentence boundary.
    """
    if not text:
        return ""

    # Strip any leftover markdown (finetuning contamination may bleed through)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'#{1,3}\s', '', text)
    text = re.sub(r'^\s*[-*]\s', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s', '', text, flags=re.MULTILINE)
    
    # Remove AI filler phrases
    fillers = [
        "great question", "certainly", "sure!", "absolutely", 
        "of course", "i'd be happy to", "as an ai", "sure"
    ]
    for f in fillers:
        pattern = r'\b' + re.escape(f) + r'\b[,.!?]?'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
    # Standardize whitespace and remove leftover punctuation artifacts at the beginning
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[,\s.!?]+', '', text).strip()
    
    # Truncate at 250 words, cutting at the last complete sentence boundary
    words = text.split()
    if len(words) <= 250:
        return text.strip()
        
    # Split text into sentences using lookbehind to preserve sentence termination marks
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    accumulated_sentences = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words:
            continue
        if current_word_count + len(sentence_words) <= 250:
            accumulated_sentences.append(sentence)
            current_word_count += len(sentence_words)
        else:
            break
            
    if accumulated_sentences:
        text = " ".join(accumulated_sentences)
    else:
        # Last-resort fallback: truncate at exactly 250 words if the first sentence exceeds the limit
        text = " ".join(words[:250]) + "."
        
    return text.strip()
