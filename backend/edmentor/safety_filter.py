import re

MAX_WORDS = 60  # voice contract is 60 words / 80 tokens

def edumentor_filter(text: str, max_words: int = MAX_WORDS) -> str:
    """
    Cleans response text for natural voice synthesis:
    - Strips markdown formatting (headers, lists, bold).
    - Removes common AI filler words.
    - Limits response length to max_words, cutting strictly at the last complete sentence boundary.
    """
    if not text:
        return ""

    # Strip any line starting with * or containing ###
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("*") or "###" in stripped_line:
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # Remove fenced code blocks entirely
    text = re.sub(r"```[\s\S]*?```", "", text)

    # Remove inline code
    text = re.sub(r"`[^`]+`", lambda m: m.group(0).strip("`"), text)
    text = text.replace("`", "")

    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    text = text.replace("*", "").replace("_", "")

    # Remove ATX headers
    text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = text.replace("#", "")

    # Convert numbered lists to natural flow
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Remove dash/asterisk bullet points
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n\s*[-*•]\s+", " ", text)

    # Remove parenthetical meta-tags or reference source lines
    text = re.sub(r"\*\(.*?\)\*", "", text)
    text = re.sub(r"\(Source:.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[Source:.*?\]", "", text, flags=re.IGNORECASE)
    
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
    
    # Truncate at max_words, cutting at the last complete sentence boundary
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
        
    # Split text into sentences using lookbehind to preserve sentence termination marks
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    accumulated_sentences = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words:
            continue
        if current_word_count + len(sentence_words) <= max_words:
            accumulated_sentences.append(sentence)
            current_word_count += len(sentence_words)
        else:
            break
            
    if accumulated_sentences:
        text = " ".join(accumulated_sentences)
    else:
        # Last-resort fallback: truncate at exactly max_words if the first sentence exceeds the limit
        text = " ".join(words[:max_words]) + "."
        
    return text.strip()
