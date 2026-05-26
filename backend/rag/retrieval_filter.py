import re

def is_low_quality(text: str, min_length: int = 60) -> bool:
    """
    Checks if a chunk is of low quality (too short, mostly repetitive words,
    or dominated by UI junk labels).
    """
    text_clean = text.strip()
    if len(text_clean) < min_length:
        return True
        
    # Check unique word density (repetition check)
    words = [w.lower() for w in re.findall(r'\b\w+\b', text_clean)]
    if not words:
        return True
        
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.50 and len(words) >= 8:
        return True  # High repetition indicates slogan/UI spam
        
    # Check UI/Junk label density
    # If the text is composed mostly of very short lines (under 15 chars) that are common UI items
    ui_keywords = {
        "sign in", "register", "about us", "contact us", "explore", "courses", 
        "platform", "legal", "terms", "policy", "free", "category", "search", 
        "view all", "home", "elms", "login", "logout", "copyright"
    }
    
    lines = [l.strip().lower() for l in text_clean.split("\n") if l.strip()]
    if not lines:
        return True
        
    ui_lines_count = 0
    for line in lines:
        # Check if line is purely a UI keyword or extremely short meaningless fragment
        if line in ui_keywords or len(line) < 6:
            ui_lines_count += 1
            
    ui_ratio = ui_lines_count / len(lines)
    if ui_ratio > 0.40:
        return True  # More than 40% UI labels/junk
        
    return False

def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculates Jaccard similarity of two strings based on lowercase words.
    """
    w1 = set(re.findall(r'\b\w+\b', text1.lower()))
    w2 = set(re.findall(r'\b\w+\b', text2.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1.intersection(w2)) / len(w1.union(w2))

def filter_retrieved_chunks(hits: list, similarity_threshold: float = 0.75) -> list:
    """
    Filters out low-quality and duplicate chunks from retrieved LlamaIndex NodeWithScore hits.
    """
    filtered_hits = []
    seen_texts = []
    
    for h in hits:
        text = h.node.text if hasattr(h, "node") else getattr(h, "text", "")
        if not text:
            continue
            
        # 1. Quality Filter
        if is_low_quality(text):
            continue
            
        # 2. De-duplication check against already accepted chunks
        is_duplicate = False
        for seen in seen_texts:
            if calculate_jaccard_similarity(text, seen) > similarity_threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            filtered_hits.append(h)
            seen_texts.append(text)
            
    return filtered_hits
