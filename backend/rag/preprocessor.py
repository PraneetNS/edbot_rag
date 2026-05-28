import re
from rag.semantic_query_expander import SemanticQueryExpander

# Context-aware trigger keywords for ambiguous abbreviation "OS"
OS_TRIGGERS = {
    "syllabus", "exam", "lab", "prep", "preparation", "course", 
    "subject", "learn", "study", "class", "curriculum", "engineering", 
    "computer", "science", "software", "system", "vtu", "lms"
}

def preprocess_query(query: str, active_topic: str = None, last_courses: list = None) -> str:
    """
    Normalizes educational abbreviations and enhances query semantics
    using the SemanticQueryExpander to maximize retrieval relevance.
    """
    if not query:
        return ""
        
    normalized = query.strip()
    
    # 1. Expand standard unambiguous abbreviations
    unambiguous_map = {
        r"\bDBMS\b": "Database Management Systems",
        r"\bDSA\b": "Data Structures and Algorithms",
        r"\bVTU\b": "Visvesvaraya Technological University",
        r"\bLMS\b": "Learning Management System",
        r"\bIELTS\b": "IELTS English language exam",
        r"\bAI\b": "Artificial Intelligence",
        r"\bML\b": "Machine Learning",
    }
    
    for pattern, replacement in unambiguous_map.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
    # 2. Context-aware expansion for ambiguous "OS"
    # Only expand "OS" to "Operating Systems" if adjacent to computer science / academic keywords
    if re.search(r"\bOS\b", normalized, flags=re.IGNORECASE):
        query_words = [w.lower() for w in re.findall(r"\b\w+\b", normalized)]
        has_academic_context = any(t in query_words for t in OS_TRIGGERS)
        if has_academic_context:
            normalized = re.sub(r"\bOS\b", "Operating Systems", normalized, flags=re.IGNORECASE)
            
    # 3. Expand common engineering roadmaps
    roadmaps_map = {
        r"\bplacement\s+roadmap\b": "engineering placement preparation roadmap",
        r"\bplacement\s+prep\b": "engineering placement preparation",
        r"\bcareer\s+roadmap\b": "career upskilling roadmap",
        r"\bjob\s+roadmap\b": "job placement preparation roadmap"
    }
    
    for pattern, replacement in roadmaps_map.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
    # Clean redundant whitespaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # 4. Integrate Semantic Query Expansion
    expander = SemanticQueryExpander()
    normalized = expander.expand_query(normalized, active_topic=active_topic, last_courses=last_courses)
    
    return normalized
