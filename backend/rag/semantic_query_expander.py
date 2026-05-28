import re

EDUCATIONAL_TAXONOMY = {
    "engineering_courses": ["AI", "Artificial Intelligence", "ML", "Machine Learning", "Python", "React", "React JS", "Programming", "Software Development", "Technical Courses", "coding", "web development"],
    "placement_prep": ["resume mentoring", "interview preparation", "DSA", "Data Structures", "Algorithms", "internships", "career guidance", "employability", "jobs", "hiring"],
    "ai_ml": ["Machine Learning", "Deep Learning", "Data Science", "Generative AI", "NLP", "Artificial Intelligence", "Neural Networks", "ChatGPT", "Prompt Engineering", "Midjourney"],
    "courses_general": ["React JS", "Artificial Intelligence", "Python", "Interior Design", "LMS", "syllabus", "programming", "classes", "training"],
    "certifications": ["VTU certification", "credentials", "badges", "certificate verification", "exam passing requirements", "diploma"]
}

class SemanticQueryExpander:
    """
    Expands student queries semantically using an educational taxonomy ontology
    and conversational active topic memory.
    """
    def __init__(self):
        pass

    def expand_query(self, query: str, active_topic: str = None, last_courses: list = None) -> str:
        """
        Enriches a query with relevant academic synonyms and conversational topic memory.
        """
        if not query:
            return ""

        query_lower = query.lower()
        expanded_terms = []

        # 1. Match against educational ontology & taxonomy
        if any(w in query_lower for w in ["engg", "engineering", "technical", "coding", "programming", "software"]):
            expanded_terms.extend(EDUCATIONAL_TAXONOMY["engineering_courses"])
            
        if any(w in query_lower for w in ["placement", "job", "career", "interview", "resume", "recruit", "internship"]):
            expanded_terms.extend(EDUCATIONAL_TAXONOMY["placement_prep"])
            
        if any(w in query_lower for w in ["ai", "artificial intelligence", "ml", "machine learning", "deep learning"]):
            expanded_terms.extend(EDUCATIONAL_TAXONOMY["ai_ml"])
            
        if any(w in query_lower for w in ["course", "syllabus", "curriculum", "program", "class"]):
            expanded_terms.extend(EDUCATIONAL_TAXONOMY["courses_general"])
            
        if any(w in query_lower for w in ["certif", "vtu", "stamp", "badge", "credentials"]):
            expanded_terms.extend(EDUCATIONAL_TAXONOMY["certifications"])

        # 2. Conversational Semantic Memory Injection
        # If the user previously discussed a specific topic/course, and is now asking a generic or follow-up query,
        # inject the active topic / previously discussed courses to keep the context active!
        follow_up_triggers = ["after", "then", "next", "later", "before", "subsequent", "follow up", "now what", "what about", "what to", "where to", "project", "projects"]
        is_generic_query = any(w in query_lower for w in ["course", "courses", "syllabus", "placement", "internship", "certif"])
        is_follow_up = any(re.search(r'\b' + re.escape(w) + r'\b', query_lower) for w in follow_up_triggers)
        
        if is_generic_query or is_follow_up:
            if active_topic and active_topic.lower() not in ["general academics", "general", "general chat"]:
                expanded_terms.append(active_topic)
            if last_courses:
                for c in last_courses:
                    if c.lower() not in ["general academics", "general"]:
                        expanded_terms.append(c)

        # De-duplicate expanded terms while maintaining order
        seen = set()
        unique_expanded = []
        for term in expanded_terms:
            t_lower = term.lower()
            if t_lower not in seen and t_lower not in query_lower:
                seen.add(t_lower)
                unique_expanded.append(term)

        # Build final enriched query
        if unique_expanded:
            enriched_query = f"{query} {' '.join(unique_expanded)}"
        else:
            enriched_query = query

        # Normalize spacing
        return re.sub(r'\s+', ' ', enriched_query).strip()
