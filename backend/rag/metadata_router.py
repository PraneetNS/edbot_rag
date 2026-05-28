class MetadataRouter:
    """
    Reranks and filters retrieved chunks by matching document metadata
    (like source file names or categories) to the active query intent.
    """
    def __init__(self):
        pass

    def route_and_filter(self, hits: list, intent: str) -> list:
        """
        Prioritizes chunks matching the active intent category, applying routing logic.
        """
        if not hits:
            return []

        intent_mappings = {
            "PLACEMENT_GUIDANCE": ["homepage.txt", "about.txt", "courses.txt"],
            "INTERNSHIP_GUIDANCE": ["homepage.txt", "about.txt", "courses.txt"],
            "LMS_SUPPORT": ["support.txt", "faq.txt", "homepage.txt"],
            "CERTIFICATION_SUPPORT": ["support.txt", "faq.txt", "courses.txt"],
            "COURSE_QUERY": ["courses.txt", "faq.txt", "homepage.txt"],
            "EXAM_ASSISTANCE": ["courses.txt", "faq.txt"],
            "ASSIGNMENT_HELP": ["courses.txt", "faq.txt"]
        }

        prioritized_files = intent_mappings.get(intent, [])
        if not prioritized_files:
            return hits

        routed_hits = []
        for h in hits:
            # Get file name from metadata
            metadata = h.node.metadata if hasattr(h, "node") else {}
            file_name = metadata.get("file_name", "").lower()
            
            # Extract simple name (e.g. support.txt from path/to/support.txt)
            import os
            simple_name = os.path.basename(file_name).lower()
            
            score_boost = 0.0
            # If Simple name matches prioritized list, apply score boost
            if simple_name in [f.lower() for f in prioritized_files]:
                score_boost = 0.15 # Up to 0.15 boost
                
            # Create a copy or update score
            h.score = (h.score or 0.0) + score_boost
            routed_hits.append(h)

        # Re-sort based on newly boosted routing scores
        routed_hits.sort(key=lambda x: x.score, reverse=True)
        return routed_hits
