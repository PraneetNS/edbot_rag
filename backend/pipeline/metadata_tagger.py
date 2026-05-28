import sys
import re
import hashlib
from pathlib import Path

# Add parent path to import config
sys.path.append(str(Path(__file__).resolve().parent))
import config

class MetadataTagger:
    """
    Standardized Educational Metadata Tagging Engine that automatically infers
    category, difficulty, domain, intent, tags, and summary, producing the exact required 11-key JSON schema.
    """
    def __init__(self):
        pass

    def _generate_id(self, content: str) -> str:
        """
        Creates a unique hash ID based on the chunk content.
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _generate_summary(self, content: str) -> str:
        """
        Generates a concise, informative educational summary of the chunk.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        if not sentences:
            return "No summary available."
            
        # Select first 2 sentences, ensuring size constraints
        summary = " ".join(sentences[:2])
        if len(summary) > 200:
            summary = summary[:197] + "..."
        return summary

    def _infer_difficulty(self, content: str) -> str:
        """
        Infers subject matter difficulty based on language/content cues.
        """
        content_lower = content.lower()
        if any(w in content_lower for w in ["advanced", "expert", "deep learning", "optimization", "complex", "dynamic programming"]):
            return "advanced"
        elif any(w in content_lower for w in ["intermediate", "react js", "classes", "loops", "functions", "sql joins"]):
            return "intermediate"
        else:
            return "beginner"

    def _infer_intent(self, content: str, default_intent: str = "COURSE_QUERY") -> str:
        """
        Dynamically classifies query intent of the chunk based on terms.
        """
        content_lower = content.lower()
        if any(w in content_lower for w in ["placement", "resume", "interview", "hire", "recruit"]):
            return "PLACEMENT_GUIDANCE"
        elif any(w in content_lower for w in ["internship", "project guidelines"]):
            return "INTERNSHIP_GUIDANCE"
        elif any(w in content_lower for w in ["password", "login", "portal", "account", "issue"]):
            return "LMS_SUPPORT"
        elif any(w in content_lower for w in ["certificate", "verified credentials", "vtu"]):
            return "CERTIFICATION_SUPPORT"
        elif any(w in content_lower for w in ["exam", "test", "quiz", "failed", "grades"]):
            return "EXAM_ASSISTANCE"
        elif any(w in content_lower for w in ["assignment", "homework", "project"]):
            return "ASSIGNMENT_HELP"
        return default_intent

    def _infer_domain(self, content: str) -> str:
        """
        Infers academic domain.
        """
        content_lower = content.lower()
        if any(w in content_lower for w in ["react", "web", "html", "javascript", "css"]):
            return "Web Development"
        elif any(w in content_lower for w in ["ai", "artificial intelligence", "machine learning", "deep learning", "nlp"]):
            return "Artificial Intelligence"
        elif any(w in content_lower for w in ["dsa", "data structures", "algorithms"]):
            return "DSA"
        elif any(w in content_lower for w in ["dbms", "database", "sql"]):
            return "DBMS"
        elif "vtu" in content_lower or "certificate" in content_lower:
            return "LMS certifications"
        return "General Technology"

    def _extract_tags(self, content: str, domain: str) -> list[str]:
        """
        Extracts key tags from the content.
        """
        words = re.findall(r'\b\w{4,}\b', content.lower())
        candidates = ["roadmap", "syllabus", "interview", "resume", "prep", "support", "certification", "vtucert"]
        
        tags = [domain.lower()]
        for c in candidates:
            if c in content.lower():
                tags.append(c)
                
        # Limit to 5 unique tags
        return list(set(tags))[:5]

    def tag_chunk(self, content: str, source: str, category: str) -> dict:
        """
        Structures the raw chunk content into the exact 11-key standard educational RAG schema.
        """
        chunk_id = self._generate_id(content)
        summary = self._generate_summary(content)
        difficulty = self._infer_difficulty(content)
        intent = self._infer_intent(content)
        domain = self._infer_domain(content)
        tags = self._extract_tags(content, domain)
        
        # Determine topic from title or first heading
        topic = "General upskilling"
        first_line = content.strip().split("\n")[0]
        if len(first_line) < 100:
            topic = re.sub(r'[#*]', '', first_line).strip()

        # Build exactly 11-key schema
        schema = {
            "id": chunk_id,
            "source": source,
            "category": category,
            "domain": domain,
            "difficulty": difficulty,
            "intent": intent,
            "topic": topic,
            "tags": tags,
            "content": content,
            "summary": summary,
            "embedding_ready": True
        }
        
        return schema

if __name__ == "__main__":
    tagger = MetadataTagger()
    sample_content = "# React JS Course Syllabus\nMaster modern hooks, states, and components to build powerful single page applications online."
    chunk_json = tagger.tag_chunk(sample_content, "courses.txt", "courses")
    import pprint
    pprint.pprint(chunk_json)
    print(f"Key count: {len(chunk_json)}") # Should print 11
