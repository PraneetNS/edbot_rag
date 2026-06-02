import re

REJECTION_RESPONSE = (
    "I am EduMentor, your dedicated AI Engineering Academic Mentor. I can only assist you with questions "
    "related to engineering courses, core CS concepts (DSA, OOP, SQL), placements, internships, VTU certifications, "
    "and LMS support.\n\n"
    "Please let me know how I can help you with your academic or career development in engineering!"
)

def is_educational_query(query: str) -> bool:
    query_lower = query.lower().strip()
    
    # Immediately block unsafe or explicitly out-of-scope categories
    safety_blocklist = [
        r"\bhack(ing|er|s)?\b", r"\bexploits?\b", r"\bbypass\s+security\b", r"\bpolitics?\b", 
        r"\belections?\b", r"\bspam\b", r"\brecipes?\b", r"\bmovies?\b", r"\bgames?\b", 
        r"\bbake\s+a\s+cake\b", r"\btell\s+a\s+joke\b", r"\bweather\b"
    ]
    for pattern in safety_blocklist:
        if re.search(pattern, query_lower):
            return False
            
    educational_keywords = [
        "course", "placement", "internship", "certif", "lms", "support", "help",
        "contact", "learn", "study", "exam", "admission", "fee", "price", "duration",
        "react", "python", "java", "coding", "partner", "vtu", "job", "career",
        "syllabus", "curriculum", "schedule", "class", "project", "assignment", "bot", 
        "hello", "hi", "hey", "edutainer", "edubot", "who are you", "what is your name", "help",
        "what", "how", "why", "who", "where", "when", "can", "explain", "tell", "describe",
        "programming", "code", "developer", "development", "software", "database", "sql",
        "computer", "science", "engineering", "math", "algorithm", "data", "structure",
        "learn", "study", "teach", "explain", "tutorial", "guide", "concept", "javascript",
        "c++", "c#", "ruby", "rust", "go", "php", "web", "html", "css", "git", "github",
        "frontend", "backend", "fullstack", "technology", "network", "security",
        "2nd year", "second year", "3rd year", "third year", "4th year", "fourth year",
        "final year", "sophomore", "junior", "senior", "capstone", "dsa", "oop", 
        "system design", "resume", "interview", "mentor", "roadmaps", "placement prep"
    ]
    
    if any(keyword in query_lower for keyword in educational_keywords):
        return True
    return False
