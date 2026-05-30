import re

# Precise keywords and regex mappings for rule-based fast classification
INTENT_RULES = {
    "PLACEMENT_GUIDANCE": [
        r"\bplacements?\b", r"\bjobs?\b", r"\bcareers?\b", r"\binterviews?\b", 
        r"\bresumes?\b", r"\brecruits?\b", r"\bhiring\b", r"\binternships?\b", r"\bemployability\b",
        r"\bplacement prep(aration)?\b", r"\bcareer guidance\b", r"\binterview prep(aration)?\b", r"\broadmaps?\b"
    ],
    "LMS_SUPPORT": [
        r"\blms\b", r"\belms\b", r"\bportals?\b", r"\blogins?\b", r"\bpasswords?\b", 
        r"\baccounts?\b", r"\bsubmissions?\b", r"\bvideo\s+not\s+loading\b", r"\bloading\s+issue\b"
    ],
    "CERTIFICATION_SUPPORT": [
        r"\bcertificates?\b", r"\bcertifications?\b", r"\bvtu\s+stamp\b", r"\bbadges?\b", 
        r"\bdownload\s+certificate\b", r"\bverify\s+certificate\b"
    ],
    "EXAM_ASSISTANCE": [
        r"\bexams?\b", r"\btests?\b", r"\bquizzes?\b", r"\bfailed?\b", r"\bre-exam\b", 
        r"\bstudy\s+tips\b", r"\bgrades?\b", r"\bpass\s+criteria\b"
    ],
    "ASSIGNMENT_HELP": [
        r"\bassignments?\b", r"\bhomeworks?\b", r"\bprojects?\b", r"\bcapstones?\b"
    ],
    "COURSE_QUERY": [
        r"\bcourses?\b", r"\bsyllabus\b", r"\bsyllabi\b", r"\bcurriculum\b", r"\breact\b", 
        r"\bpython\b", r"\bjava\b", r"\bclasses?\b", r"\blearn\b", r"\bdsa\b", r"\balgorithms?\b",
        r"\bdata structures?\b", r"\boop\b", r"\bsystem design\b", r"\bsecond year\b", r"\b2nd year\b",
        r"\bthird year\b", r"\b3rd year\b", r"\bfourth year\b", r"\b4th year\b", r"\bfinal year\b",
        r"\bsophomores?\b", r"\bjuniors?\b", r"\bseniors?\b", r"\bmentorship\b", r"\bmentor\b"
    ]
}

# Educational safety blocklist for off-scope queries
SAFETY_BLOCKLIST = [
    r"\bhack(ing|er|s)?\b", r"\bexploits?\b", r"\bbypass\s+security\b", r"\bpolitics?\b", 
    r"\belections?\b", r"\bspam\b", r"\brecipes?\b", r"\bmovies?\b", r"\bgames?\b", 
    r"\bbake\s+a\s+cake\b", r"\btell\s+a\s+joke\b", r"\bweather\b"
]

def classify_intent(query: str, llm=None) -> tuple[str, float]:
    """
    Classifies student query intents using a hybrid fast check + semantic fallback.
    Returns: (intent_label, confidence_score)
    """
    if not query:
        return "COURSE_QUERY", 1.0
        
    query_clean = query.strip()
    query_lower = query_clean.lower()
    
    # 0. Fast check for simple greetings/banter so they are classified as COURSE_QUERY (safe, in-scope)
    q_simple = query_lower.rstrip("?!.")
    if q_simple in ["hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening", "how are you", "who are you", "what's your name", "tell me about yourself"]:
        return "COURSE_QUERY", 1.0
        
    # 1. Strict Educational Safety & Guardrail Check (OUT_OF_SCOPE)
    for pattern in SAFETY_BLOCKLIST:
        if re.search(pattern, query_lower):
            return "OUT_OF_SCOPE", 1.0
            
    # 2. Fast Rule-Based Regex Classification
    # Score each category based on matches
    scores = {intent: 0 for intent in INTENT_RULES}
    for intent, patterns in INTENT_RULES.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                scores[intent] += 1
                
    # Find intent with the highest keyword score
    max_intent = max(scores, key=scores.get)
    if scores[max_intent] > 0:
        return max_intent, 1.0
        
    # 3. LLM Fallback (If LLM is provided and online)
    if llm is not None:
        try:
            prompt = f"""Analyze the student query below and classify it into EXACTLY one of these intent categories:
- COURSE_QUERY (general syllabus, class, or learning details)
- LMS_SUPPORT (login, LMS portals, password, video issues)
- PLACEMENT_GUIDANCE (resumes, interviews, job roadmaps)
- ASSIGNMENT_HELP (assignments, homework, capstone projects)
- EXAM_ASSISTANCE (exams, quizzes, grades, test preparations)
- CERTIFICATION_SUPPORT (credentials, certificate download, VTU stamps)
- OUT_OF_SCOPE (recipes, jokes, security hacks, general off-topic inquiries)

Rules:
- Respond with ONLY the category name. Do not explain your choice.

Student Query: "{query_clean}"

Category:"""
            # Timeout set aggressively to keep intent routing fast
            response = str(llm.complete(prompt)).strip()
            
            # Match LLM response to valid labels
            valid_labels = {
                "COURSE_QUERY", "LMS_SUPPORT", "PLACEMENT_GUIDANCE", 
                "ASSIGNMENT_HELP", "EXAM_ASSISTANCE", "CERTIFICATION_SUPPORT", "OUT_OF_SCOPE"
            }
            for label in valid_labels:
                if label.lower() in response.lower():
                    # Confidence is high if LLM matches exactly
                    return label, 0.85
        except Exception as e:
            print(f"Error during LLM intent classification: {e}")
            
    # 4. Local Word-Weighted Offline Fallback (If LLM offline or fails)
    # Simple semantic heuristics using keyword groups
    if "failed" in query_lower or "exam" in query_lower or "quiz" in query_lower:
        return "EXAM_ASSISTANCE", 0.75
    elif "placement" in query_lower or "resume" in query_lower or "internship" in query_lower:
        return "PLACEMENT_GUIDANCE", 0.75
    elif "certificate" in query_lower or "vtu" in query_lower or "stamp" in query_lower:
        return "CERTIFICATION_SUPPORT", 0.75
    elif "login" in query_lower or "portal" in query_lower or "password" in query_lower:
        return "LMS_SUPPORT", 0.75
    elif "assignment" in query_lower or "project" in query_lower:
        return "ASSIGNMENT_HELP", 0.75
        
    # Standard Default fallback category
    return "COURSE_QUERY", 0.60
