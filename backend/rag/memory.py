import time
import re

class ContextMemory:
    """
    Manages persistent student learning attributes (goals, weak subjects, target domains)
    with score decay, confidence classification, and selective relevance retrieval.
    """
    def __init__(self, session_id: str, max_capacity: int = 5):
        self.session_id = session_id
        self.max_capacity = max_capacity
        self.target_domains = {}  # Format: {value: {score, last_used, confidence}}
        self.weak_subjects = {}
        self.learning_goals = {}

    def _enforce_capacity(self, category_dict):
        """
        Enforces maximum capacity limits by removing the lowest-scoring, oldest items.
        """
        if len(category_dict) > self.max_capacity:
            sorted_keys = sorted(
                category_dict.keys(),
                key=lambda k: (category_dict[k]["score"], category_dict[k]["last_used"])
            )
            while len(category_dict) > self.max_capacity:
                del category_dict[sorted_keys.pop(0)]

    def extract_memories(self, query: str, intent: str, expanded_query: str):
        """
        Scans query for explicit student facts. Keeps facts with confidence >= 0.65.
        """
        query_lower = expanded_query.lower()
        now = time.time()
        
        # 1. Target Domain Extraction
        domains_keywords = {
            "ai/ml": ["ai/ml", "artificial intelligence", "machine learning", "deep learning", r"\bai\b", r"\bml\b"],
            "Web Development": ["web dev", "web development", "frontend", "backend", "full stack", "react", "html"],
            "Cybersecurity": ["cybersecurity", "cyber security", "ethical hacking", "network security"],
            "Data Science": ["data science", "data analytics", "data analyst", "pandas", "numpy"]
        }
        
        for domain, keywords in domains_keywords.items():
            match_found = False
            for kw in keywords:
                if kw.startswith(r"\b"):
                    if re.search(kw, query_lower):
                        match_found = True
                        break
                elif kw in query_lower:
                    match_found = True
                    break
                    
            if match_found:
                # High confidence if explicitly mentioned
                self.target_domains[domain] = {
                    "value": domain,
                    "score": 1.0,
                    "last_used": now,
                    "confidence": "High Confidence"
                }

        # 2. Weak Subjects Extraction (Failed, struggling, backlog)
        subjects_keywords = {
            "DBMS": ["dbms", "database", "sql"],
            "DSA": ["dsa", "data structures", "algorithms"],
            "Python": ["python", "py"],
            "Java": ["java"]
        }
        
        if any(k in query_lower for k in ["fail", "failed", "backlog", "struggle", "difficult", "hard"]):
            for sub, keywords in subjects_keywords.items():
                if any(k in query_lower for k in keywords):
                    self.weak_subjects[sub] = {
                        "value": sub,
                        "score": 1.0,
                        "last_used": now,
                        "confidence": "High Confidence"
                    }

        # 3. Learning Goals Extraction
        goals_patterns = [
            (r"internship", "Get an Internship"),
            (r"placement", "Prepare for Placements"),
            (r"certif", "Earn Certifications"),
            (r"pass", "Pass Exams")
        ]
        
        for pattern, goal_label in goals_patterns:
            if re.search(pattern, query_lower):
                # Medium confidence for inferred goals
                self.learning_goals[goal_label] = {
                    "value": goal_label,
                    "score": 0.8,
                    "last_used": now,
                    "confidence": "Medium Confidence"
                }

        # Enforce capacity limits across memory categories
        self._enforce_capacity(self.target_domains)
        self._enforce_capacity(self.weak_subjects)
        self._enforce_capacity(self.learning_goals)

        # Automatically decay scores for all stored memory attributes
        self.decay_memories()

    def decay_memories(self):
        """
        Decays memory scores by 0.1 per turn. Removes stale memories below score 0.3.
        """
        decay_factor = 0.1
        minimum_score = 0.3
        
        for category in [self.target_domains, self.weak_subjects, self.learning_goals]:
            to_remove = []
            for key, attr in category.items():
                attr["score"] -= decay_factor
                if attr["score"] < minimum_score:
                    to_remove.append(key)
            for key in to_remove:
                del category[key]

    def retrieve_relevant_memories(self, query: str) -> str:
        """
        Implements relevance-based prompt injection. Retrieves only matching
        academic memories related to the active query keywords.
        """
        query_lower = query.lower()
        matched_facts = []
        
        # Look for domain overlaps
        for domain, attr in self.target_domains.items():
            if domain.lower() in query_lower or any(kw in query_lower for kw in ["job", "career", "placement", "internship"]):
                matched_facts.append(f"Student is targeting the domain: {domain} ({attr['confidence']}).")
                attr["last_used"] = time.time()
                attr["score"] = min(1.0, attr["score"] + 0.1) # Boost score if referenced
                
        # Look for weak subject overlaps
        for sub, attr in self.weak_subjects.items():
            if sub.lower() in query_lower or any(kw in query_lower for kw in ["failed", "exams", "quizzes", "backlog"]):
                matched_facts.append(f"Student has a weak subject in: {sub} ({attr['confidence']}).")
                attr["last_used"] = time.time()
                attr["score"] = min(1.0, attr["score"] + 0.1)
                
        # Look for goal overlaps
        for goal, attr in self.learning_goals.items():
            if any(w in query_lower for w in goal.lower().split()):
                matched_facts.append(f"Student's current goal is: {goal} ({attr['confidence']}).")
                attr["last_used"] = time.time()
                attr["score"] = min(1.0, attr["score"] + 0.1)

        if not matched_facts:
            return ""
            
        return "\nActive Student Profile Memory:\n" + "\n".join(f"- {fact}" for fact in matched_facts)

    def get_visual_profile(self) -> dict:
        """
        Returns highest-scoring active memories formatted for Visual Sidebar Widget.
        """
        # Find highest-scoring target domain, weak subject, and learning goal
        domain = max(self.target_domains.values(), key=lambda x: x["score"]) if self.target_domains else None
        subject = max(self.weak_subjects.values(), key=lambda x: x["score"]) if self.weak_subjects else None
        goal = max(self.learning_goals.values(), key=lambda x: x["score"]) if self.learning_goals else None
        
        return {
            "target_domain": {
                "value": domain["value"],
                "confidence": domain["confidence"]
            } if domain else None,
            "weak_subject": {
                "value": subject["value"],
                "confidence": subject["confidence"]
            } if subject else None,
            "learning_goal": {
                "value": goal["value"],
                "confidence": goal["confidence"]
            } if goal else None
        }
