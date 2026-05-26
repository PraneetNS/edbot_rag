import time

class ConversationState:
    """
    Manages dynamic topic tracking, academic workflows, longitudinal goals,
    and active mentor modes for a student's session.
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.active_topic = "General Academics"
        self.active_intent = "COURSE_QUERY"
        self.workflow = "general_mentoring"
        self.mode = "academic_mentor"  # Modes: academic_mentor, placement_coach, support_assistant
        self.last_courses_discussed = []
        self.active_goal = None
        self.last_updated = time.time()

    def update_state(self, query: str, intent: str, expanded_query: str):
        """
        Dynamically transitions conversation state based on intent and query contents.
        """
        self.last_updated = time.time()
        self.active_intent = intent
        
        # 1. Topic Extraction & Course History
        query_lower = expanded_query.lower()
        
        course_keywords = {
            "react": "React JS",
            "python": "Python",
            "java": "Java",
            "dbms": "DBMS",
            "database management systems": "DBMS",
            "database": "DBMS",
            "dsa": "DSA",
            "data structures and algorithms": "DSA",
            "data structures": "DSA",
            "ielts": "IELTS English",
            "artificial intelligence": "AI",
            "ai": "AI",
            "interior design": "Interior Design",
            "midjourney": "Midjourney Mastery",
            "growth marketing": "Growth Marketing",
            "financial planning": "Financial Planning",
            "stock trading": "Stock Trading",
            "cryptocurrency": "Cryptocurrency"
        }
        
        for kw, course_name in course_keywords.items():
            if kw in query_lower:
                self.active_topic = course_name
                if course_name not in self.last_courses_discussed:
                    self.last_courses_discussed.append(course_name)
                    # Limit history size to prevent overflow
                    if len(self.last_courses_discussed) > 4:
                        self.last_courses_discussed.pop(0)
                break
                
        # 2. Mentor Mode & Workflow Transition Matrix
        if intent in ["PLACEMENT_GUIDANCE", "INTERNSHIP_GUIDANCE"]:
            self.mode = "placement_coach"
            self.workflow = "placement_preparation"
            
            # Goal extraction
            if "ai" in query_lower or "artificial intelligence" in query_lower or "machine learning" in query_lower:
                self.active_goal = "Prepare for AI placements"
            elif "web" in query_lower or "frontend" in query_lower:
                self.active_goal = "Prepare for Web Development placements"
            elif self.active_goal is None:
                self.active_goal = "General Career Ingestion"
                
        elif intent in ["LMS_SUPPORT", "CERTIFICATION_SUPPORT"]:
            self.mode = "support_assistant"
            self.workflow = "portal_technical_support"
            
            if "certificate" in query_lower or "vtu" in query_lower:
                self.active_goal = "Download verified VTU credentials"
                
        elif intent in ["EXAM_ASSISTANCE", "ASSIGNMENT_HELP"]:
            self.mode = "academic_mentor"
            
            # Subject recovery workflow triggered if student failed
            if any(k in query_lower for k in ["fail", "failed", "re-exam", "backlog"]):
                self.workflow = "subject_recovery_guidance"
                if self.active_topic != "General Academics":
                    self.active_goal = f"Recover & Pass {self.active_topic}"
            else:
                self.workflow = "assignment_coaching"
                if self.active_topic != "General Academics" and self.active_goal is None:
                    self.active_goal = f"Master {self.active_topic}"
        else:
            # Maintain active mode, default workflow
            self.workflow = "general_mentoring"

    def to_dict(self) -> dict:
        """
        Serialize state for backend responses and frontend API visualizers.
        """
        return {
            "session_id": self.session_id,
            "active_topic": self.active_topic,
            "active_intent": self.active_intent,
            "workflow": self.workflow,
            "mode": self.mode,
            "last_courses_discussed": self.last_courses_discussed,
            "active_goal": self.active_goal,
            "last_updated": self.last_updated
        }
