import sys
import json
from pathlib import Path

# Add parent path to import config
sys.path.append(str(Path(__file__).resolve().parent))
import config

CONVERSATIONAL_DATASETS = [
    {
        "conversation_type": "placement_guidance",
        "student_query": "I am weak in DSA. How do I prepare for upcoming placements?",
        "mentor_response": "Let's identify whether arrays, recursion, stack, queue, or dynamic programming is causing difficulty. I recommend practicing standard problems on arrays and hashing first, then advancing step-by-step into trees and graphs. We can set up a personalized 6-week technical schedule for you.",
        "difficulty": "beginner",
        "domain": "DSA",
        "intent": "PLACEMENT_GUIDANCE",
        "topic": "Data Structures and Algorithms",
        "tags": ["DSA", "placements", "career prep", "study plan"]
    },
    {
        "conversation_type": "weak_subject_mentoring",
        "student_query": "I failed my DBMS midterms and struggle with SQL Joins.",
        "mentor_response": "SQL Joins can be tricky, but they are highly visual. Think of them as Venn Diagrams: INNER JOIN gets overlapping records, LEFT JOIN gets all left table records plus matches, and RIGHT JOIN is the opposite. Let's do a 15-minute hands-on practice session on joint queries to secure your backlog recovery.",
        "difficulty": "intermediate",
        "domain": "DBMS",
        "intent": "EXAM_ASSISTANCE",
        "topic": "Database Management Systems",
        "tags": ["DBMS", "SQL Joins", "academic recovery", "tutoring"]
    },
    {
        "conversation_type": "certification_guidance",
        "student_query": "How do I download my verified VTU virtual internship certificate?",
        "mentor_response": "To download your verified VTU certificate, log into the Edutainer dashboard, navigate to the 'My Certifications' widget, verify that your virtual internship hours are fully approved, and click the blue 'Download PDF' button. If a VTU stamp is missing, notify support@edutainer.in.",
        "difficulty": "beginner",
        "domain": "LMS support",
        "intent": "CERTIFICATION_SUPPORT",
        "topic": "VTU Certifications",
        "tags": ["VTU", "certificate", "LMS support", "downloads"]
    },
    {
        "conversation_type": "semester_planning",
        "student_query": "I want to balance learning React JS and preparing for my web dev internship.",
        "mentor_response": "That is a great target. Allocate 4 hours a week to learning modern React hooks and components on the Edutainer LMS, and 2 hours building a simple frontend project (such as a dashboard or portfolio). Project building is highly rated by internship recruiters.",
        "difficulty": "intermediate",
        "domain": "Web Development",
        "intent": "INTERNSHIP_GUIDANCE",
        "topic": "React JS Web Development",
        "tags": ["React", "web dev", "semester planning", "internships"]
    }
]

WORKFLOW_DATASETS = [
    {
        "intent": "certificate_issue",
        "examples": [
            "Certificate not visible",
            "I didn't receive my VTU internship certificate",
            "Where is my verified credentials badge?"
        ],
        "workflow": [
            "Verify all course modules are marked as 100% completed",
            "Check that final assessments are graded and passed successfully",
            "Click 'Refresh Dashboard' in your profile settings",
            "If still missing, escalate the request to support@edutainer.in with your user ID"
        ],
        "category": "LMS support",
        "domain": "LMS workflow",
        "difficulty": "beginner",
        "topic": "Certificate Troubleshooting",
        "tags": ["workflows", "LMS", "credentials", "support"]
    },
    {
        "intent": "lms_login_issue",
        "examples": [
            "LMS login failing",
            "Cannot access my course dashboard",
            "Password reset link not working"
        ],
        "workflow": [
            "Clear browser cookies and local storage",
            "Navigate to the login portal and click 'Forgot Password'",
            "Check spam/junk folder for the verified reset link",
            "Contact technical team support@edutainer.in if account remains locked out"
        ],
        "category": "LMS support",
        "domain": "LMS workflow",
        "difficulty": "beginner",
        "topic": "LMS Portal Authentication",
        "tags": ["workflows", "LMS", "login", "password reset"]
    }
]

def generate_synthetic_data():
    """
    Generates and structures conversational and workflow JSON datasets, saving them in the data/ directories.
    """
    print("Generating EduBot synthetic educational intelligence datasets...")
    
    # 1. Save Conversational Dialogues
    dialogues_path = config.DATA_DIR / "synthetic_dialogues" / "conversations.json"
    with open(dialogues_path, "w", encoding="utf-8") as f:
        json.dump(CONVERSATIONAL_DATASETS, f, indent=4)
    print(f"Structured conversational dataset saved to: {dialogues_path}")
    
    # 2. Save LMS Workflows
    workflows_path = config.DATA_DIR / "workflows" / "lms_workflows.json"
    with open(workflows_path, "w", encoding="utf-8") as f:
        json.dump(WORKFLOW_DATASETS, f, indent=4)
    print(f"Structured LMS workflow dataset saved to: {workflows_path}")

if __name__ == "__main__":
    generate_synthetic_data()
