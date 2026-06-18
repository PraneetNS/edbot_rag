import os
import sys
from pathlib import Path
import unittest
from fastapi.testclient import TestClient

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

# Mock environment variables or offline checks if needed
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-purposes"
os.environ["JWT_EXPIRE_HOURS"] = "2"

# Import app and helper functions after path is adjusted
from api import app, is_educational_query, REJECTION_RESPONSE

class TestEduBotAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "ok")

    def test_state_endpoint_default(self):
        response = self.client.get("/state/test_session_123")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["session_id"], "test_session_123")
        self.assertEqual(data["active_topic"], "general")

    def test_is_educational_query_validation(self):
        # Academic queries should return True
        self.assertTrue(is_educational_query("What is the React JS syllabus?"))
        self.assertTrue(is_educational_query("Can you help me prepare for placements?"))
        # Non-academic/unsafe queries should return False
        self.assertFalse(is_educational_query("How to hack a computer"))
        self.assertFalse(is_educational_query("How to bake a cake"))

    def test_query_guardrail_rejection(self):
        response = self.client.post(
            "/query",
            json={"question": "How to bake a cake", "session_id": "test_session"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["active_intent"], "OUT_OF_SCOPE")
        self.assertEqual(data["response"], REJECTION_RESPONSE)

    def test_query_empty_validation(self):
        response = self.client.post(
            "/query",
            json={"question": "", "session_id": "test_session"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["active_intent"], "INVALID_QUERY")
        self.assertIn("cannot be empty", data["response"])

if __name__ == "__main__":
    unittest.main()
