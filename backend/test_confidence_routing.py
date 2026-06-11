import os
import sys
import asyncio
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Resolve path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

# Ensure offline modes for HF hub
os.environ["HF_HUB_OFFLINE"] = "1"

# Import components
from edmentor.intent_router import is_off_domain
from edmentor.safety_filter import edumentor_filter
from edmentor.confidence_router import generate_response_with_routing
from edmentor.rag_engine import rag_retrieve_and_respond

class TestEdmentorRoutingPipeline(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Setup session id
        self.session_id = "test_session"

    def test_intent_router(self):
        print("\n--- Testing Intent Router ---")
        # On-domain queries should return False (not off-domain)
        self.assertFalse(is_off_domain("how do I prepare for engineering placements"))
        self.assertFalse(is_off_domain("explain dynamic programming concept"))
        self.assertFalse(is_off_domain("tell me about internships"))

        # Off-domain queries should return True (is off-domain)
        self.assertTrue(is_off_domain("how do I bake a chocolate cake"))
        self.assertTrue(is_off_domain("what is the weather in New York"))
        self.assertTrue(is_off_domain("who won the football match today"))

    def test_safety_filter_cleanup(self):
        print("\n--- Testing Safety Filter Cleanup ---")
        # Test markdown stripping and filler removal
        raw_text = "**Yes**, absolutely! Here is the thing: dynamic programming is easy. Certainly, as an AI, I can help."
        cleaned = edumentor_filter(raw_text)
        print(f"Raw: '{raw_text}'")
        print(f"Cleaned: '{cleaned}'")
        
        # Verify markdown bold stripped
        self.assertNotIn("**", cleaned)
        # Verify fillers removed
        self.assertNotIn("absolutely", cleaned.lower())
        self.assertNotIn("certainly", cleaned.lower())
        self.assertNotIn("as an ai", cleaned.lower())

    def test_safety_filter_sentence_boundary_truncation(self):
        print("\n--- Testing Safety Filter Sentence Boundary Truncation ---")
        # Construct a response with exactly 80 words, split across 3 sentences.
        # Sentence 1: 30 words. Sentence 2: 35 words (Total = 65 words). Sentence 3: 15 words (Total = 80 words).
        # It should cut exactly after Sentence 2 (at 65 words) since Sentence 3 would exceed 75 words.
        s1 = " ".join(["word"] * 30) + "."
        s2 = " ".join(["cool"] * 35) + "."
        s3 = " ".join(["last"] * 15) + "."
        
        full_text = f"{s1} {s2} {s3}"
        self.assertEqual(len(full_text.split()), 80)
        
        trimmed = edumentor_filter(full_text, max_words=75)
        trimmed_words = trimmed.split()
        
        print(f"Original word count: 80")
        print(f"Trimmed word count: {len(trimmed_words)}")
        print(f"Trimmed text ends with: '{' '.join(trimmed_words[-5:])}'")
        
        # Trimmed word count should be 65
        self.assertEqual(len(trimmed_words), 65)
        self.assertTrue(trimmed.endswith("cool."))
        self.assertNotIn("last", trimmed)

    async def test_first_turn_greeting(self):
        print("\n--- Testing First Turn Greeting Behavior ---")
        for q in ["hello", "hi", "hey", "Sup?", "dsa"]:
            response, routing_mode = await generate_response_with_routing(q, self.session_id)
            print(f"Query: '{q}' -> Response: '{response}' | Mode: '{routing_mode}'")
            self.assertEqual(routing_mode, "first_turn_greeting")
            self.assertIn("Tell me what you are working on", response)

    @patch('edmentor.rag_engine.get_chroma_resources')
    @patch('edmentor.qwen_client.qwen_client.is_available')
    @patch('edmentor.qwen_client.qwen_client.generate')
    async def test_rag_direct_response(self, mock_generate, mock_is_available, mock_get_chroma):
        print("\n--- Testing Direct RAG Retrieval Response ---")
        mock_is_available.return_value = True
        mock_generate.return_value = "Start with a base case, then write the recursive call."
        # Mock ChromaDB query return value
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["Student: explain recursion\nMentor: Start with a base case, then write the recursive call."]],
            "metadatas": [[{"section": "dsa_concepts", "id": "dsa_001"}]],
            "distances": [[0.1]]
        }
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [0.1] * 384
        mock_get_chroma.return_value = (mock_col, mock_embedder)

        response, routing_mode = await generate_response_with_routing("explain recursion", self.session_id)
        print(f"Response: '{response}'")
        print(f"Routing Mode: '{routing_mode}'")

        self.assertEqual(routing_mode, "rag_direct")
        self.assertEqual(response, "Start with a base case, then write the recursive call.")

if __name__ == "__main__":
    unittest.main()
