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
from edmentor.confidence_router import generate_response_with_routing, USE_LOCAL_MODEL
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
        # Construct a response with exactly 260 words, split across 3 sentences.
        # Sentence 1: 100 words. Sentence 2: 120 words (Total = 220 words). Sentence 3: 40 words (Total = 260 words).
        # It should cut exactly after Sentence 2 (at 220 words) since Sentence 3 would exceed 250 words.
        s1 = " ".join(["word"] * 100) + "."
        s2 = " ".join(["cool"] * 120) + "."
        s3 = " ".join(["last"] * 40) + "."
        
        full_text = f"{s1} {s2} {s3}"
        self.assertEqual(len(full_text.split()), 260)
        
        trimmed = edumentor_filter(full_text)
        trimmed_words = trimmed.split()
        
        print(f"Original word count: 260")
        print(f"Trimmed word count: {len(trimmed_words)}")
        print(f"Trimmed text ends with: '{' '.join(trimmed_words[-5:])}'")
        
        # Trimmed word count should be 220
        self.assertEqual(len(trimmed_words), 220)
        self.assertTrue(trimmed.endswith("cool."))
        self.assertNotIn("last", trimmed)

    @patch('edmentor.groq_client.llm.chat')
    async def test_groq_interim_mode_direct(self, mock_chat):
        print("\n--- Testing Groq Interim Mode (Direct Response) ---")
        # Force USE_LOCAL_MODEL = False
        with patch('edmentor.confidence_router.USE_LOCAL_MODEL', False):
            # Mock Groq to return a valid mentoring response
            mock_chat.return_value = "Keep practicing recursion first before moving to dynamic programming."
            
            response, routing_mode = await generate_response_with_routing("explain dynamic programming", self.session_id)
            print(f"Response: '{response}'")
            print(f"Routing Mode: '{routing_mode}'")
            
            self.assertEqual(response, "Keep practicing recursion first before moving to dynamic programming.")
            self.assertEqual(routing_mode, "groq_interim_direct")
            mock_chat.assert_called_once()

    @patch('edmentor.groq_client.llm.chat')
    @patch('edmentor.rag_engine.get_chroma_resources')
    async def test_groq_interim_mode_fallback(self, mock_get_chroma, mock_chat):
        print("\n--- Testing Groq Interim Mode (RAG Fallback on Error) ---")
        
        # Mock ChromaDB response with n_results=3
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["Start with recursion.", "Practice trees."]],
            "distances": [[0.3, 0.4]]
        }
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [0.1] * 384
        mock_get_chroma.return_value = (mock_col, mock_embedder)
        
        # Force USE_LOCAL_MODEL = False
        with patch('edmentor.confidence_router.USE_LOCAL_MODEL', False):
            # 1. Mock Groq to return empty (simulating failure)
            mock_chat.return_value = ""
            
            # Second call to mock_chat will generate RAG wrapped response
            # Since rag_retrieve_and_respond will call groq_llm.chat to generate response from wrapped prompt
            mock_chat.side_effect = ["", "Here are insights: Start with recursion. Practice trees."]
            
            response, routing_mode = await generate_response_with_routing("explain dynamic programming", self.session_id)
            print(f"Response: '{response}'")
            print(f"Routing Mode: '{routing_mode}'")
            
            self.assertEqual(routing_mode, "groq_interim_rag_fallback")
            self.assertIn("Start with recursion", response)

if __name__ == "__main__":
    unittest.main()
