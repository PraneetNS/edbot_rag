import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from api import is_educational_query
from rag.intent_router import classify_intent
from rag.formatter import conversational_fallback

# Create some mock hits with low and high scores
class MockNode:
    def __init__(self, node_id, text, file_name):
        self.node_id = node_id
        self.text = text
        self.metadata = {"file_name": file_name}

class MockHit:
    def __init__(self, score, node_id, text, file_name):
        self.score = score
        self.node = MockNode(node_id, text, file_name)
        self.text = text

mock_hits_low = [
    MockHit(score=0.15, node_id="1", text="This is some unrelated text about VTU certifications.", file_name="support.txt")
]

mock_hits_high = [
    MockHit(score=0.85, node_id="2", text="The main office is located at No - 110, 7th Cross Rd, Dollar Layout, BTM 2nd Stage, Bengaluru, Karnataka 560076.", file_name="faq.txt")
]

def run_tests():
    print("=== STARTING RAG RELAXATION TESTS ===")
    
    test_cases = [
        # (query, expect_educational, expect_intent, mock_hits, expect_substring_in_fallback)
        ("Hello!", True, "COURSE_QUERY", [], "Engineering Academic Mentor"),
        ("who are you?", True, "COURSE_QUERY", [], "dedicated Edutainer AI Engineering Academic Mentor"),
        ("how are you doing today?", True, "COURSE_QUERY", [], "support your engineering learning journey"),
        ("what is Python?", True, "COURSE_QUERY", [], None), # general academic query (allowed)
        ("what is a compiler?", True, "COURSE_QUERY", [], None), # general academic query (allowed)
        ("I am a 2nd year student, what should I study?", True, "COURSE_QUERY", [], "2nd-year engineering students"),
        ("How can a 3rd year student get internships?", True, "PLACEMENT_GUIDANCE", [], "3rd-year engineering students"),
        ("I am in final year, how to prepare for placement drives?", True, "PLACEMENT_GUIDANCE", [], "4th-year/final-year engineering students"),
        ("Explain binary search tree", True, "COURSE_QUERY", [], None), # general CS query (allowed)
        ("Where is the Edutainer office?", True, "COURSE_QUERY", mock_hits_high, "Bengaluru"), # proprietary, matches high score docs
        ("What is the secret flight schedule?", True, "COURSE_QUERY", mock_hits_low, "I currently do not have enough verified course information available"), # low score proprietary
        ("Can you tell me a recipe for chocolate cake?", False, "OUT_OF_SCOPE", [], "I am EduBot, your dedicated AI Engineering Academic Mentor"), # Blocked by safety / is_educational_query
        ("how to hack a website?", False, "OUT_OF_SCOPE", [], "I am EduBot, your dedicated AI Engineering Academic Mentor") # Blocked
    ]
    
    failures = 0
    
    for i, (query, exp_edu, exp_intent, hits, exp_fallback) in enumerate(test_cases):
        print(f"\nTest {i+1}: '{query}'")
        
        # Test 1: Educational check
        is_edu = is_educational_query(query)
        edu_ok = is_edu == exp_edu
        print(f"  Educational Filter: Got {is_edu} (Expected {exp_edu}) -> {'PASS' if edu_ok else 'FAIL'}")
        
        # Test 2: Intent check
        intent, score = classify_intent(query)
        intent_ok = intent == exp_intent
        print(f"  Intent Router: Got {intent} (Expected {exp_intent}) -> {'PASS' if intent_ok else 'FAIL'}")
        
        # Test 3: Fallback responses
        fallback_res = conversational_fallback(query, hits)
        fallback_ok = True
        if exp_fallback:
            fallback_ok = exp_fallback.lower() in fallback_res.lower()
            print(f"  Fallback Text check: Got '{fallback_res[:60]}...' -> {'PASS' if fallback_ok else 'FAIL'}")
        else:
            print(f"  Fallback Text: '{fallback_res[:60]}...'")
            
        if not (edu_ok and intent_ok and fallback_ok):
            failures += 1
            print("  --> TEST RESULT: FAILED")
        else:
            print("  --> TEST RESULT: PASSED")
            
    print("\n" + "="*40)
    print(f"TEST RUN COMPLETE: {len(test_cases) - failures}/{len(test_cases)} Passed.")
    print("="*40)
    
    if failures > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
