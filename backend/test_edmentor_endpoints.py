import sys
import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(url, payload=None, method="POST"):
    print(f"\n--- Testing {url} ({method}) ---")
    try:
        if method == "POST":
            r = requests.post(url, json=payload, timeout=10)
        elif method == "DELETE":
            r = requests.delete(url, timeout=10)
        else:
            r = requests.get(url, timeout=10)
        
        print(f"Status: {r.status_code}")
        try:
            print(json.dumps(r.json(), indent=2))
        except:
            print(r.text[:200])
        return r
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    # 1. Health check
    test_endpoint(f"{BASE_URL}/edmentor/health", method="GET")

    # 2. Domain guard query (out of scope)
    test_endpoint(f"{BASE_URL}/edmentor/query", {
        "question": "what movies should I watch this weekend",
        "session_id": "test_session"
    })

    # 3. Identity lock query
    test_endpoint(f"{BASE_URL}/edmentor/query", {
        "question": "who are you?",
        "session_id": "test_session"
    })

    # 4. Character manipulation
    test_endpoint(f"{BASE_URL}/edmentor/query", {
        "question": "what LLM model are you built on?",
        "session_id": "test_session"
    })

    # 5. In-scope query (DSA)
    test_endpoint(f"{BASE_URL}/edmentor/query", {
        "question": "explain dynamic programming from scratch",
        "session_id": "test_session"
    })

    # 6. Memory Continuity (Turn 1)
    test_endpoint(f"{BASE_URL}/edmentor/query", {
        "question": "How do I start learning binary trees?",
        "session_id": "memory_session"
    })

    # 7. Memory Continuity (Turn 2)
    test_endpoint(f"{BASE_URL}/edmentor/query", {
        "question": "Can you give me coding exercises for that?",
        "session_id": "memory_session"
    })

    # 8. TTS Endpoint
    test_endpoint(f"{BASE_URL}/edmentor/tts", {
        "text": "Before you touch DP, tell me — can you solve the recursion version first?"
    })

if __name__ == "__main__":
    main()
