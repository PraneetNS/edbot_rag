import sys
import os
from pathlib import Path
import asyncio

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

# Ensure offline modes for libraries
os.environ["HF_HUB_OFFLINE"] = "1"

async def test_pipeline(groq_key: str = None):
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        # Force re-read of env in client
        import edmentor.groq_client
        edmentor.groq_client.GROQ_API_KEY = groq_key
        edmentor.groq_client.llm.groq._init_client()

    from edmentor.topic_classifier import EdmentorTopicClassifier
    from rag.retrieval.retriever import retrieve_chunks_for_edmentor
    from edmentor.guard import guard as edmentor_guard
    from edmentor.memory import memory as edmentor_memory
    from edmentor.prompt import build_messages
    from edmentor.groq_client import llm as edmentor_llm
    from edmentor.voice_limit import enforce_voice_limit

    classifier = EdmentorTopicClassifier()

    question = "explain dynamic programming from scratch"
    print(f"\n==================================================")
    print(f"Testing Edmentor Pipeline Locally")
    print(f"Question: \"{question}\"")
    print(f"==================================================")

    # 1. Domain Guard Check
    is_blocked, guard_response, reason = edmentor_guard.check(question)
    print(f"1. Domain Guard: Blocked={is_blocked}, Reason={reason}")
    if is_blocked:
        print(f"Guard response: {guard_response}")
        return

    # 2. Topic Classifier
    topic = classifier.classify(question)
    print(f"2. Classified Topic: {topic}")

    # 3. RAG Retrieval
    print("3. Retrieving chunks...")
    chunks = retrieve_chunks_for_edmentor(question, topic=topic, top_k=3)
    print(f"Retrieved {len(chunks)} chunks:")
    for idx, c in enumerate(chunks, 1):
        snippet = c.replace("\n", " ")[:150]
        print(f"   ({idx}) {snippet}...")

    # 4. Build Messages
    history = edmentor_memory.get_last_turns("test_session", n=2)
    messages = build_messages(history=history, chunks=chunks, question=question)

    # 5. Call LLM
    print("4. Calling LLM (Groq)...")
    raw_response = await edmentor_llm.chat(messages)
    print(f"Raw Response: {raw_response}")

    # 6. Voice limit enforcement
    final_response = enforce_voice_limit(raw_response, max_words=80)
    word_count = len(final_response.split())
    print(f"\n5. Final Enforced Response (under 80 words):")
    print("-" * 50)
    print(final_response)
    print("-" * 50)
    print(f"Word count: {word_count}")
    print(f"Groq Available: {edmentor_llm.groq.is_available}")

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(test_pipeline(key))
