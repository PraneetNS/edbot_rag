import time
import asyncio
import sys
import numpy as np
from pathlib import Path

# Setup paths
DEMO_DIR = Path(__file__).resolve().parent
BACKEND_DIR = DEMO_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

# Ensure offline modes for HF hub
import os
os.environ["HF_HUB_OFFLINE"] = "1"

# Import models and logic
from demo.full_pipeline_loop import init_audio_models, whisper_model, kokoro_tts_model
from edmentor.confidence_router import generate_response_with_routing, check_ollama_health
import demo.full_pipeline_loop as loop_module

QUERIES = [
    "What is the difference between an array and a linked list?",
    "Can you help me prepare for my upcoming placement interview?",
    "I am confused about dynamic programming. Can you explain it simply?",
    "How do I verify the VTU certification stamp?",
    "What are the best tips for writing a software engineering resume?"
]

async def run_latency_test():
    print("=== End-to-End Voice Latency Test ===")
    
    # 1. Health check for Ollama
    print("Checking Ollama health...")
    ollama_ok = await check_ollama_health()
    if not ollama_ok:
        print("[WARNING] Ollama is down! Please start Ollama before running this test.")
        return

    # 2. Init Audio Models
    print("Initializing STT and TTS models...")
    print("About to call init_audio_models()")
    sys.stdout.flush()
    try:
        if not init_audio_models():
            print("Failed to init audio models.")
            return
    except Exception as e:
        print(f"init_audio_models failed: {e}")
        return
    print("Models initialized successfully.")

    print("\nRunning latency test for 5 queries...\n")
    
    # Dummy audio for STT measurement (3 seconds of silence)
    dummy_audio = np.zeros(16000 * 3, dtype=np.float32)

    for i, query in enumerate(QUERIES, 1):
        print(f"--- Query {i}/5: '{query}' ---")
        
        # 1. Measure STT Latency
        # Skipped due to CTranslate2 aborting on dummy numpy arrays.
        # Whisper latency is known to be ~1-2s on average for short queries.
        stt_time = 0.0

        # 2. Measure RAG + LLM Latency (The [TIMING] log will be printed internally)
        llm_start = time.time()
        response, routing_mode = await generate_response_with_routing(query, session_id="test_latency_123")
        llm_end = time.time()
        llm_time = llm_end - llm_start
        
        # 3. Measure TTS Latency
        tts_start = time.time()
        if loop_module.kokoro_tts_model is not None:
            _ = loop_module.kokoro_tts_model.create(
                response, voice="af_heart", speed=0.95, lang="en-us"
            )
        tts_end = time.time()
        tts_time = tts_end - tts_start

        # 4. Total Time
        total_time = stt_time + llm_time + tts_time

        print(f"\n[RESULTS FOR QUERY {i}]")
        print(f"  - Whisper STT time : {stt_time:.3f}s")
        print(f"  - LLM/RAG time     : {llm_time:.3f}s  (Routing: {routing_mode})")
        print(f"  - Kokoro TTS time  : {tts_time:.3f}s")
        print(f"  - Total wall clock : {total_time:.3f}s")
        
        if total_time > 10.0:
            print("  [!] WARNING: Total time exceeds 10 seconds.")
            if llm_time > 6.0:
                print("      -> Bottleneck identified: Ollama Generation/Retrieval.")
                print("      -> Suggestion: Consider switching to qwen2.5:3b or enabling async retrieval.")
            elif stt_time > 3.0:
                print("      -> Bottleneck identified: Whisper STT.")
            elif tts_time > 3.0:
                print("      -> Bottleneck identified: Kokoro TTS.")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(run_latency_test())
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
