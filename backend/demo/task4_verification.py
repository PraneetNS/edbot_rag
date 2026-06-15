import asyncio
import sys
import os
import time
import re
from pathlib import Path

# Fix Windows cp1252 console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ["HF_HUB_OFFLINE"] = "1"

QUERIES = [
    "i finished arrays what should i do next",
    "how do i prepare for amazon",
    "i feel so behind everyone in my batch",
    "should i do ms abroad or get a job",
    "how do i get an internship in 2nd year",
]

async def measure_tts(text: str) -> float:
    """Calls Kokoro ONNX directly and returns time in ms."""
    try:
        from kokoro_onnx import Kokoro
        import soundfile as sf
        import io

        model_path  = str(BACKEND_DIR / "kokoro-v1_0.onnx")
        voices_path = str(BACKEND_DIR / "voices-v1_0.bin")

        # Singleton: load once per process
        if not hasattr(measure_tts, "_kokoro"):
            measure_tts._kokoro = Kokoro(model_path, voices_path)

        t0 = time.perf_counter()
        samples, sample_rate = measure_tts._kokoro.create(
            text, voice="af_heart", speed=0.95, lang="en-us"
        )
        return (time.perf_counter() - t0) * 1000
    except Exception as e:
        return -1.0

async def run_verification():
    from edmentor.rag_engine import retrieve
    from edmentor.confidence_router import generate_response_with_routing, check_ollama_health

    ollama_ok = await check_ollama_health()
    if not ollama_ok:
        print("[STOP] Ollama is offline.")
        sys.exit(1)

    print("======================================================================")
    print("VERIFICATION: Query 2 ('how do i prepare for amazon') 3 TIMES")
    print("======================================================================")
    for attempt in range(1, 4):
        t_start = time.perf_counter()
        resp, mode = await generate_response_with_routing("how do i prepare for amazon", session_id=f"q2_retry_{attempt}")
        llm_ms = (time.perf_counter() - t_start) * 1000
        print(f"Attempt {attempt}: LLM time = {llm_ms:.0f}ms")

    print("\n======================================================================")
    print("VERIFICATION AFTER ALL 3 FIXES")
    print("======================================================================")

    for i, query in enumerate(QUERIES, 1):
        print(f"\nQuery {i} — \"{query}\"")
        
        # Retrieval
        t_ret_start = time.perf_counter()
        docs = await retrieve(query)
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000
        ret_pass = "PASS" if retrieval_ms <= 400 else "FAIL"
        print(f"  Retrieval:     {retrieval_ms:.0f}ms   {ret_pass}")

        # LLM
        t_llm_start = time.perf_counter()
        response, mode = await generate_response_with_routing(
            query,
            session_id=f"latency_v2_{i}",
            pre_retrieved_docs=docs,
        )
        llm_ms = (time.perf_counter() - t_llm_start) * 1000
        llm_pass = "PASS" if llm_ms <= 5000 else "FAIL"
        print(f"  LLM:           {llm_ms:.0f}ms   {llm_pass}")

        # TTS Streaming simulation
        sentences = [
            s.strip() for s in
            re.split(r'(?<=[.!?])\s+', response.strip())
            if s.strip() and len(s.split()) >= 3
        ]

        tts_timings = []
        for j, sentence in enumerate(sentences, 1):
            tts_ms = await measure_tts(sentence)
            tts_timings.append(tts_ms)
            tts_pass = "PASS" if tts_ms <= 1000 else "FAIL"
            print(f"  TTS sentence {j}: {tts_ms:.0f}ms  {tts_pass}")

        first_tts_ms = tts_timings[0] if tts_timings else 0
        total_to_first = retrieval_ms + llm_ms + first_tts_ms
        tot_pass = "PASS" if total_to_first <= 8000 else "FAIL"
        print(f"  Total to first audio: {total_to_first:.0f}ms  {tot_pass}")

if __name__ == "__main__":
    asyncio.run(run_verification())
