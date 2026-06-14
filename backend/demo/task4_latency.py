"""
task4_latency.py — Full pipeline latency measurement for Task 4
===============================================================
Measures per-stage timing for 5 queries through:
  ChromaDB retrieval -> qwen2.5:3b LLM -> Kokoro TTS

STT timing is noted as "N/A (text mode)" since we're sending text queries
directly — STT is measured in the live voice loop, not here.

Budgets:
  Retrieval: ≤ 400ms
  LLM:       ≤ 5000ms
  TTS:       ≤ 2000ms
  Total:     ≤ 10000ms  (excludes STT — add ~1-2s for real voice)
"""

import asyncio
import sys
import os
import time
import io
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

BUDGETS = {
    "retrieval_ms": 400,
    "llm_ms":       5000,
    "tts_ms":       2000,
    "total_ms":     10000,
}

async def measure_tts(text: str) -> float:
    """Calls Kokoro ONNX directly and returns time in ms."""
    try:
        from kokoro_onnx import Kokoro
        import soundfile as sf

        model_path  = str(BACKEND_DIR / "kokoro-v1_0.onnx")
        voices_path = str(BACKEND_DIR / "voices-v1_0.bin")

        # Singleton: load once per process
        if not hasattr(measure_tts, "_kokoro"):
            print("  [TTS init] Loading Kokoro ONNX model...")
            measure_tts._kokoro = Kokoro(model_path, voices_path)

        t0 = time.perf_counter()
        samples, sample_rate = measure_tts._kokoro.create(
            text, voice="af_heart", speed=0.95, lang="en-us"
        )
        # Encode to WAV bytes (mirrors what tts_router.py does)
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV")
        tts_ms = (time.perf_counter() - t0) * 1000
        audio_bytes = len(buf.getvalue())
        print(f"  [TTS] Kokoro — input_words={len(text.split())} output_ms={int(tts_ms)} audio_bytes={audio_bytes}")
        return tts_ms
    except ImportError:
        print("  [TTS] kokoro-onnx not installed — TTS skipped")
        return -1.0
    except FileNotFoundError:
        print("  [TTS] Kokoro model files not found — TTS skipped")
        return -1.0
    except Exception as e:
        print(f"  [TTS] Error: {e}")
        return -1.0


async def run_latency_test():
    from edmentor.rag_engine import retrieve
    from edmentor.confidence_router import generate_response_with_routing, check_ollama_health
    from edmentor.prompt_builder import build_prompt
    from edmentor.memory import memory as edmentor_memory

    print("=" * 70)
    print("TASK 4 — Full Pipeline Latency Measurement")
    print("=" * 70)

    ollama_ok = await check_ollama_health()
    print(f"\nOllama: {'ONLINE' if ollama_ok else 'OFFLINE'}")
    if not ollama_ok:
        print("[STOP] Ollama is offline. Cannot run latency test.")
        sys.exit(1)

    all_results = []

    for i, query in enumerate(QUERIES, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}/5: \"{query}\"")
        print(f"{'='*70}")

        wall_start = time.perf_counter()

        # ── Stage 1: Retrieval ─────────────────────────────────────────
        t_ret_start = time.perf_counter()
        docs = await retrieve(query)
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000
        print(f"  Retrieval:  {retrieval_ms:.0f}ms  (docs={len(docs)})")

        # ── Stage 2: Full LLM pipeline via confidence_router ──────────
        # (guard + retrieval already done, but router will re-retrieve if
        #  pre_retrieved_docs is passed — pass them to skip double-retrieval)
        t_llm_start = time.perf_counter()
        response, mode = await generate_response_with_routing(
            query,
            session_id=f"latency_t4_{i}",
            pre_retrieved_docs=docs,
        )
        llm_ms = (time.perf_counter() - t_llm_start) * 1000
        print(f"  LLM+guard:  {llm_ms:.0f}ms  (mode={mode}, words={len(response.split())})")
        print(f"  Response:   \"{response[:120]}\"")

        # ── Stage 3: Kokoro TTS ────────────────────────────────────────
        tts_ms = await measure_tts(response)

        wall_ms = (time.perf_counter() - wall_start) * 1000

        # ── Budget checks ──────────────────────────────────────────────
        def check(label, val_ms, budget_ms):
            status = "OK" if val_ms <= budget_ms or val_ms < 0 else "OVER BUDGET"
            return f"  {label:<14} {val_ms:>7.0f}ms  (budget <={budget_ms}ms)  {status}"

        print()
        print(f"  STT time:      N/A (text mode — add ~1-2s for real voice)")
        print(check("Retrieval:", retrieval_ms, BUDGETS["retrieval_ms"]))
        print(check("LLM:", llm_ms,       BUDGETS["llm_ms"]))
        if tts_ms >= 0:
            print(check("TTS:", tts_ms, BUDGETS["tts_ms"]))
        else:
            print(f"  TTS:           SKIPPED (Kokoro not installed)")
        print(check("Total:", wall_ms, BUDGETS["total_ms"]))

        all_results.append({
            "query": query,
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
            "tts_ms": tts_ms,
            "total_ms": wall_ms,
            "mode": mode,
        })

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("LATENCY SUMMARY -- All 5 queries")
    print(f"{'='*70}")
    print(f"{'#':<3} {'Retrieval':>10} {'LLM':>8} {'TTS':>8} {'Total':>8}  Mode")
    print(f"{'-'*70}")
    for i, r in enumerate(all_results, 1):
        tts_str = f"{r['tts_ms']:>8.0f}" if r["tts_ms"] >= 0 else "    skip"
        print(f"{i:<3} {r['retrieval_ms']:>9.0f}ms {r['llm_ms']:>7.0f}ms {tts_str}ms {r['total_ms']:>7.0f}ms  {r['mode']}")

    print(f"{'-'*70}")

    # ── Bottleneck report ──────────────────────────────────────────────────
    over_retrieval = sum(1 for r in all_results if r["retrieval_ms"] > BUDGETS["retrieval_ms"])
    over_llm       = sum(1 for r in all_results if r["llm_ms"]       > BUDGETS["llm_ms"])
    over_tts       = sum(1 for r in all_results if r["tts_ms"] >= 0 and r["tts_ms"] > BUDGETS["tts_ms"])
    over_total     = sum(1 for r in all_results if r["total_ms"]     > BUDGETS["total_ms"])

    print("\nBottleneck Report:")
    if over_retrieval >= 3:
        print(f"  ✗ RETRIEVAL over budget on {over_retrieval}/5 queries. Check HF_HUB_OFFLINE=1, embeddings cache.")
    else:
        print(f"  ✓ Retrieval OK ({over_retrieval}/5 over budget)")

    if over_llm >= 3:
        print(f"  ✗ LLM over budget on {over_llm}/5 queries. Confirm qwen2.5:3b, check GPU vs CPU.")
    else:
        print(f"  ✓ LLM OK ({over_llm}/5 over budget)")

    if over_tts >= 3:
        print(f"  ✗ TTS over budget on {over_tts}/5 queries. Ensure Kokoro loaded once at startup.")
    else:
        print(f"  ✓ TTS OK ({over_tts}/5 over budget)" if over_tts >= 0 else "  - TTS skipped")

    if over_total >= 3:
        print(f"  ✗ TOTAL over 10s budget on {over_total}/5 queries.")
    else:
        print(f"  ✓ Total wall-clock OK ({over_total}/5 over budget)")


if __name__ == "__main__":
    asyncio.run(run_latency_test())
