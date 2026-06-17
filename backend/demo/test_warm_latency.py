import time
import urllib.request
import json
import sys

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000"

def measure_warm_tts_latency():
    print("=== TTS WARM RUN LATENCY DIAGNOSTICS ===")
    
    # Sentence to test
    payload = {
        "text": "Here is the roadmap for your placement preparation.",
        "voice": "af_heart",
        "speed": 0.95
    }
    
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        f"{BASE_URL}/edmentor/tts",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    print("Sending request to warmed-up Kokoro TTS server...")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            content = resp.read()
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000
            
            print(f"Response status: {resp.status}")
            print(f"Content-type: {resp.headers.get('Content-Type')}")
            print(f"Audio size: {len(content)} bytes")
            if "application/json" in resp.headers.get('Content-Type', ''):
                print(f"JSON Body: {content.decode('utf-8')}")
            print(f"Warm serving latency: {elapsed_ms:.1f}ms")
            
            if elapsed_ms < 400.0:
                print("\033[32m[PASS] Warm serve latency is under 400ms budget!\033[0m")
                sys.exit(0)
            else:
                print("\033[31m[FAIL] Warm serve latency exceeds 400ms budget.\033[0m")
                sys.exit(1)
                
    except Exception as e:
        print(f"\033[31mError during request: {e}\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    measure_warm_tts_latency()
