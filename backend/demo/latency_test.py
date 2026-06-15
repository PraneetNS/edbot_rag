import os, sys, time, asyncio, re
from pathlib import Path
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BACKEND_DIR))
os.environ["HF_HUB_OFFLINE"] = "1"

from edmentor.confidence_router import generate_response_with_routing
from demo.full_pipeline_loop import init_audio_models, kokoro_tts
import demo.full_pipeline_loop as fpl

queries = [
    "i finished arrays what should i do next",
    "how do i prepare for amazon",
    "i feel so behind everyone in my batch",
    "should i do ms abroad or get a job",
    "how do i get an internship in 2nd year"
]

from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

whisper_model = None
kokoro_model = None

def local_init():
    global whisper_model, kokoro_model
    print("Loading whisper on cpu...")
    whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
    print("Loading kokoro...")
    kokoro_model = Kokoro("kokoro-v1_0.onnx", "voices-v1_0.bin")
    
def local_tts(text):
    samples, _ = kokoro_model.create(text, voice="af_heart", speed=0.95, lang="en-us")
    return samples

async def run_latency_test():
    print("Initializing models...")
    local_init()
    
    for q in queries:
        print(f"\n{'='*50}\nQUERY: '{q}'\n{'='*50}")
        
        # 0. Generate dummy audio for the query to test Whisper
        audio_array = local_tts(q)
        audio_float = audio_array.astype(np.float32)
        
        # 1. STT
        t0 = time.time()
        segments, _ = whisper_model.transcribe(audio_float, beam_size=1)
        stt_text = " ".join([s.text for s in segments]).strip()
        stt_time = int((time.time() - t0) * 1000)
        
        # 2. Retrieval & LLM
        response, mode = await generate_response_with_routing(q)
            
        # 3. TTS
        t2 = time.time()
        local_tts(response)
        tts_time = int((time.time() - t2) * 1000)
        
        print(f"STT time:       {stt_time}ms")
        print(f"TTS time:       {tts_time}ms")

if __name__ == "__main__":
    asyncio.run(run_latency_test())
