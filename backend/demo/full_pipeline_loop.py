import os
import sys
import logging
import asyncio
from pathlib import Path
import numpy as np

# Resolve system paths to allow imports from backend root
DEMO_DIR = Path(__file__).resolve().parent
BACKEND_DIR = DEMO_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

# Ensure offline modes for HF hub
os.environ["HF_HUB_OFFLINE"] = "1"

logger = logging.getLogger(__name__)

# Lazy load audio packages to avoid crash at startup if dependencies are missing
try:
    from faster_whisper import WhisperModel
    import sounddevice as sd
    from kokoro_onnx import Kokoro
    import soundfile as sf
    _AUDIO_LIBS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Audio/STT/TTS libraries not fully installed: {e}. Run 'pip install -r requirements-local.txt'")
    _AUDIO_LIBS_AVAILABLE = False

# Import modular components
from edmentor.intent_router import is_off_domain
from edmentor.confidence_router import generate_response_with_routing
from edmentor.rag_engine import rag_retrieve_and_respond
from edmentor.safety_filter import edumentor_filter

# Global models
whisper_model = None
kokoro_tts_model = None

def init_audio_models():
    """Initializes Whisper STT and Kokoro TTS models once."""
    global whisper_model, kokoro_tts_model
    if not _AUDIO_LIBS_AVAILABLE:
        print("Audio libraries are missing. Cannot initialize models.")
        return False

    import torch
    # 1. Initialize Whisper (transcribe on CPU if CUDA is not available)
    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_compute = "int8"
    print(f"Loading Whisper model ('base.en') on device={whisper_device}...")
    whisper_model = WhisperModel("base.en", device=whisper_device, compute_type=whisper_compute)

    # 2. Initialize Kokoro TTS
    onnx_path = BACKEND_DIR / "kokoro-v1_0.onnx"
    voices_path = BACKEND_DIR / "voices-v1_0.bin"
    if onnx_path.exists() and voices_path.exists():
        print("Loading Kokoro ONNX voice synthesizer...")
        kokoro_tts_model = Kokoro(str(onnx_path), str(voices_path))
    else:
        print(f"Kokoro model files not found in: {BACKEND_DIR}. TTS audio generation will be simulated.")
    
    return True

def kokoro_tts(text: str) -> np.ndarray:
    """Generates audio samples using Kokoro ONNX."""
    if kokoro_tts_model is None:
        print(f"[Simulation] Speaking: {text}")
        return np.zeros(16000, dtype=np.float32)  # Return dummy silence
    try:
        samples, _ = kokoro_tts_model.create(
            text, 
            voice="af_heart", 
            speed=0.95, 
            lang="en-us"
        )
        return samples
    except Exception as e:
        logger.error(f"Kokoro TTS generation error: {e}")
        return np.zeros(16000, dtype=np.float32)

async def full_pipeline(audio_array: np.ndarray) -> np.ndarray:
    """
    Orchestrated voice loop (Requirements Two):
    STT -> Intent boundary -> confidence routing -> safety filter -> TTS.
    """
    if whisper_model is None:
        print("Whisper model not initialized.")
        return None

    # 1. STT
    # Convert audio to float32 if needed and transcribe
    audio_float = audio_array.astype(np.float32)
    segments, _ = whisper_model.transcribe(audio_float, beam_size=1)
    query = " ".join([s.text for s in segments]).strip()
    if not query:
        print("No audio input detected.")
        return None
    print(f"STT Transcript: '{query}'")

    # 2. Intent check
    if is_off_domain(query):
        response = "That's outside my lane. I'm here for engineering, placements, DSA, and your career. What do you need help with there?"
        print(f"Intent check: OFF-DOMAIN. Response: {response}")
    else:
        # 3. LLM with routing (local Qwen vs direct RAG fallback)
        # Encapsulated in the confidence_router
        response, routing_mode = await generate_response_with_routing(query)
        print(f"LLM routing mode: {routing_mode}")

    # 4. Safety filter
    response = edumentor_filter(response)
    print(f"Final safety filtered response: '{response}'")

    # 5. TTS voice generation
    audio = kokoro_tts(response)
    return audio

async def run_live_audio_loop():
    """Continuously records audio from microphone and runs full_pipeline."""
    if not init_audio_models():
        print("Failed to initialize audio models. Exiting loop.")
        return

    print("\n=== Edmentor Live Voice Loop Active ===")
    print("Press Ctrl+C to terminate.")
    
    sample_rate = 16000
    duration = 5.0  # record in 5-second blocks
    
    while True:
        try:
            print("\nListening...")
            loop = asyncio.get_running_loop()
            
            def record_mic():
                return sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
                
            audio_recording = await loop.run_in_executor(None, record_mic)
            await asyncio.sleep(duration)
            sd.wait()
            
            audio_data = audio_recording.flatten()
            print("Processing voice...")
            audio_output = await full_pipeline(audio_data)
            
            if audio_output is not None and len(audio_output) > 0:
                print("Playing response...")
                def play_audio():
                    sd.play(audio_output, sample_rate)
                    sd.wait()
                await loop.run_in_executor(None, play_audio)
                
        except KeyboardInterrupt:
            print("\nTerminating voice loop.")
            break
        except Exception as e:
            logger.error(f"Error in live voice loop: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    # If run directly, launch interactive live microphone test loop
    if _AUDIO_LIBS_AVAILABLE:
        asyncio.run(run_live_audio_loop())
    else:
        print("Local audio libraries are not installed. Cannot run mic loop directly.")
        print("Please install via: pip install -r requirements-local.txt")
