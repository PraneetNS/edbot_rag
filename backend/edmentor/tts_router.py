"""
edmentor/tts_router.py
───────────────────────
TTS endpoint for Edmentor voice output.

Strategy:
    1. Try kokoro-onnx (pip installable, no local model needed)
    2. If unavailable, return {"tts_unavailable": true}
       → frontend falls back to Web SpeechSynthesis

Endpoint: POST /edmentor/tts
    Body:  {"text": "..."}
    Returns: audio/mpeg stream  (if Kokoro available)
             JSON {"tts_unavailable": true}  (if not)
"""

import io
import logging
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

tts_router = APIRouter()


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"   # Kokoro default voice (warm, clear)
    speed: float = 0.95        # Slightly slower than default — mentor cadence


def _try_kokoro(text: str, voice: str, speed: float):
    """
    Attempt Kokoro TTS generation.
    Returns audio bytes on success, None on failure.
    """
    try:
        from kokoro_onnx import Kokoro
        import soundfile as sf
        import numpy as np
        from pathlib import Path

        # Resolve paths dynamically relative to the backend directory
        backend_dir = Path(__file__).resolve().parent.parent
        model_path = str(backend_dir / "kokoro-v1_0.onnx")
        voices_path = str(backend_dir / "voices-v1_0.bin")

        kokoro = Kokoro(model_path, voices_path)
        samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")

        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV")
        buf.seek(0)
        return buf.read(), "audio/wav"

    except ImportError:
        logger.info("kokoro-onnx not installed — TTS unavailable")
        return None, None
    except FileNotFoundError:
        logger.info("Kokoro model files not found — TTS unavailable")
        return None, None
    except Exception as e:
        logger.warning(f"Kokoro TTS error: {e}")
        return None, None


@tts_router.post("/edmentor/tts")
async def tts_endpoint(req: TTSRequest):
    """
    Generate Kokoro TTS audio for an Edmentor response.

    Returns:
        audio/wav stream if Kokoro is available.
        JSON {"tts_unavailable": true} if not — frontend uses SpeechSynthesis.
    """
    text = req.text.strip()
    if not text:
        return JSONResponse({"tts_unavailable": True, "reason": "empty_text"})

    audio_bytes, media_type = _try_kokoro(text, req.voice, req.speed)

    if audio_bytes is None:
        return JSONResponse({"tts_unavailable": True, "reason": "kokoro_not_installed"})

    def audio_stream():
        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i : i + chunk_size]

    return StreamingResponse(
        audio_stream(),
        media_type=media_type,
        headers={"Content-Disposition": "inline; filename=edmentor_response.wav"},
    )
