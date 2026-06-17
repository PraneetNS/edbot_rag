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
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

tts_router = APIRouter()


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"   # Kokoro default voice (warm, clear)
    speed: float = 0.95        # Slightly slower than default — mentor cadence


# ── Kokoro singleton — loaded once at module import time ─────────────────────
# Loading the ONNX model per-request adds 3-6s cold start. Pre-loading here
# means the model is ready the moment the first /edmentor/tts request arrives.
_kokoro_instance = None
_kokoro_error: str | None = None

def _load_kokoro():
    global _kokoro_instance, _kokoro_error
    try:
        from kokoro_onnx import Kokoro
        import soundfile as sf  # noqa: F401 — ensure soundfile is importable

        backend_dir = Path(__file__).resolve().parent.parent
        model_path  = str(backend_dir / "kokoro-v1_0.onnx")
        voices_path = str(backend_dir / "voices-v1_0.bin")

        logger.info("[TTS] Loading Kokoro ONNX model...")
        _kokoro_instance = Kokoro(model_path, voices_path)
        logger.info("[TTS] Kokoro ready.")
    except ImportError:
        _kokoro_error = "kokoro-onnx not installed"
        logger.warning(f"[TTS] {_kokoro_error}")
    except FileNotFoundError:
        _kokoro_error = "Kokoro model files not found"
        logger.warning(f"[TTS] {_kokoro_error}")
    except Exception as e:
        _kokoro_error = str(e)
        logger.warning(f"[TTS] Kokoro load error: {e}")

# Load deferred to FastAPI startup event handler to ensure warmup order
# _load_kokoro()


def _try_kokoro(text: str, voice: str, speed: float):
    """
    Generate TTS using the pre-loaded Kokoro singleton.
    Returns audio bytes on success, None on failure.
    """
    if _kokoro_instance is None:
        logger.error(f"[TTS] Kokoro not available — reason: {_kokoro_error}")
        return None, None

    try:
        import soundfile as sf

        samples, sample_rate = _kokoro_instance.create(text, voice=voice, speed=speed, lang="en-us")

        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV")
        buf.seek(0)
        return buf.read(), "audio/wav"

    except Exception as e:
        logger.warning(f"[TTS] Kokoro generation error: {e}")
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
