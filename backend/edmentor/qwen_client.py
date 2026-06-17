import logging
import requests
import asyncio

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"

class QwenClient:
    """
    Local Qwen client using Ollama backend.
    """
    def is_available(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1)
            return r.status_code == 200
        except Exception:
            return False

    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        if not self.is_available():
            return "Local Qwen model is offline."
        try:
            loop = asyncio.get_running_loop()
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3
                }
            }
            if system_prompt:
                payload["system"] = system_prompt
                
            def sync_post():
                r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=60)
                if r.status_code == 200:
                    text = r.json().get("response", "").strip()
                    text = text.replace('"', '').replace('“', '').replace('”', '')
                    return text
                return "Error connecting to LLM server."

            return await loop.run_in_executor(None, sync_post)
        except Exception as e:
            logger.error(f"Error calling Qwen generate: {e}")
            return "An unexpected error occurred during direct response generation."

qwen_client = QwenClient()
