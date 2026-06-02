import json
import logging
import requests
import asyncio
from typing import AsyncGenerator
from ai_core.config import OLLAMA_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

class QwenOllamaClient:
    """
    Async client for sending prompt generations and streaming tokens from Ollama.
    """
    def __init__(self):
        pass

    def check_active(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=1)
            return r.status_code == 200
        except Exception:
            return False

    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Runs prompt generation synchronously inside an executor for async safety."""
        if not self.check_active():
            logger.warning("Ollama local service is offline. Direct LLM response unavailable.")
            return "I am sorry, my local LLM daemon is offline. Please start Ollama (Mistral/Qwen) to generate response."
            
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
                    return r.json().get("response", "").strip()
                return "Error connecting to LLM server."

            response = await loop.run_in_executor(None, sync_post)
            return response
        except Exception as e:
            logger.error(f"Error calling Qwen/Ollama generate: {e}")
            return "An unexpected error occurred during direct response generation."

    async def generate_stream(self, prompt: str, system_prompt: str = None) -> AsyncGenerator[str, None]:
        """Streams generated tokens asynchronously from the Ollama generation socket."""
        if not self.check_active():
            logger.warning("Ollama offline. Streaming unavailable.")
            yield "I am sorry, my local LLM daemon is offline. Please start Ollama (Mistral/Qwen) to stream response."
            return

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.3
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            # We run the streaming request in a thread pool using requests stream=True
            def sync_stream_connect():
                return requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json=payload,
                    stream=True,
                    timeout=10
                )

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, sync_stream_connect)
            
            if response.status_code != 200:
                yield "Error connecting to LLM server."
                return

            # Read stream lines inside the executor to make it async safe
            def get_next_line(iter_lines):
                try:
                    return next(iter_lines)
                except StopIteration:
                    return None

            iter_lines = response.iter_lines()
            while True:
                line = await loop.run_in_executor(None, get_next_line, iter_lines)
                if line is None:
                    break
                    
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except Exception as e:
                        logger.warning(f"Error parsing token line: {e}")
                # Yield context control
                await asyncio.sleep(0.001)
                
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield "An error occurred during response streaming."
