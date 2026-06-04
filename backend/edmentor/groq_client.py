"""
edmentor/groq_client.py
────────────────────────
Async Groq API client — primary LLM for Edmentor.
Falls back to Ollama/Mistral if Groq key is missing or unavailable.

Model: llama-3.1-8b-instant
    - Fastest Groq model (~200-400ms end-to-end)
    - Sufficient quality for 80-word mentor responses
    - Freely available on Groq's free tier

GROQ_API_KEY must be set in backend/.env
"""

import os
import logging
import asyncio
from typing import List, Dict, AsyncGenerator, Optional

logger = logging.getLogger(__name__)

# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = __file__.replace("edmentor/groq_client.py", ".env").replace(
        "edmentor\\groq_client.py", ".env"
    )
    load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed — rely on shell env

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_MAX_TOKENS = 200      # ~80 words generous headroom
GROQ_TEMPERATURE = 0.4     # Slight warmth — mentor feel, not robotic


class GroqClient:
    """
    Async Groq LLM client.
    Uses the official `groq` Python SDK where available,
    falls back to direct httpx requests if SDK unavailable.
    """

    def __init__(self):
        self._client = None
        self._available = False
        self._init_client()

    def _init_client(self):
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set — Groq client unavailable.")
            return
        try:
            import groq as groq_sdk
            self._client = groq_sdk.AsyncGroq(api_key=GROQ_API_KEY)
            self._available = True
            logger.info(f"GroqClient: initialised with model={GROQ_MODEL}")
        except ImportError:
            logger.error(
                "groq package not installed. Run: pip install groq"
            )

    @property
    def is_available(self) -> bool:
        return self._available and self._client is not None

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = GROQ_MAX_TOKENS,
    ) -> str:
        """
        Send a chat completion request to Groq.

        Args:
            messages:   Full messages array (system + history + user turn).
            max_tokens: Override token limit if needed.

        Returns:
            Response text string. Never raises — returns error string on failure.
        """
        if not self.is_available:
            return self._groq_unavailable_message()

        try:
            completion = await self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=GROQ_TEMPERATURE,
                stop=None,
            )
            return completion.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"GroqClient.chat error: {e}")
            return self._groq_unavailable_message()

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = GROQ_MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion tokens from Groq.
        Yields individual token strings as they arrive.
        """
        if not self.is_available:
            yield self._groq_unavailable_message()
            return

        try:
            stream = await self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=GROQ_TEMPERATURE,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        except Exception as e:
            logger.error(f"GroqClient.chat_stream error: {e}")
            yield self._groq_unavailable_message()

    @staticmethod
    def _groq_unavailable_message() -> str:
        """
        Edmentor-voice canned response when Groq is unreachable.
        Sounds natural when spoken — does NOT dump raw chunks.
        """
        return (
            "I am having a connection issue right now. That is on me, not you. "
            "Give it a moment and ask me again."
        )


# ── Ollama fallback (secondary) ───────────────────────────────────────────────

class OllamaFallbackClient:
    """
    Secondary LLM when Groq is unavailable.
    Uses Ollama/Mistral running locally.
    Produces an Edmentor-voice response from the system prompt + messages.
    """

    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

    def is_available(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.OLLAMA_URL}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Calls Ollama /api/chat with full messages array (supports system role)."""
        if not self.is_available():
            return GroqClient._groq_unavailable_message()

        try:
            import requests
            payload = {
                "model": self.OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.4, "num_predict": 200},
            }
            loop = asyncio.get_running_loop()
            def _sync():
                r = requests.post(
                    f"{self.OLLAMA_URL}/api/chat",
                    json=payload,
                    timeout=60,
                )
                if r.status_code == 200:
                    return r.json().get("message", {}).get("content", "").strip()
                return GroqClient._groq_unavailable_message()

            return await loop.run_in_executor(None, _sync)
        except Exception as e:
            logger.error(f"OllamaFallback.chat error: {e}")
            return GroqClient._groq_unavailable_message()


# ── Unified LLM client ────────────────────────────────────────────────────────

class EdmentorLLM:
    """
    Unified LLM router:
        1. Groq API (primary — fast, always online)
        2. Ollama (secondary — local fallback)
        3. Canned voice response (last resort)
    """

    def __init__(self):
        self.groq = GroqClient()
        self.ollama = OllamaFallbackClient()

    async def chat(self, messages: List[Dict[str, str]]) -> str:
        if self.groq.is_available:
            return await self.groq.chat(messages)
        logger.warning("Groq unavailable — trying Ollama fallback")
        return await self.ollama.chat(messages)

    async def chat_stream(
        self, messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        if self.groq.is_available:
            async for token in self.groq.chat_stream(messages):
                yield token
        else:
            logger.warning("Groq unavailable — streaming from Ollama fallback")
            response = await self.ollama.chat(messages)
            # Simulate streaming for Ollama (no native async stream in fallback)
            words = response.split()
            for i in range(0, len(words), 3):
                yield " ".join(words[i : i + 3]) + " "
                await asyncio.sleep(0.02)

    def status(self) -> dict:
        return {
            "groq_available": self.groq.is_available,
            "groq_model": GROQ_MODEL,
            "ollama_available": self.ollama.is_available(),
            "ollama_model": self.ollama.OLLAMA_MODEL,
        }


# Module-level singleton
llm = EdmentorLLM()
