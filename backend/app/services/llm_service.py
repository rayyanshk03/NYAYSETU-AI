"""Reusable LLM service for NyayaSetu AI.

Supports configurable LLM providers:
- 'template': Deterministic local grounded template synthesis (default; zero API keys or external dependencies required)
- 'ollama': Local Ollama instance (e.g. llama3.2, mistral, gemma2) via HTTP REST
- 'openai': OpenAI or any OpenAI-compatible provider (Groq, OpenRouter, vLLM, DeepSeek) using standard HTTP POST
- 'gemini': Google Gemini API via OpenAI-compatible endpoint or Google REST

Features:
- Configurable model name, temperature, and max tokens.
- No hardcoded API keys; loads from environment.
- Graceful error recovery: falls back to template synthesis if network or provider fails.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Manages LLM inference requests across hosted providers, local Ollama, and fallback templates."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        self.model_name = model_name or settings.LLM_MODEL_NAME
        self.api_key = api_key or settings.LLM_API_KEY
        self.api_base = api_base or settings.LLM_API_BASE
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        self.timeout = timeout or settings.LLM_REQUEST_TIMEOUT

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """
        Execute an LLM generation call using the configured provider.
        Returns the generated text string, or None if the provider cannot be reached
        (allowing the caller to use deterministic grounded fallback).
        """
        if self.provider == "template":
            return None  # Signal caller to use grounded template synthesis

        if self.provider == "ollama":
            return self._call_ollama(prompt, system_prompt)

        if self.provider in ("openai", "groq", "gemini"):
            return self._call_openai_compatible(prompt, system_prompt)

        logger.warning("Unrecognized LLM provider '%s'. Using template synthesis.", self.provider)
        return None

    def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Generate text using a local Ollama instance."""
        base_url = (self.api_base or settings.OLLAMA_BASE_URL).rstrip("/")
        endpoint = f"{base_url}/api/generate"

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get("response", "").strip()
                    if response_text:
                        return response_text
                else:
                    logger.warning(
                        "Ollama request to %s failed with status %d: %s",
                        endpoint,
                        response.status_code,
                        response.text[:200],
                    )
        except Exception as e:
            logger.warning("Ollama connection error (%s). Falling back to template synthesis.", e)

        return None

    def _call_openai_compatible(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Generate text using an OpenAI-compatible API endpoint (OpenAI, Groq, Gemini, etc.)."""
        if not self.api_key:
            logger.warning("No LLM_API_KEY provided for provider '%s'. Falling back to template synthesis.", self.provider)
            return None

        # Determine default base URL if not explicitly configured
        base_url = self.api_base
        if not base_url:
            if self.provider == "groq":
                base_url = "https://api.groq.com/openai/v1"
            elif self.provider == "gemini":
                base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
            else:
                base_url = "https://api.openai.com/v1"

        endpoint = f"{base_url.rstrip('/')}/chat/completions"

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if choices and "message" in choices[0] and "content" in choices[0]["message"]:
                        text = choices[0]["message"]["content"].strip()
                        if text:
                            return text
                else:
                    logger.warning(
                        "LLM API request to %s returned HTTP %d: %s",
                        endpoint,
                        response.status_code,
                        response.text[:200],
                    )
        except Exception as e:
            logger.warning("LLM API request failed (%s). Falling back to template synthesis.", e)

        return None


# Global singleton instance
llm_service = LLMService()
