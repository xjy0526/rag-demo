"""
llm_clients.py — Qwen client helpers built on DashScope's OpenAI-compatible API.
"""

from __future__ import annotations

import base64
import mimetypes
import traceback
from dataclasses import dataclass
from pathlib import Path

from src.config import QWEN_BASE_URL, QWEN_TEXT_MODEL, QWEN_VISION_MODEL


@dataclass
class LLMResponse:
    content: str


class QwenChatClient:
    """Small wrapper that exposes an invoke() API similar to LangChain chat models."""

    def __init__(
        self,
        api_key: str,
        model: str = QWEN_TEXT_MODEL,
        base_url: str = QWEN_BASE_URL,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt: str) -> LLMResponse:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return LLMResponse(content=content.strip())


def get_qwen_llm(api_key: str, model: str = QWEN_TEXT_MODEL, temperature: float = 0.2, max_tokens: int = 4096):
    """Return a thin Qwen chat client or None when no API key is provided."""
    try:
        if not api_key:
            return None
        return QwenChatClient(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(f"[llm_clients] Qwen init failed: {e}")
        return None


def call_llm(llm, prompt: str, fallback: str = "") -> str:
    """Safe LLM call with error handling."""
    try:
        if llm is None:
            return fallback or "[LLM not initialised — check DASHSCOPE_API_KEY]"
        response = llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content.strip()
        return str(response).strip()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[llm_clients] LLM call failed: {e}\n{tb}")
        return fallback or f"[LLM error: {e}]"


def _image_to_data_url(image_path: str) -> str:
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def call_qwen_vision(
    api_key: str,
    prompt: str,
    image_path: str,
    model: str = QWEN_VISION_MODEL,
    fallback: str = "",
) -> str:
    """Call a Qwen vision model with a local image."""
    try:
        if not api_key:
            return fallback or "[Vision model not initialised — check DASHSCOPE_API_KEY]"

        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
                    ],
                }
            ],
        )
        content = response.choices[0].message.content or ""
        return content.strip()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[llm_clients] Vision call failed: {e}\n{tb}")
        return fallback or f"[Vision LLM error: {e}]"


# Backward-compatible aliases to reduce churn in the rest of the codebase.
def get_groq_llm(api_key: str, model: str = QWEN_TEXT_MODEL):
    return get_qwen_llm(api_key, model=model, temperature=0.1, max_tokens=2048)


def get_gemini_llm(api_key: str, model: str = QWEN_TEXT_MODEL):
    return get_qwen_llm(api_key, model=model, temperature=0.2, max_tokens=4096)
