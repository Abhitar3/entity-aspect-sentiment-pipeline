# lc_llm.py
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None  # type: ignore[assignment]

load_dotenv()


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Central OpenAI LLM client.
    """
    if ChatOpenAI is None:
        raise RuntimeError(
            "langchain-openai/openai is not installed. "
            "Install with: pip install langchain-openai openai"
        )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    model = os.getenv("OPENAI_MODEL", "gpt-4.1").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def is_rate_limit_error(exc: Exception) -> bool:
    """
    Provider-agnostic rate-limit detector.
    Works for Groq/OpenAI style exceptions without importing provider-specific classes.
    """
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "rate_limit" in name:
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    msg = str(exc).lower()
    tokens = ("rate limit", "rate_limit", "429", "tpm", "tpd", "too many requests")
    return any(t in msg for t in tokens)
