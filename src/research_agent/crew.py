"""Central LLM configuration (Gemini) and crew factory."""

from __future__ import annotations

import os

from crewai import LLM

from research_agent.utils.helpers import load_env


def get_llm() -> LLM:
    """Return Gemini LLM configured from environment."""
    load_env()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model = os.getenv("MODEL", "gemini/gemini-2.5-flash")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "8192"))

    return LLM(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
