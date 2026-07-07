"""Retry helpers for Gemini / CrewAI rate limits (429)."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TypeVar

from research_agent.utils.helpers import get_env_int
from research_agent.utils.logger import setup_logger

logger = setup_logger("llm_retry")

T = TypeVar("T")

_RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "quota", "rate limit", "rate-limit")


def is_rate_limit_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker.lower() in message for marker in _RATE_LIMIT_MARKERS)


def parse_retry_delay_seconds(exc: BaseException, default: float = 35.0) -> float:
    """Parse 'retry in Ns' from Gemini error payloads."""
    match = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", str(exc), re.I)
    if match:
        return float(match.group(1)) + 1.0
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", str(exc), re.I)
    if match:
        return float(match.group(1)) + 1.0
    return default


def truncate_error(exc: BaseException, max_len: int = 200) -> str:
    text = str(exc).replace("\n", " ")
    return text[:max_len] + ("..." if len(text) > max_len else "")


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int | None = None,
    default_delay: float = 35.0,
) -> T:
    """Call fn, retrying on API rate-limit errors with backoff."""
    attempts = max_attempts or get_env_int("LLM_RETRY_MAX_ATTEMPTS", 4)
    last_exc: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if not is_rate_limit_error(exc) or attempt >= attempts:
                raise
            delay = parse_retry_delay_seconds(exc, default_delay)
            logger.warning(
                "Rate limit hit (attempt %d/%d), sleeping %.0fs: %s",
                attempt,
                attempts,
                delay,
                truncate_error(exc),
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc
