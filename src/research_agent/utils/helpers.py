"""Shared helper utilities."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_PROJECT_ROOT: Path | None = None


def get_project_root() -> Path:
    """Resolve repository root (contains pyproject.toml)."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT

    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            _PROJECT_ROOT = parent
            return parent

    _PROJECT_ROOT = Path.cwd()
    return _PROJECT_ROOT


def load_env() -> None:
    """Load .env from project root if present."""
    env_path = get_project_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def sanitize_filename(name: str) -> str:
    """Create a safe filename fragment."""
    return re.sub(r"[^\w\-.]", "_", name)[:128]


def truncate_text(text: str, max_len: int = 4000) -> str:
    """Truncate long strings for logs and prompts."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 50] + "\n... [truncated]"


def get_env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def resolve_dataset_path(dataset_path: str) -> str:
    """Return project-relative path when possible (for prompts and reproducibility)."""
    path = Path(dataset_path).expanduser()
    if not path.is_absolute():
        path = get_project_root() / path
    path = path.resolve()
    try:
        return str(path.relative_to(get_project_root()))
    except ValueError:
        return str(path)


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    result.update(override)
    return result
