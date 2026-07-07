"""Structured JSON logging for research reproducibility."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_agent.utils.helpers import get_project_root

LOG_DIR = get_project_root() / "logs"


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            payload["data"] = record.extra_data
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logger(
    name: str = "research_agent",
    level: str | None = None,
    log_subdir: str = "agent_logs",
) -> logging.Logger:
    """Configure root-style logger with console and JSON file handlers."""
    import os

    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / log_subdir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    json_path = LOG_DIR / log_subdir / f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
    file_handler = logging.FileHandler(json_path, encoding="utf-8")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    return logger


def log_structured(logger: logging.Logger, message: str, **data: Any) -> None:
    """Emit a log line with attached structured data."""
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        "(structured)",
        0,
        message,
        (),
        None,
    )
    record.extra_data = data  # type: ignore[attr-defined]
    logger.handle(record)


def write_json_log(subdir: str, filename: str, payload: dict[str, Any]) -> Path:
    """Persist a JSON artifact under logs/."""
    out_dir = LOG_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
