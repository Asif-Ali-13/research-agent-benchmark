"""Persist agent artifacts for reproducibility."""

from __future__ import annotations

import json
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from research_agent.utils.logger import write_json_log


class LoggingToolInput(BaseModel):
    subdir: str = Field(default="agent_logs")
    filename: str = Field(...)
    payload_json: str = Field(..., description="JSON string payload")


class ArtifactLoggingTool(BaseTool):
    name: str = "artifact_logging_tool"
    description: str = "Save prompts, code, execution logs, or feedback as JSON under logs/."
    args_schema: type[BaseModel] = LoggingToolInput

    def _run(self, filename: str, payload_json: str, subdir: str = "agent_logs") -> str:
        payload: dict[str, Any] = json.loads(payload_json)
        path = write_json_log(subdir, filename, payload)
        return f"Saved artifact to {path}"


def log_artifact(subdir: str, filename: str, payload: dict[str, Any]) -> str:
    path = write_json_log(subdir, filename, payload)
    return str(path)
