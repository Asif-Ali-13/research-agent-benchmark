"""Compute research evaluation metrics from run results."""

from __future__ import annotations

import json
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from research_agent.evaluation.reliability_metrics import ReliabilityMetrics
from research_agent.models.result_schema import AgentRunResult


class MetricsToolInput(BaseModel):
    results_json: str = Field(..., description="JSON list of AgentRunResult dicts")


class MetricsComputationTool(BaseTool):
    name: str = "metrics_computation_tool"
    description: str = "Compute TCR, ESR, RR, FIS, and DA from experiment run results."
    args_schema: type[BaseModel] = MetricsToolInput

    def _run(self, results_json: str) -> str:
        data: list[dict[str, Any]] = json.loads(results_json)
        results = [AgentRunResult.model_validate(r) for r in data]
        metrics = ReliabilityMetrics.from_results(results)
        return json.dumps(metrics.summary(), indent=2)
