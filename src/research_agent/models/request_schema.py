"""API and CLI request schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RunMode(str, Enum):
    BASELINE = "baseline"
    FEEDBACK = "feedback"
    FULL = "full"


class BenchmarkMode(str, Enum):
    BASELINE = "baseline"
    FEEDBACK_LOOP = "feedback_loop"


class AnalysisRequest(BaseModel):
    """Request to run an analytics workflow."""

    query: str = Field(..., description="Natural language analytics task")
    dataset_path: str = Field(..., description="Path to CSV dataset")
    mode: RunMode = RunMode.FEEDBACK
    max_iterations: int = Field(default=3, ge=1, le=10)
    task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRequest(BaseModel):
    """Request to run benchmark suite."""

    mode: BenchmarkMode = BenchmarkMode.FEEDBACK_LOOP
    task_ids: list[str] | None = None
    limit: int | None = Field(default=None, ge=1)


class CompareExperimentsRequest(BaseModel):
    """Compare baseline vs feedback-loop experiment results."""

    baseline_dir: str = "experiments/baseline"
    feedback_dir: str = "experiments/feedback_loop"
    output_path: str = "reports/tables/comparison_metrics.csv"
