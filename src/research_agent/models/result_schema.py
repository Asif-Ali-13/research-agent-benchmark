"""Result and feedback schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from research_agent.evaluation.decision_critic import DecisionCriticResult


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


class FeedbackItem(BaseModel):
    category: str
    severity: str  # low | medium | high
    message: str
    recommendation: str


class StructuredFeedback(BaseModel):
    iteration: int
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    items: list[FeedbackItem] = Field(default_factory=list)
    summary: str = ""
    qualification_score: float | None = Field(default=None, ge=0.0, le=1.0)
    qualified: bool | None = None
    decision_score: float | None = Field(default=None, ge=0.0, le=100.0)
    decision_summary: str = ""
    decision_recommendations: list[str] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    duration_seconds: float = 0.0
    output_files: list[str] = Field(default_factory=list)
    error_message: str | None = None


class EvaluationResult(BaseModel):
    task_completion_score: float = Field(ge=0.0, le=1.0)
    execution_success: bool
    output_completeness: float = Field(ge=0.0, le=1.0)
    visualization_present: bool = False
    ml_metrics_valid: bool = False
    notes: str = ""
    qualification_score: float | None = Field(default=None, ge=0.0, le=1.0)
    task_qualified: bool | None = None
    decision_evaluation: DecisionCriticResult | None = None
    decision_quality_score: float | None = Field(default=None, ge=0.0, le=100.0)
    structured_feedback: StructuredFeedback | None = None


class AgentRunResult(BaseModel):
    run_id: str
    mode: str
    query: str
    dataset_path: str
    plan: str = ""
    generated_code: str = ""
    execution: ExecutionResult | None = None
    evaluation: EvaluationResult | None = None
    feedback_history: list[StructuredFeedback] = Field(default_factory=list)
    final_report: str = ""
    iterations: int = 0
    recovered_from_failure: bool = False
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
