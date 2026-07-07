"""Research reliability metrics: TCR, ESR, RR, FIS, DA, DQS, BAS, ORS."""

from __future__ import annotations

from dataclasses import dataclass, field

from research_agent.models.result_schema import AgentRunResult, ExecutionStatus


def _feedback_improvement_for_result(result: AgentRunResult) -> float:
    if len(result.feedback_history) >= 2:
        return round(result.feedback_history[-1].score - result.feedback_history[-2].score, 4)
    return 0.0


def compute_run_level_metrics(result: AgentRunResult) -> dict[str, float]:
    """Compute single-run DQS/BAS/ORS metrics from decision-critic output."""
    evaluation = result.evaluation
    if evaluation and evaluation.decision_evaluation is not None:
        scores = evaluation.decision_evaluation.scores.model_dump()
        weights = {
            "data_understanding": 0.10,
            "preprocessing": 0.15,
            "feature_engineering": 0.15,
            "model_selection": 0.20,
            "evaluation_strategy": 0.15,
            "statistical_validity": 0.10,
            "explainability": 0.05,
            "business_alignment": 0.10,
        }
        dqs = round(sum(weights[name] * (scores.get(name, 0.0) / 10.0) for name in weights) * 100.0, 4)
        bas = round((scores.get("business_alignment", 0.0) / 10.0) * 100.0, 4)
    else:
        dqs = 0.0
        bas = 0.0

    tcr = round(evaluation.task_completion_score, 4) if evaluation else 0.0
    esr = 1.0 if result.execution and result.execution.status == ExecutionStatus.SUCCESS else 0.0
    if evaluation and evaluation.decision_evaluation is not None:
        scores = evaluation.decision_evaluation.scores.model_dump()
        decision_components = [
            "data_understanding",
            "preprocessing",
            "model_selection",
            "evaluation_strategy",
            "statistical_validity",
            "business_alignment",
        ]
        da = round(sum(scores.get(name, 0.0) / 10.0 for name in decision_components) / len(decision_components), 4)
    else:
        da = 1.0 if evaluation and evaluation.task_completion_score >= 0.5 else 0.0
    fis = _feedback_improvement_for_result(result)
    ors = round(
        0.15 * tcr + 0.15 * esr + 0.20 * da + 0.20 * (dqs / 100.0) + 0.15 * (bas / 100.0) + 0.15 * fis,
        4,
    ) * 100.0

    return {
        "tcr": tcr,
        "esr": esr,
        "da": da,
        "fis": fis,
        "dqs": round(dqs, 4),
        "bas": round(bas, 4),
        "ors": round(ors, 4),
    }


@dataclass
class ReliabilityMetrics:
    """Aggregate metrics across multiple agent runs."""

    total_tasks: int = 0
    completed_tasks: int = 0
    total_executions: int = 0
    successful_executions: int = 0
    total_failures: int = 0
    recovered_failures: int = 0
    feedback_improvements: list[float] = field(default_factory=list)
    correct_decisions: int = 0
    total_decisions: int = 0
    decision_scores: list[float] = field(default_factory=list)
    decision_accuracy_scores: list[float] = field(default_factory=list)
    business_alignment_scores: list[float] = field(default_factory=list)

    @property
    def task_completion_rate(self) -> float:
        return self.completed_tasks / self.total_tasks if self.total_tasks else 0.0

    @property
    def execution_success_rate(self) -> float:
        return (
            self.successful_executions / self.total_executions if self.total_executions else 0.0
        )

    @property
    def recovery_rate(self) -> float:
        return self.recovered_failures / self.total_failures if self.total_failures else 0.0

    @property
    def feedback_improvement_score(self) -> float:
        if not self.feedback_improvements:
            return 0.0
        return sum(self.feedback_improvements) / len(self.feedback_improvements)

    @property
    def decision_accuracy(self) -> float:
        if self.decision_accuracy_scores:
            return sum(self.decision_accuracy_scores) / len(self.decision_accuracy_scores)
        return self.correct_decisions / self.total_decisions if self.total_decisions else 0.0

    @property
    def decision_quality_score(self) -> float:
        if not self.decision_scores:
            return 0.0
        return sum(self.decision_scores) / len(self.decision_scores)

    @property
    def business_alignment_score(self) -> float:
        if not self.business_alignment_scores:
            return 0.0
        return sum(self.business_alignment_scores) / len(self.business_alignment_scores)

    @property
    def overall_reliability_score(self) -> float:
        return round(
            0.15 * self.task_completion_rate
            + 0.15 * self.execution_success_rate
            + 0.20 * self.decision_accuracy
            + 0.20 * (self.decision_quality_score / 100.0)
            + 0.15 * (self.business_alignment_score / 100.0)
            + 0.15 * self.feedback_improvement_score,
            4,
        ) * 100.0

    @classmethod
    def from_results(cls, results: list[AgentRunResult]) -> ReliabilityMetrics:
        m = cls()
        m.total_tasks = len(results)

        for r in results:
            eval_score = r.evaluation.task_completion_score if r.evaluation else 0.0
            completed = eval_score >= 0.7 or (r.execution and r.execution.status == ExecutionStatus.SUCCESS)
            if completed:
                m.completed_tasks += 1

            if r.execution:
                m.total_executions += 1
                if r.execution.status == ExecutionStatus.SUCCESS:
                    m.successful_executions += 1
                else:
                    m.total_failures += 1
                    if r.recovered_from_failure:
                        m.recovered_failures += 1

            run_feedback_improvement = _feedback_improvement_for_result(r)
            m.feedback_improvements.append(run_feedback_improvement)

            m.total_decisions += 1
            if r.evaluation and r.evaluation.decision_evaluation is not None:
                run_metrics = compute_run_level_metrics(r)
                m.decision_scores.append(run_metrics["dqs"])
                m.business_alignment_scores.append(run_metrics["bas"])
                m.decision_accuracy_scores.append(run_metrics["da"])
                if run_metrics["da"] >= 0.5:
                    m.correct_decisions += 1
            elif r.evaluation and r.evaluation.task_completion_score >= 0.5:
                m.correct_decisions += 1
                m.decision_accuracy_scores.append(1.0)
            else:
                m.decision_accuracy_scores.append(0.0)

        return m

    def summary(self) -> dict[str, float]:
        return {
            "TCR": round(self.task_completion_rate, 4),
            "ESR": round(self.execution_success_rate, 4),
            "RR": round(self.recovery_rate, 4),
            "FIS": round(self.feedback_improvement_score, 4),
            "DA": round(self.decision_accuracy, 4),
            "DQS": round(self.decision_quality_score, 4),
            "BAS": round(self.business_alignment_score, 4),
            "ORS": round(self.overall_reliability_score, 4),
            "total_tasks": float(self.total_tasks),
            "completed_tasks": float(self.completed_tasks),
        }
