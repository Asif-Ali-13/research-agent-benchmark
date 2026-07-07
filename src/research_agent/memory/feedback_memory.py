"""In-memory feedback history for revision loops."""

from __future__ import annotations

from research_agent.models.result_schema import StructuredFeedback


class FeedbackMemory:
    """Track feedback across iterations within a single run."""

    def __init__(self) -> None:
        self.history: list[StructuredFeedback] = []

    def add(self, feedback: StructuredFeedback) -> None:
        self.history.append(feedback)

    def latest(self) -> StructuredFeedback | None:
        return self.history[-1] if self.history else None

    def improvement_score(self) -> float:
        """FIS: score_after - score_before (last two iterations)."""
        if len(self.history) < 2:
            return 0.0
        return self.history[-1].score - self.history[-2].score

    def should_continue(self, max_iterations: int) -> bool:
        latest = self.latest()
        if latest is None:
            return True
        if latest.passed:
            return False
        return len(self.history) < max_iterations

    def to_prompt_context(self) -> str:
        if not self.history:
            return "No prior feedback."
        lines = []
        for fb in self.history:
            lines.append(f"Iteration {fb.iteration} (score={fb.score:.2f}): {fb.summary}")
            if fb.decision_score is not None:
                lines.append(f"  - Decision score: {fb.decision_score:.1f}/100")
                lines.append(f"  - Decision summary: {fb.decision_summary}")
            for item in fb.items:
                lines.append(f"  - [{item.severity}] {item.category}: {item.message}")
                lines.append(f"    Recommendation: {item.recommendation}")
            if fb.decision_recommendations:
                lines.append("  - Decision recommendations:")
                for recommendation in fb.decision_recommendations:
                    lines.append(f"    * {recommendation}")
        return "\n".join(lines)
