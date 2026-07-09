"""Multi-agent workflow with structured feedback revision loop."""

from __future__ import annotations

import re
import uuid

from research_agent.crew import get_llm
from research_agent.evaluation.reliability_metrics import compute_run_level_metrics
from research_agent.evaluation.scoring import evaluate_run
from research_agent.memory.feedback_memory import FeedbackMemory
from research_agent.memory.memory_manager import MemoryManager
from research_agent.models.result_schema import AgentRunResult, ExecutionStatus
from research_agent.tools.csv_tool import load_dataset_summary
from research_agent.tools.execution_tool import execute_code
from research_agent.utils.file_handler import FileHandler
from research_agent.utils.codegen import build_code_generation_user_prompt
from research_agent.utils.helpers import get_env_int, get_project_root, resolve_dataset_path, truncate_text
from research_agent.utils.llm_retry import call_with_retry
from research_agent.tools.logging_tool import log_artifact
from research_agent.utils.logger import setup_logger

logger = setup_logger("feedback_flow")


class FeedbackFlow:
    """Plan → code → execute → evaluate → feedback → revise (loop) → report."""

    def __init__(self) -> None:
        self.llm = get_llm()
        self.file_handler = FileHandler()
        self.memory = MemoryManager()
        self.root = get_project_root()
        self.max_iterations = get_env_int("MAX_FEEDBACK_ITERATIONS", 3)

    def _read_prompt(self, name: str) -> str:
        return (self.root / "prompts" / name).read_text(encoding="utf-8")

    def _extract_code(self, text: str) -> str:
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _generate_with_llm(self, system: str, user: str) -> str:
        from crewai import Agent, Crew, Process, Task

        def _kickoff() -> str:
            agent = Agent(role="Assistant", goal=system, backstory=system, llm=self.llm, verbose=False)
            task = Task(description=user, expected_output="Response", agent=agent)
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            return str(crew.kickoff())

        return call_with_retry(_kickoff)

    def run(
        self,
        query: str,
        dataset_path: str,
        task_id: str | None = None,
        expected_outputs: list[str] | None = None,
    ) -> AgentRunResult:
        run_id = str(uuid.uuid4())[:8]
        feedback_memory = FeedbackMemory()
        dataset_path = resolve_dataset_path(dataset_path)
        dataset_summary = load_dataset_summary(dataset_path)

        plan = self._generate_with_llm(
            self._read_prompt("planner_prompt.txt"),
            f"Query: {query}\nDataset: {dataset_path}\nSummary:\n{dataset_summary}",
        )

        code = ""
        execution = None
        evaluation = None
        first_failure = False
        recovered = False

        code_prompt = self._read_prompt("code_prompt.txt")

        for iteration in range(self.max_iterations):
            feedback_context = feedback_memory.to_prompt_context()
            code_user_prompt = build_code_generation_user_prompt(
                query=query,
                dataset_path=dataset_path,
                plan=plan,
                dataset_summary=dataset_summary,
                code_prompt=code_prompt,
                run_id=run_id,
                iteration=iteration,
                feedback_context=feedback_context,
            )
            code_raw = self._generate_with_llm(
                self._read_prompt("code_prompt.txt"),
                code_user_prompt,
            )
            code = self._extract_code(code_raw)
            self.file_handler.save_generated_code(run_id, code, iteration=iteration)

            execution = execute_code(
                code=code,
                run_id=f"{run_id}_i{iteration}",
                dataset_path=dataset_path,
                artifacts_run_id=run_id,
            )
            evaluation, feedback = evaluate_run(
                code=code,
                execution=execution,
                task_description=query,
                expected_outputs=expected_outputs,
                iteration=iteration,
                dataset_summary=dataset_summary,
                mode="feedback_loop",
                plan=plan,
            )
            feedback_memory.add(feedback)
            self.memory.save_feedback(run_id, iteration, feedback.model_dump())

            iteration_artifact = {
                "generated_code": code,
                "execution_result": execution.model_dump() if execution else None,
                "decision_score": evaluation.decision_quality_score if evaluation else None,
                "execution_score": evaluation.task_completion_score if evaluation else None,
                "feedback": feedback.model_dump(),
                "revision": feedback.decision_recommendations,
                "final_decision": feedback.decision_summary,
                "json_metrics": {
                    "task_completion_score": evaluation.task_completion_score if evaluation else 0.0,
                    "decision_quality_score": evaluation.decision_quality_score if evaluation else 0.0,
                    "business_alignment_score": (
                        evaluation.decision_evaluation.scores.business_alignment
                        if evaluation and evaluation.decision_evaluation is not None
                        else 0.0
                    ),
                },
            }
            log_artifact("feedback_logs", f"{run_id}_iter_{iteration}.json", iteration_artifact)

            if execution.status != ExecutionStatus.SUCCESS and not first_failure:
                first_failure = True
            elif first_failure and execution.status == ExecutionStatus.SUCCESS:
                recovered = True

            qual_score = feedback.qualification_score if feedback.qualification_score is not None else 0.0
            gap_count = len(feedback.items)
            if feedback.passed:
                logger.info("Feedback loop passed at iteration %d", iteration)
                break
            logger.info(
                "Feedback iteration %d: passed=False heuristic=%.2f qualification=%.2f gaps=%d",
                iteration,
                feedback.score,
                qual_score,
                gap_count,
            )

            if not feedback_memory.should_continue(self.max_iterations):
                break

        output_files = execution.output_files if execution else []
        figure_paths = [f for f in output_files if f.endswith((".png", ".jpg", ".jpeg", ".svg"))]
        report = self._generate_with_llm(
            self._read_prompt("report_prompt.txt"),
            (
                f"Query: {query}\nIterations: {len(feedback_memory.history)}\n"
                f"Stdout: {truncate_text(execution.stdout if execution else '', 2500)}\n"
                f"Saved figures: {figure_paths or 'none'}\n"
                f"Decision score: {evaluation.decision_quality_score if evaluation else 0.0}\n"
                f"Business alignment: {evaluation.decision_evaluation.scores.business_alignment if evaluation and evaluation.decision_evaluation else 'n/a'}\n"
                f"Before vs after comparison:\n"
                f"- Initial feedback: {feedback_memory.history[0].summary if len(feedback_memory.history) > 0 else 'n/a'}\n"
                f"- Latest feedback: {feedback_memory.history[-1].summary if feedback_memory.history else 'n/a'}\n"
                f"Feedback history:\n{feedback_memory.to_prompt_context()}"
            ),
        )

        run_metrics = compute_run_level_metrics(
            AgentRunResult(
                run_id=run_id,
                mode="feedback_loop",
                query=query,
                dataset_path=dataset_path,
                plan=plan,
                generated_code=code,
                execution=execution,
                evaluation=evaluation,
                feedback_history=feedback_memory.history,
                final_report=report,
                iterations=len(feedback_memory.history),
                recovered_from_failure=recovered,
                metadata={"task_id": task_id},
            )
        )
        result = AgentRunResult(
            run_id=run_id,
            mode="feedback_loop",
            query=query,
            dataset_path=dataset_path,
            plan=plan,
            generated_code=code,
            execution=execution,
            evaluation=evaluation,
            feedback_history=feedback_memory.history,
            final_report=report,
            iterations=len(feedback_memory.history),
            recovered_from_failure=recovered,
            metrics={
                "task_completion_score": evaluation.task_completion_score if evaluation else 0.0,
                "execution_success": 1.0 if execution and execution.status == ExecutionStatus.SUCCESS else 0.0,
                "decision_quality_score": evaluation.decision_quality_score if evaluation else 0.0,
                "business_alignment_score": (
                    evaluation.decision_evaluation.scores.business_alignment
                    if evaluation and evaluation.decision_evaluation is not None
                    else 0.0
                ),
                "feedback_improvement_score": run_metrics["fis"],
                "overall_reliability_score": run_metrics["ors"],
                "dqs": run_metrics["dqs"],
                "bas": run_metrics["bas"],
                "ors": run_metrics["ors"],
                "fis": run_metrics["fis"],
                "FIS": run_metrics["fis"],
            },
            metadata={"task_id": task_id},
        )

        log_artifact("agent_logs", f"{run_id}_feedback.json", result.to_dict())
        self.memory.save_run(run_id, "feedback_loop", result.to_dict())
        self.file_handler.save_report(report, f"report_{run_id}.md")
        return result
