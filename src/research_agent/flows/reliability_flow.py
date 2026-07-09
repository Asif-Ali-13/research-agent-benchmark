"""Baseline single-pass workflow (no feedback revision loop)."""

from __future__ import annotations

import re
import uuid
from typing import Any

from crewai import Agent, Crew, Process, Task

from research_agent.crew import get_llm
from research_agent.evaluation.reliability_metrics import compute_run_level_metrics
from research_agent.evaluation.scoring import evaluate_run
from research_agent.memory.memory_manager import MemoryManager
from research_agent.models.result_schema import AgentRunResult, ExecutionStatus
from research_agent.tools.csv_tool import load_dataset_summary
from research_agent.tools.execution_tool import execute_code
from research_agent.utils.file_handler import FileHandler
from research_agent.utils.codegen import build_code_generation_user_prompt
from research_agent.utils.helpers import get_project_root, resolve_dataset_path, truncate_text
from research_agent.utils.llm_retry import call_with_retry
from research_agent.tools.logging_tool import log_artifact
from research_agent.utils.logger import setup_logger

logger = setup_logger("reliability_flow")


class ReliabilityFlow:
    """Single-agent-style baseline: plan → code → execute → evaluate → report (one pass)."""

    def __init__(self) -> None:
        self.llm = get_llm()
        self.file_handler = FileHandler()
        self.memory = MemoryManager()
        self.root = get_project_root()

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

    def _run_agent_task(self, role: str, goal: str, description: str) -> str:
        def _kickoff() -> str:
            agent = Agent(role=role, goal=goal, backstory=goal, llm=self.llm, verbose=True)
            task = Task(description=description, expected_output="Structured text output", agent=agent)
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
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
        dataset_path = resolve_dataset_path(dataset_path)
        dataset_summary = load_dataset_summary(dataset_path)

        logger.info("Baseline flow run_id=%s", run_id)

        plan = self._run_agent_task(
            role="Planner",
            goal="Create execution plan",
            description=(
                f"{self._read_prompt('planner_prompt.txt')}\n\n"
                f"Query: {query}\nDataset: {dataset_path}\nSummary:\n{dataset_summary}"
            ),
        )

        code_user_prompt = build_code_generation_user_prompt(
            query=query,
            dataset_path=dataset_path,
            plan=plan,
            dataset_summary=dataset_summary,
            code_prompt=self._read_prompt("code_prompt.txt"),
            run_id=run_id,
        )
        code_raw = self._run_agent_task(
            role="Code Generator",
            goal="Generate Python analysis code",
            description=code_user_prompt,
        )
        code = self._extract_code(code_raw)
        self.file_handler.save_generated_code(run_id, code, iteration=0)

        execution = execute_code(
            code=code,
            run_id=run_id,
            dataset_path=dataset_path,
            artifacts_run_id=run_id,
        )
        evaluation, feedback = evaluate_run(
            code=code,
            execution=execution,
            task_description=query,
            expected_outputs=expected_outputs,
            iteration=0,
            dataset_summary=dataset_summary,
            mode="baseline",
            plan=plan,
        )
        if evaluation.task_qualified is not None:
            evaluation.notes = (
                f"{evaluation.notes} | qualified={evaluation.task_qualified}"
            )

        figure_paths = [
            f for f in execution.output_files if f.endswith((".png", ".jpg", ".jpeg", ".svg"))
        ]
        report = self._run_agent_task(
            role="Report Writer",
            goal="Write markdown report",
            description=(
                f"{self._read_prompt('report_prompt.txt')}\n\n"
                f"Query: {query}\nPlan: {truncate_text(plan, 1500)}\n"
                f"Stdout: {truncate_text(execution.stdout, 2000)}\n"
                f"Saved figures: {figure_paths or 'none'}\n"
                f"Decision score: {evaluation.decision_quality_score if evaluation else 0.0}\n"
                f"Business alignment: {evaluation.decision_evaluation.scores.business_alignment if evaluation.decision_evaluation else 'n/a'}\n"
                f"Status: {execution.status.value}"
            ),
        )

        run_metrics = compute_run_level_metrics(
            AgentRunResult(
                run_id=run_id,
                mode="baseline",
                query=query,
                dataset_path=dataset_path,
                plan=plan,
                generated_code=code,
                execution=execution,
                evaluation=evaluation,
                feedback_history=[feedback],
                final_report=report,
                iterations=1,
                recovered_from_failure=False,
                metadata={"task_id": task_id},
            )
        )
        result = AgentRunResult(
            run_id=run_id,
            mode="baseline",
            query=query,
            dataset_path=dataset_path,
            plan=plan,
            generated_code=code,
            execution=execution,
            evaluation=evaluation,
            feedback_history=[feedback],
            final_report=report,
            iterations=1,
            recovered_from_failure=False,
            metrics={
                "task_completion_score": evaluation.task_completion_score,
                "execution_success": 1.0 if execution.status == ExecutionStatus.SUCCESS else 0.0,
                "decision_quality_score": evaluation.decision_quality_score or 0.0,
                "business_alignment_score": (
                    evaluation.decision_evaluation.scores.business_alignment
                    if evaluation.decision_evaluation is not None
                    else 0.0
                ),
                "feedback_improvement_score": run_metrics["fis"],
                "overall_reliability_score": run_metrics["ors"],
                "dqs": run_metrics["dqs"],
                "bas": run_metrics["bas"],
                "ors": run_metrics["ors"],
                "fis": run_metrics["fis"],
            },
            metadata={"task_id": task_id},
        )

        log_artifact("execution_logs", f"{run_id}_baseline.json", result.to_dict())
        self.memory.save_run(run_id, "baseline", result.to_dict())
        self.file_handler.save_report(report, f"report_{run_id}.md")
        return result
