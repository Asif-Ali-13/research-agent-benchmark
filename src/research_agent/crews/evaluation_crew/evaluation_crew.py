"""Evaluation crew for benchmark analysis and experiment comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Process, Task

from research_agent.crew import get_llm
from research_agent.tools.metrics_tool import MetricsComputationTool

CONFIG_DIR = Path(__file__).parent / "config"


def _load_config(filename: str) -> dict[str, Any]:
    with open(CONFIG_DIR / filename, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_evaluation_crew() -> Crew:
    agents_cfg = _load_config("agents.yaml")
    tasks_cfg = _load_config("tasks.yaml")
    llm = get_llm()

    benchmark_evaluator = Agent(**agents_cfg["benchmark_evaluator"], llm=llm)
    metrics_analyst = Agent(**agents_cfg["metrics_analyst"], llm=llm, tools=[MetricsComputationTool()])

    evaluate_task = Task(
        description=tasks_cfg["evaluate_benchmark_task"]["description"],
        expected_output=tasks_cfg["evaluate_benchmark_task"]["expected_output"],
        agent=benchmark_evaluator,
    )
    compare_task = Task(
        description=tasks_cfg["compare_experiments_task"]["description"],
        expected_output=tasks_cfg["compare_experiments_task"]["expected_output"],
        agent=metrics_analyst,
        context=[evaluate_task],
    )

    return Crew(
        agents=[benchmark_evaluator, metrics_analyst],
        tasks=[evaluate_task, compare_task],
        process=Process.sequential,
        verbose=True,
    )


def run_evaluation_crew(inputs: dict[str, Any]) -> str:
    return str(build_evaluation_crew().kickoff(inputs=inputs))
