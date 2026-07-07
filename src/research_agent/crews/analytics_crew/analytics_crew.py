"""Multi-agent analytics crew (CrewAI sequential process)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Process, Task

from research_agent.crew import get_llm
from research_agent.tools.csv_tool import CSVInspectionTool
from research_agent.tools.execution_tool import CodeExecutionTool
from research_agent.tools.logging_tool import ArtifactLoggingTool
from research_agent.tools.metrics_tool import MetricsComputationTool
from research_agent.tools.model_tool import ModelTrainingTool
from research_agent.tools.visualization_tool import VisualizationTool
from research_agent.utils.helpers import get_project_root

CONFIG_DIR = Path(__file__).parent / "config"


def _load_config(filename: str) -> dict[str, Any]:
    with open(CONFIG_DIR / filename, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_prompts() -> dict[str, str]:
    root = get_project_root() / "prompts"
    return {
        "planner": (root / "planner_prompt.txt").read_text(encoding="utf-8"),
        "evaluator": (root / "evaluator_prompt.txt").read_text(encoding="utf-8"),
        "feedback": (root / "feedback_prompt.txt").read_text(encoding="utf-8"),
        "report": (root / "report_prompt.txt").read_text(encoding="utf-8"),
    }


def build_analytics_crew() -> Crew:
    """Build six-agent sequential crew from YAML config."""
    agents_cfg = _load_config("agents.yaml")
    tasks_cfg = _load_config("tasks.yaml")
    prompts = _load_prompts()
    llm = get_llm()

    planner = Agent(
        **agents_cfg["planner_agent"],
        llm=llm,
        tools=[CSVInspectionTool()],
        backstory=agents_cfg["planner_agent"]["backstory"] + "\n\n" + prompts["planner"],
    )
    coder = Agent(
        **agents_cfg["code_agent"],
        llm=llm,
        tools=[CSVInspectionTool(), VisualizationTool(), ModelTrainingTool()],
    )
    executor = Agent(
        **agents_cfg["executor_agent"],
        llm=llm,
        tools=[CodeExecutionTool(), ArtifactLoggingTool()],
    )
    evaluator = Agent(
        **agents_cfg["evaluator_agent"],
        llm=llm,
        tools=[MetricsComputationTool()],
        backstory=agents_cfg["evaluator_agent"]["backstory"] + "\n\n" + prompts["evaluator"],
    )
    feedback = Agent(
        **agents_cfg["feedback_agent"],
        llm=llm,
        backstory=agents_cfg["feedback_agent"]["backstory"] + "\n\n" + prompts["feedback"],
    )
    reporter = Agent(
        **agents_cfg["report_agent"],
        llm=llm,
        backstory=agents_cfg["report_agent"]["backstory"] + "\n\n" + prompts["report"],
    )

    planning = Task(description=tasks_cfg["planning_task"]["description"], expected_output=tasks_cfg["planning_task"]["expected_output"], agent=planner)
    coding = Task(description=tasks_cfg["coding_task"]["description"], expected_output=tasks_cfg["coding_task"]["expected_output"], agent=coder, context=[planning])
    execution = Task(description=tasks_cfg["execution_task"]["description"], expected_output=tasks_cfg["execution_task"]["expected_output"], agent=executor, context=[coding])
    evaluation = Task(description=tasks_cfg["evaluation_task"]["description"], expected_output=tasks_cfg["evaluation_task"]["expected_output"], agent=evaluator, context=[planning, coding])
    feedback_t = Task(description=tasks_cfg["feedback_task"]["description"], expected_output=tasks_cfg["feedback_task"]["expected_output"], agent=feedback, context=[evaluation])
    reporting = Task(description=tasks_cfg["reporting_task"]["description"], expected_output=tasks_cfg["reporting_task"]["expected_output"], agent=reporter, context=[planning, evaluation, feedback_t])

    return Crew(
        agents=[planner, coder, executor, evaluator, feedback, reporter],
        tasks=[planning, coding, execution, evaluation, feedback_t, reporting],
        process=Process.sequential,
        verbose=True,
    )


def run_analytics_crew(inputs: dict[str, Any]) -> str:
    """Kick off full multi-agent crew with template inputs."""
    crew = build_analytics_crew()
    return str(crew.kickoff(inputs=inputs))
