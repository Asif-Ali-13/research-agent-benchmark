"""Tests for agent flows (mocked / structural)."""

from __future__ import annotations

from pathlib import Path

import yaml

from research_agent.utils.helpers import get_project_root


def test_agent_yaml_configs_exist() -> None:
    root = get_project_root()
    analytics_agents = root / "src/research_agent/crews/analytics_crew/config/agents.yaml"
    analytics_tasks = root / "src/research_agent/crews/analytics_crew/config/tasks.yaml"
    eval_agents = root / "src/research_agent/crews/evaluation_crew/config/agents.yaml"

    assert analytics_agents.exists()
    assert analytics_tasks.exists()
    assert eval_agents.exists()

    agents = yaml.safe_load(analytics_agents.read_text())
    assert "planner_agent" in agents
    assert "code_agent" in agents
    assert "report_agent" in agents


def test_prompts_exist() -> None:
    root = get_project_root() / "prompts"
    for name in ["planner_prompt.txt", "evaluator_prompt.txt", "feedback_prompt.txt", "report_prompt.txt"]:
        assert (root / name).exists()


def test_flows_importable() -> None:
    from research_agent.flows.feedback_flow import FeedbackFlow
    from research_agent.flows.reliability_flow import ReliabilityFlow

    assert FeedbackFlow is not None
    assert ReliabilityFlow is not None
