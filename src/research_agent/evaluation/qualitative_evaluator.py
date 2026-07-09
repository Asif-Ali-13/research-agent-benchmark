"""Optional LLM-based task qualification layer."""

from __future__ import annotations

import json
import re

from research_agent.crew import get_llm
from research_agent.evaluation.task_qualification import TaskQualificationResult
from research_agent.models.result_schema import FeedbackItem
from research_agent.utils.helpers import get_project_root, truncate_text
from research_agent.utils.llm_retry import call_with_retry
from research_agent.utils.logger import setup_logger

logger = setup_logger("qualitative_evaluator")


def _parse_llm_feedback(text: str) -> list[FeedbackItem]:
    """Parse JSON or markdown-style gap list from LLM response."""
    items: list[FeedbackItem] = []
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            for gap in data.get("gaps", []):
                items.append(
                    FeedbackItem(
                        category=str(gap.get("category", "general")),
                        severity=str(gap.get("severity", "medium")),
                        message=str(gap.get("message", "")),
                        recommendation=str(gap.get("recommendation", "")),
                    )
                )
            return items
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    for line in text.splitlines():
        match = re.match(
            r"[-*]\s*\[(high|medium|low)\]\s*(\w+):\s*(.+?)(?:\s*→\s*(.+))?$",
            line.strip(),
            re.I,
        )
        if match:
            items.append(
                FeedbackItem(
                    category=match.group(2),
                    severity=match.group(1).lower(),
                    message=match.group(3).strip(),
                    recommendation=(match.group(4) or "").strip(),
                )
            )
    return items


def _merge_qualification(
    rule_result: TaskQualificationResult,
    llm_gaps: list[FeedbackItem],
    llm_score: float | None = None,
    llm_summary: str = "",
) -> TaskQualificationResult:
    seen = {(g.category, g.message) for g in rule_result.gaps}
    merged = list(rule_result.gaps)
    for gap in llm_gaps:
        key = (gap.category, gap.message)
        if key not in seen:
            merged.append(gap)
            seen.add(key)

    from research_agent.evaluation.task_qualification import qualification_pass_threshold, score_from_gaps

    score = llm_score if llm_score is not None else score_from_gaps(merged)
    has_high = any(g.severity == "high" for g in merged)
    threshold = qualification_pass_threshold()
    qualified = not has_high and score >= threshold
    summary = llm_summary or rule_result.summary
    return TaskQualificationResult(qualified=qualified, score=score, gaps=merged, summary=summary)


def qualify_task_llm(
    *,
    query: str,
    plan: str,
    stdout: str,
    stderr: str,
    output_files: list[str],
    expected_outputs: list[str] | None,
    rule_result: TaskQualificationResult,
) -> TaskQualificationResult:
    """Augment rule-based qualification with LLM assessment."""
    prompt_path = get_project_root() / "prompts" / "evaluator_prompt.txt"
    system_prompt = prompt_path.read_text(encoding="utf-8")

    user_prompt = (
        f"Query: {truncate_text(query, 1500)}\n"
        f"Plan: {truncate_text(plan, 2000)}\n"
        f"Expected outputs: {expected_outputs or []}\n"
        f"Stdout:\n{truncate_text(stdout, 2500)}\n"
        f"Stderr:\n{truncate_text(stderr, 1000)}\n"
        f"Output files: {output_files}\n"
        f"Rule-based qualification score: {rule_result.score:.2f}\n"
        f"Rule-based gaps: {[g.model_dump() for g in rule_result.gaps]}\n"
        "Respond with JSON only: "
        '{"score": 0.0-1.0, "summary": "...", "gaps": [{"category": "...", '
        '"severity": "low|medium|high", "message": "...", "recommendation": "..."}]}'
    )

    def _kickoff() -> str:
        from crewai import Agent, Crew, Process, Task

        llm = get_llm()
        agent = Agent(role="Evaluator", goal=system_prompt, backstory=system_prompt, llm=llm, verbose=False)
        task = Task(description=user_prompt, expected_output="JSON assessment", agent=agent)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        return str(crew.kickoff())

    try:
        raw = call_with_retry(_kickoff)
        llm_gaps = _parse_llm_feedback(raw)
        json_match = re.search(r"\{[\s\S]*\}", raw)
        llm_score: float | None = None
        llm_summary = ""
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                llm_score = float(data.get("score", rule_result.score))
                llm_summary = str(data.get("summary", ""))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return _merge_qualification(rule_result, llm_gaps, llm_score, llm_summary)
    except Exception as exc:
        logger.warning("LLM qualification failed, using rule-based only: %s", exc)
        return rule_result
