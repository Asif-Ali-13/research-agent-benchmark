"""Heuristic scoring for evaluator agent and automated checks."""

from __future__ import annotations

import ast
import os
import re

from research_agent.evaluation.decision_critic import criticize_decision
from research_agent.models.result_schema import (
    EvaluationResult,
    ExecutionResult,
    ExecutionStatus,
    FeedbackItem,
    StructuredFeedback,
)
from research_agent.utils.helpers import get_env_bool, get_env_float, load_env


def execution_has_critical_stderr(stderr: str) -> bool:
    if not stderr.strip():
        return False
    markers = (
        "ChainedAssignmentError",
        "Traceback (most recent call last)",
        "ValueError:",
        "KeyError:",
        "ParserError:",
    )
    return any(m in stderr for m in markers)


def dataset_has_missing_values(dataset_summary: str) -> bool:
    """Parse missing-value counts from load_dataset_summary() output."""
    match = re.search(r"Missing values:\s*(\{.*\})", dataset_summary)
    if not match:
        return False
    try:
        missing = ast.literal_eval(match.group(1))
        if isinstance(missing, dict):
            return any(int(v) > 0 for v in missing.values())
    except (SyntaxError, ValueError, TypeError):
        pass
    return False


def imputation_likely_failed(stdout: str, task_description: str) -> bool:
    """Detect when imputation was requested but counts did not decrease."""
    if not re.search(r"imput|fillna|fill\s+missing", task_description, re.I):
        return False
    before = re.search(r"Missing.*\(initial\):\s*(\d+)", stdout, re.I)
    after = re.search(r"Missing.*\(after[^)]*\):\s*(\d+)", stdout, re.I)
    if before and after:
        return int(before.group(1)) > 0 and int(before.group(1)) == int(after.group(1))
    return False


def score_execution(execution: ExecutionResult | None) -> float:
    if execution is None:
        return 0.0
    if execution.status == ExecutionStatus.SUCCESS:
        if execution_has_critical_stderr(execution.stderr):
            return 0.55
        return 1.0
    if execution.status == ExecutionStatus.TIMEOUT:
        return 0.2
    if execution.status == ExecutionStatus.BLOCKED:
        return 0.0
    return 0.4


def score_output_completeness(
    execution: ExecutionResult | None,
    expected_outputs: list[str] | None = None,
) -> float:
    if execution is None:
        return 0.0
    score = 0.3
    if execution.stdout.strip():
        score += 0.3
    if execution.output_files:
        score += 0.2
    if expected_outputs:
        found = sum(1 for o in expected_outputs if any(o in f for f in execution.output_files))
        score += 0.2 * (found / max(len(expected_outputs), 1))
    return min(score, 1.0)


def task_requires_visualization(task_description: str) -> bool:
    return bool(
        re.search(
            r"plot|chart|figure|visualiz|heatmap|pairplot|hist|bar chart|savefig|save figure",
            task_description,
            re.I,
        )
    )


def task_requires_preprocessing(task_description: str) -> bool:
    return bool(
        re.search(
            r"imput|encode|scale|preprocess|engineer|one-?hot|fill\s+missing|handle\s+missing",
            task_description,
            re.I,
        )
    )


def task_requires_ml_metrics(task_description: str) -> bool:
    return bool(
        re.search(
            r"accuracy|f1|rmse|r2|precision|recall|classif|regress|predict|model|train",
            task_description,
            re.I,
        )
    )


def detect_visualization(code: str, execution: ExecutionResult | None) -> bool:
    if execution and execution.output_files:
        if any(f.lower().endswith((".png", ".html", ".jpg", ".jpeg", ".svg")) for f in execution.output_files):
            return True
    viz_patterns = [r"plt\.savefig", r"\.savefig\(", r"write_image", r"to_html"]
    return any(re.search(p, code, re.I) for p in viz_patterns)


def detect_preprocessing(code: str) -> bool:
    patterns = [
        r"imput",
        r"fillna",
        r"OneHotEncoder",
        r"get_dummies",
        r"StandardScaler",
        r"ColumnTransformer",
        r"preprocess",
    ]
    return any(re.search(p, code, re.I) for p in patterns)


def detect_ml_metrics(stdout: str) -> bool:
    patterns = [r"accuracy", r"f1", r"r2", r"rmse", r"precision", r"recall", r"auc"]
    return any(re.search(p, stdout, re.I) for p in patterns)


def _feedback_pass_threshold() -> float:
    load_env()
    return get_env_float("FEEDBACK_PASS_THRESHOLD", 0.75)


def _should_use_llm_qualification(mode: str, override: bool | None) -> bool:
    if override is not None:
        return override
    load_env()
    if mode == "feedback_loop":
        return get_env_bool("USE_LLM_QUALIFICATION", True)
    return get_env_bool("QUALIFICATION_IN_BASELINE", False)


def _merge_gap_items(existing: list[FeedbackItem], new_gaps: list[FeedbackItem]) -> list[FeedbackItem]:
    seen = {(g.category, g.message) for g in existing}
    merged = list(existing)
    for gap in new_gaps:
        key = (gap.category, gap.message)
        if key not in seen:
            merged.append(gap)
            seen.add(key)
    return merged


def build_structured_feedback(
    iteration: int,
    evaluation: EvaluationResult,
    execution: ExecutionResult | None,
    code: str,
    task_description: str = "",
    dataset_summary: str = "",
    decision_evaluation=None,
) -> StructuredFeedback:
    items: list[FeedbackItem] = []
    score = evaluation.task_completion_score

    if execution and execution.status != ExecutionStatus.SUCCESS:
        items.append(
            FeedbackItem(
                category="execution",
                severity="high",
                message=f"Execution failed: {execution.error_message or execution.stderr[:500]}",
                recommendation="Fix runtime errors, verify imports and column names, add try/except.",
            )
        )
        score = min(score, 0.4)
    elif execution and execution_has_critical_stderr(execution.stderr):
        items.append(
            FeedbackItem(
                category="execution",
                severity="high",
                message=f"Runtime warnings/errors in stderr: {execution.stderr[:400]}",
                recommendation=(
                    "Avoid pandas inplace chained assignment; use df['col'] = df['col'].fillna(value)."
                ),
            )
        )
        score = min(score, 0.5)

    if execution and imputation_likely_failed(execution.stdout, task_description):
        items.append(
            FeedbackItem(
                category="preprocessing",
                severity="high",
                message="Missing values were not reduced after imputation (possible inplace fillna bug).",
                recommendation="Use df['Age'] = df['Age'].fillna(median) and verify missing count is zero.",
            )
        )
        score = min(score, 0.45)

    if task_requires_visualization(task_description) and not evaluation.visualization_present:
        items.append(
            FeedbackItem(
                category="visualization",
                severity="medium",
                message="No visualization artifacts detected.",
                recommendation=(
                    "Save plots with plt.savefig(os.path.join(os.environ['FIGURES_DIR'], 'name.png'))."
                ),
            )
        )

    if not evaluation.output_completeness >= 0.75:
        items.append(
            FeedbackItem(
                category="completeness",
                severity="medium",
                message="Output appears incomplete relative to task requirements.",
                recommendation="Ensure EDA summary, metrics printout, and saved outputs.",
            )
        )

    if decision_evaluation is not None:
        if decision_evaluation.decision_score < 70:
            items.append(
                FeedbackItem(
                    category="decision",
                    severity="medium",
                    message=(
                        f"Decision quality is only {decision_evaluation.decision_score:.1f}/100; "
                        "the analytical choices need stronger justification."
                    ),
                    recommendation="Refine the analysis plan with stronger data understanding, preprocessing, model comparison, and business-focused metrics.",
                )
            )
        if decision_evaluation.scores.business_alignment < 7:
            items.append(
                FeedbackItem(
                    category="business",
                    severity="medium",
                    message="The chosen approach does not clearly align with the stated business objective.",
                    recommendation="Reframe the solution around business impact and the primary success metric for the task.",
                )
            )
        if decision_evaluation.scores.explainability < 7:
            items.append(
                FeedbackItem(
                    category="explainability",
                    severity="low",
                    message="The workflow does not provide enough explanation of why predictions or conclusions are made.",
                    recommendation="Add feature importance, permutation importance, SHAP, or LIME where appropriate.",
                )
            )

    needs_preprocess = task_requires_preprocessing(task_description) or (
        dataset_has_missing_values(dataset_summary)
        and task_requires_ml_metrics(task_description)
    )
    if needs_preprocess and not detect_preprocessing(code):
        items.append(
            FeedbackItem(
                category="preprocessing",
                severity="low",
                message="Limited evidence of preprocessing steps.",
                recommendation="Add missing value handling and feature scaling where needed.",
            )
        )

    passed = score >= 0.75 and (execution is None or execution.status == ExecutionStatus.SUCCESS)
    summary = "Task meets quality threshold." if passed else "Revisions recommended before final report."

    return StructuredFeedback(
        iteration=iteration,
        score=round(score, 3),
        passed=passed,
        items=items,
        summary=summary,
    )


def evaluate_run(
    code: str,
    execution: ExecutionResult | None,
    task_description: str,
    expected_outputs: list[str] | None = None,
    iteration: int = 0,
    dataset_summary: str = "",
    mode: str = "baseline",
    plan: str = "",
    use_llm_qualification: bool | None = None,
) -> tuple[EvaluationResult, StructuredFeedback]:
    exec_score = score_execution(execution)
    completeness = score_output_completeness(execution, expected_outputs)
    viz = detect_visualization(code, execution)
    ml_ok = detect_ml_metrics(execution.stdout if execution else "")
    needs_viz = task_requires_visualization(task_description)
    needs_ml = task_requires_ml_metrics(task_description)
    viz_score = 1.0 if (viz or not needs_viz) else 0.0
    ml_score = 1.0 if (ml_ok or not needs_ml) else 0.0

    if imputation_likely_failed(execution.stdout if execution else "", task_description):
        exec_score = min(exec_score, 0.45)

    task_score = (
        0.4 * exec_score
        + 0.3 * completeness
        + 0.15 * viz_score
        + 0.15 * ml_score
    )

    decision_evaluation = criticize_decision(
        code=code,
        task_description=task_description,
        dataset_summary=dataset_summary,
        execution_stdout=execution.stdout if execution else "",
    )

    evaluation = EvaluationResult(
        task_completion_score=round(task_score, 3),
        execution_success=exec_score >= 1.0,
        output_completeness=completeness,
        visualization_present=viz,
        ml_metrics_valid=ml_ok,
        notes=f"Evaluated for: {task_description[:200]}",
        decision_evaluation=decision_evaluation,
        decision_quality_score=decision_evaluation.decision_score,
    )

    feedback = build_structured_feedback(
        iteration,
        evaluation,
        execution,
        code,
        task_description,
        dataset_summary,
        decision_evaluation=decision_evaluation,
    )

    from research_agent.evaluation.task_qualification import qualify_task

    qualification = qualify_task(
        code=code,
        execution=execution,
        task_description=task_description,
        expected_outputs=expected_outputs,
        dataset_summary=dataset_summary,
        visualization_present=viz,
        mode=mode,
    )

    if _should_use_llm_qualification(mode, use_llm_qualification):
        from research_agent.evaluation.qualitative_evaluator import qualify_task_llm

        qualification = qualify_task_llm(
            query=task_description,
            plan=plan,
            stdout=execution.stdout if execution else "",
            stderr=execution.stderr if execution else "",
            output_files=execution.output_files if execution else [],
            expected_outputs=expected_outputs,
            rule_result=qualification,
        )

    feedback.items = _merge_gap_items(feedback.items, qualification.gaps)
    feedback.qualification_score = qualification.score
    feedback.qualified = qualification.qualified
    feedback.decision_score = decision_evaluation.decision_score
    feedback.decision_summary = decision_evaluation.summary
    feedback.decision_recommendations = decision_evaluation.recommendations
    evaluation.qualification_score = qualification.score
    evaluation.task_qualified = qualification.qualified
    evaluation.notes = f"{evaluation.notes} | {qualification.summary}"

    if mode == "feedback_loop":
        heuristic_threshold = _feedback_pass_threshold()
        exec_ok = execution is not None and execution.status == ExecutionStatus.SUCCESS
        decision_ok = (
            decision_evaluation.decision_score >= 70
            or not decision_evaluation.recommendations
        )
        feedback.passed = (
            exec_ok
            and evaluation.task_completion_score >= heuristic_threshold
            and qualification.qualified
            and decision_ok
        )
        feedback.summary = (
            "Task meets heuristic and qualification thresholds."
            if feedback.passed
            else "Revisions needed: heuristic or qualification requirements not met."
        )
    elif get_env_bool("QUALIFICATION_IN_BASELINE", False):
        load_env()
        feedback.qualified = qualification.qualified
        feedback.qualification_score = qualification.score

    evaluation.structured_feedback = feedback
    return evaluation, feedback
