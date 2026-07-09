"""Rule-based context-aware task qualification."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from research_agent.evaluation.scoring import (
    dataset_has_missing_values,
    detect_preprocessing,
    detect_visualization,
    imputation_likely_failed,
    task_requires_preprocessing,
    task_requires_visualization,
)
from research_agent.models.result_schema import ExecutionResult, ExecutionStatus, FeedbackItem
from research_agent.utils.helpers import get_env_bool, get_env_float, load_env

_STDERR_WARNING_MARKERS = (
    "Warning:",
    "Pandas4Warning",
    "FutureWarning",
    "DeprecationWarning",
    "UserWarning",
)


@dataclass
class TaskQualificationResult:
    qualified: bool
    score: float
    gaps: list[FeedbackItem] = field(default_factory=list)
    summary: str = ""


def _require_clean_stderr() -> bool:
    load_env()
    return os.getenv("REQUIRE_CLEAN_STDERR", "false").lower() in ("1", "true", "yes")


def qualification_pass_threshold() -> float:
    load_env()
    return get_env_float("QUALIFICATION_PASS_THRESHOLD", 0.8)


def _qualification_pass_threshold() -> float:
    return qualification_pass_threshold()


def _fail_on_metric_worsened() -> bool:
    load_env()
    return get_env_bool("QUALIFICATION_FAIL_ON_METRIC_WORSENED", True)


def execution_has_stderr_warnings(stderr: str) -> bool:
    if not stderr.strip():
        return False
    return any(marker in stderr for marker in _STDERR_WARNING_MARKERS)


def _stdout_has_metric(stdout: str, metric: str) -> bool:
    return bool(re.search(rf"\b{re.escape(metric)}\b", stdout, re.I))


def _expected_metrics(expected_outputs: list[str] | None) -> list[str]:
    if not expected_outputs:
        return []
    mapping = {
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "rmse": "rmse",
        "r2": "r2",
        "auc": "auc",
    }
    found: list[str] = []
    for item in expected_outputs:
        key = item.lower().strip()
        if key in mapping:
            found.append(mapping[key])
    return found


def _task_requires_compare_before_after(task_description: str) -> bool:
    return bool(
        re.search(
            r"before\s+vs\s+after|compare\s+rmse|compare.*before|rmse\s+before",
            task_description,
            re.I,
        )
    )


def _compare_before_after_reported(stdout: str) -> bool:
    has_before = bool(re.search(r"before|initial", stdout, re.I))
    has_after = bool(re.search(r"after|clipped|worsened|improved", stdout, re.I))
    has_rmse = _stdout_has_metric(stdout, "rmse")
    return has_before and has_after and has_rmse


def _task_involves_outlier_clipping(task_description: str) -> bool:
    return bool(re.search(r"outlier|iqr|clip", task_description, re.I))


def _clips_on_full_dataset_before_split(code: str) -> bool:
    """Detect IQR/clip on full dataframe before train_test_split (data leakage risk)."""
    split_match = re.search(r"train_test_split\s*\(", code)
    if not split_match:
        return False
    split_pos = split_match.start()
    pre_split = code[:split_pos]
    if re.search(r"X_train\s*\[.*\]\.quantile|X_train\s*\[.*\]\.clip", pre_split, re.I):
        return False
    if re.search(r"X_train.*quantile|X_train.*\.clip", code, re.I):
        return False
    clips_full_df = bool(
        re.search(
            r"df(_clipped)?\s*\[[^\]]+\]\s*=\s*[^\n]*\.clip|"
            r"df(_clipped)?\s*\[[^\]]+\]\s*\.quantile|"
            r"df(_clipped)?\['[^']+'\]\.quantile",
            pre_split,
            re.I,
        )
    )
    return clips_full_df


def _detect_pandas_api_issues(code: str) -> list[FeedbackItem]:
    items: list[FeedbackItem] = []
    if re.search(r"\.drop\s*\(\s*columns\s*=", code) and re.search(r"axis\s*=\s*1", code):
        items.append(
            FeedbackItem(
                category="code_quality",
                severity="medium",
                message="pandas drop() uses both columns= and axis=1 (invalid in pandas 3.x).",
                recommendation="Use df.drop(columns=[...], inplace=True) without axis=1.",
            )
        )
    if re.search(r"\.fillna\s*\([^)]*inplace\s*=\s*True", code):
        items.append(
            FeedbackItem(
                category="code_quality",
                severity="medium",
                message="Inplace fillna detected; may cause chained assignment issues.",
                recommendation="Use df['col'] = df['col'].fillna(value) and verify counts.",
            )
        )
    if re.search(r"select_dtypes\s*\(\s*include\s*=\s*['\"]object['\"]", code):
        items.append(
            FeedbackItem(
                category="code_quality",
                severity="low",
                message="select_dtypes(include='object') triggers pandas 3 deprecation warnings.",
                recommendation="Use include=['object', 'string'] or exclude= for string columns.",
            )
        )
    return items


def score_from_gaps(gaps: list[FeedbackItem]) -> float:
    score = 1.0
    for gap in gaps:
        if gap.severity == "high":
            score -= 0.4
        elif gap.severity == "medium":
            score -= 0.15
        else:
            score -= 0.05
    return max(round(score, 3), 0.0)


def qualify_task(
    *,
    code: str,
    execution: ExecutionResult | None,
    task_description: str,
    expected_outputs: list[str] | None = None,
    dataset_summary: str = "",
    visualization_present: bool = False,
    mode: str = "baseline",
) -> TaskQualificationResult:
    """Apply rule-based checks against task requirements."""
    gaps: list[FeedbackItem] = []
    stdout = execution.stdout if execution else ""

    if execution is None or execution.status != ExecutionStatus.SUCCESS:
        gaps.append(
            FeedbackItem(
                category="execution",
                severity="high",
                message="Code did not execute successfully.",
                recommendation="Fix runtime errors and re-run.",
            )
        )
    elif execution_has_stderr_warnings(execution.stderr):
        severity = "high" if _require_clean_stderr() else "medium"
        gaps.append(
            FeedbackItem(
                category="execution",
                severity=severity,
                message=f"Stderr contains warnings: {execution.stderr[:300]}",
                recommendation="Fix deprecation warnings and pandas API usage.",
            )
        )

    needs_viz = task_requires_visualization(task_description) or any(
        o.lower() in ("png", "jpg", "svg") for o in (expected_outputs or [])
    )
    if needs_viz and not visualization_present and not detect_visualization(code, execution):
        gaps.append(
            FeedbackItem(
                category="visualization",
                severity="high",
                message="Task requires a saved visualization but none was detected.",
                recommendation="Save plot to FIGURES_DIR with plt.savefig(...).",
            )
        )

    for metric in _expected_metrics(expected_outputs):
        if not _stdout_has_metric(stdout, metric):
            gaps.append(
                FeedbackItem(
                    category="metrics",
                    severity="high",
                    message=f"Expected metric '{metric}' not found in stdout.",
                    recommendation=f"Print {metric} clearly in the output.",
                )
            )

    if imputation_likely_failed(stdout, task_description):
        gaps.append(
            FeedbackItem(
                category="preprocessing",
                severity="high",
                message="Imputation did not reduce missing value counts.",
                recommendation="Verify before/after missing counts in stdout.",
            )
        )

    if _task_requires_compare_before_after(task_description):
        if not _compare_before_after_reported(stdout):
            gaps.append(
                FeedbackItem(
                    category="completeness",
                    severity="medium",
                    message="Before/after comparison not clearly reported in stdout.",
                    recommendation="Print RMSE (or metric) before and after the transformation.",
                )
            )
        elif re.search(r"worsened|worse", stdout, re.I):
            strict_feedback = mode == "feedback_loop" and _fail_on_metric_worsened()
            gaps.append(
                FeedbackItem(
                    category="analysis",
                    severity="high" if strict_feedback else "medium",
                    message="Metric worsened after transformation; review clipping or methodology.",
                    recommendation=(
                        "Compute IQR bounds on the training set only, clip train and test "
                        "separately, or try alternative outlier handling."
                    ),
                )
            )

    if (
        mode == "feedback_loop"
        and _task_involves_outlier_clipping(task_description)
        and _clips_on_full_dataset_before_split(code)
    ):
        gaps.append(
            FeedbackItem(
                category="methodology",
                severity="medium",
                message="Outlier bounds appear computed on the full dataset before train/test split.",
                recommendation=(
                    "Split first, compute Q1/Q3/IQR on X_train only, then clip "
                    "X_train and X_test with those bounds."
                ),
            )
        )

    if re.search(r"engineer|feature engineering", task_description, re.I):
        if not re.search(r"engineer|avg_|per_|ratio|_feat|rooms_per", code, re.I):
            gaps.append(
                FeedbackItem(
                    category="feature_engineering",
                    severity="medium",
                    message="Limited evidence of engineered features in code.",
                    recommendation="Create and use explicit engineered columns.",
                )
            )

    needs_preprocess = task_requires_preprocessing(task_description) or (
        dataset_has_missing_values(dataset_summary) and "imput" in task_description.lower()
    )
    if needs_preprocess and not detect_preprocessing(code):
        gaps.append(
            FeedbackItem(
                category="preprocessing",
                severity="medium",
                message="Task requires preprocessing but little evidence in code.",
                recommendation="Add imputation, encoding, or scaling as required.",
            )
        )

    gaps.extend(_detect_pandas_api_issues(code))

    score = score_from_gaps(gaps)
    has_high = any(g.severity == "high" for g in gaps)
    threshold = _qualification_pass_threshold()
    qualified = not has_high and score >= threshold and (
        execution is not None and execution.status == ExecutionStatus.SUCCESS
    )
    summary = (
        "Task requirements met."
        if qualified
        else f"Qualification gaps ({len(gaps)}); score={score:.2f}."
    )
    return TaskQualificationResult(qualified=qualified, score=score, gaps=gaps, summary=summary)
