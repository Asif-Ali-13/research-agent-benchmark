"""Heuristic decision critic for analytical quality assessment.

This module evaluates the quality of analytical decisions without executing code.
It inspects generated code, task context, dataset summary, and execution output
and produces structured recommendations for improving the analysis workflow.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


class DecisionScoreBreakdown(BaseModel):
    """Score breakdown for each decision-making category."""

    data_understanding: float = Field(ge=0.0, le=10.0)
    preprocessing: float = Field(ge=0.0, le=10.0)
    feature_engineering: float = Field(ge=0.0, le=10.0)
    model_selection: float = Field(ge=0.0, le=10.0)
    evaluation_strategy: float = Field(ge=0.0, le=10.0)
    statistical_validity: float = Field(ge=0.0, le=10.0)
    business_alignment: float = Field(ge=0.0, le=10.0)
    explainability: float = Field(ge=0.0, le=10.0)


class DecisionCriticResult(BaseModel):
    """Structured output from the decision critic."""

    decision_score: float = Field(ge=0.0, le=100.0)
    scores: DecisionScoreBreakdown
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = ""


def _extract_model_names(code: str) -> list[str]:
    patterns = [
        r"RandomForest(?:Classifier|Regressor)?",
        r"GradientBoosting(?:Classifier|Regressor)?",
        r"XGB(?:Classifier|Regressor)?",
        r"LGBM(?:Classifier|Regressor)?",
        r"LogisticRegression",
        r"DecisionTree(?:Classifier|Regressor)?",
        r"LinearRegression",
        r"SVC",
    ]
    models = []
    for pattern in patterns:
        if re.search(pattern, code, re.I):
            models.append(re.search(pattern, code, re.I).group(0))
    return models


def _looks_like_business_task(task_description: str) -> bool:
    business_terms = [
        "churn",
        "fraud",
        "customer",
        "business",
        "revenue",
        "cost",
        "risk",
        "profit",
        "retention",
        "reduce",
        "improve",
        "optimiz",
    ]
    return any(term in task_description.lower() for term in business_terms)


def _looks_like_explainable_task(task_description: str) -> bool:
    explain_terms = ["explain", "shap", "feature importance", "permutation", "lime", "interpretable"]
    return any(term in task_description.lower() for term in explain_terms)


def _looks_like_leakage_task(task_description: str) -> bool:
    terms = ["leak", "leakage", "target leakage", "split", "validation", "overfit", "overfitting"]
    return any(term in task_description.lower() for term in terms)


def criticize_decision(
    code: str,
    task_description: str,
    dataset_summary: str = "",
    execution_stdout: str = "",
) -> DecisionCriticResult:
    """Critique the analytical decisions embedded in the generated solution."""

    code_lower = code.lower()
    task_lower = task_description.lower()
    stdout_lower = execution_stdout.lower()
    dataset_lower = dataset_summary.lower()

    data_score = 4.0
    if any(term in code_lower for term in ["describe", "shape", "info", "head", "value_counts"]):
        data_score += 2.0
    if any(term in code_lower for term in ["isna", "isnull", "fillna", "dropna", "duplicated", "drop_duplicates"]):
        data_score += 1.0
    if any(term in task_lower for term in ["missing", "imbalance", "class", "duplicate", "quality", "eda", "explore"]):
        data_score += 1.0
    if any(term in dataset_lower for term in ["missing values", "missing"]):
        data_score += 1.0
    data_score = min(10.0, data_score)

    preprocessing_score = 4.0
    preprocessing_signals = [
        "fillna",
        "dropna",
        "imput",
        "onehotencoder",
        "get_dummies",
        "standardscaler",
        "minmaxscaler",
        "columntransformer",
        "train_test_split",
        "clip",
        "iqr",
        "quantile",
    ]
    found_preprocessing = [s for s in preprocessing_signals if s in code_lower]
    preprocessing_score += min(4.0, 0.5 * len(found_preprocessing))
    if any(term in code_lower for term in ["train_test_split", "stratify"]):
        preprocessing_score += 1.0
    if any(term in code_lower for term in ["standardscaler", "minmaxscaler"]) and any(
        model in code_lower for model in ["randomforest", "gradientboosting", "xgb", "lgbm"]
    ):
        preprocessing_score -= 1.5
    preprocessing_score = min(10.0, max(0.0, preprocessing_score))

    feature_score = 3.0
    if any(term in code_lower for term in ["corr", "correlation", "drop", "feature", "variance", "selectkbest", "pca"]):
        feature_score += 2.0
    if any(term in code_lower for term in ["permutation", "feature_importance", "importance"]):
        feature_score += 2.0
    if any(term in task_lower for term in ["feature", "engineer", "multicollinearity"]):
        feature_score += 1.0
    feature_score = min(10.0, feature_score)

    model_names = _extract_model_names(code)
    model_score = 4.0
    if model_names:
        model_score += 1.5
    if len(model_names) > 1:
        model_score += 1.5
    if any(term in task_lower for term in ["compare", "benchmark", "which", "why"]):
        model_score += 1.0
    if any(model in code_lower for model in ["randomforest", "gradientboosting", "xgb", "lgbm"]):
        model_score += 1.0
    if not model_names:
        model_score -= 1.0
    model_score = min(10.0, max(0.0, model_score))

    evaluation_score = 4.0
    if any(term in stdout_lower for term in ["accuracy", "precision", "recall", "f1", "rmse", "mae", "r2", "auc", "roc"]):
        evaluation_score += 2.0
    if any(term in task_lower for term in ["precision", "recall", "accuracy", "rmse", "mae", "r2", "auc", "f1"]):
        evaluation_score += 1.5
    if any(term in task_lower for term in ["churn", "fraud", "business", "improve", "reduce", "risk"]):
        evaluation_score += 1.0
    evaluation_score = min(10.0, evaluation_score)

    statistical_score = 4.0
    if any(term in code_lower for term in ["train_test_split", "stratify"]):
        statistical_score += 2.0
    if any(term in code_lower for term in ["dropna", "fillna", "imput"]) and "train_test_split" in code_lower:
        statistical_score += 1.0
    if any(term in task_lower for term in ["leak", "leakage", "validation", "overfit"]):
        statistical_score += 1.0
    if any(term in code_lower for term in ["target", "y"]):
        statistical_score += 0.5
    if any(term in code_lower for term in ["drop(columns=", "drop([", "target"]):
        statistical_score -= 0.5
    statistical_score = min(10.0, max(0.0, statistical_score))

    business_score = 4.0
    if _looks_like_business_task(task_description):
        business_score += 2.0
    if any(term in stdout_lower for term in ["recall", "precision", "f1", "auc", "rmse", "mae", "r2"]):
        business_score += 1.0
    if task_lower.count("churn") or task_lower.count("fraud"):
        business_score += 1.0
    if any(term in task_lower for term in ["accuracy", "class", "predict"]):
        business_score -= 0.5
    business_score = min(10.0, max(0.0, business_score))

    explainability_score = 2.0
    if any(term in code_lower for term in ["shap", "permutation", "feature_importance", "lime", "explain"]):
        explainability_score += 4.0
    if _looks_like_explainable_task(task_description):
        explainability_score += 2.0
    if any(term in code_lower for term in ["feature_importance", "permutation"]):
        explainability_score += 1.5
    explainability_score = min(10.0, explainability_score)

    scores = DecisionScoreBreakdown(
        data_understanding=round(data_score, 1),
        preprocessing=round(preprocessing_score, 1),
        feature_engineering=round(feature_score, 1),
        model_selection=round(model_score, 1),
        evaluation_strategy=round(evaluation_score, 1),
        statistical_validity=round(statistical_score, 1),
        business_alignment=round(business_score, 1),
        explainability=round(explainability_score, 1),
    )

    decision_score = round(sum(scores.model_dump().values()) / 8.0 * 10.0, 1)

    strengths = []
    weaknesses = []
    recommendations: list[str] = []

    if scores.data_understanding >= 7:
        strengths.append("The workflow shows evidence of data exploration and quality checks.")
    else:
        weaknesses.append("The analysis does not clearly demonstrate strong data understanding.")
        recommendations.append("Inspect the target variable, missing values, class balance, and data quality before modeling.")

    if scores.preprocessing >= 7:
        strengths.append("Preprocessing choices are reasonably aligned with the task.")
    else:
        weaknesses.append("Preprocessing decisions are weak or inconsistent with the selected model.")
        recommendations.append("Refine missing value handling, scaling, encoding, and leakage prevention.")

    if scores.model_selection >= 7:
        strengths.append("The selected model appears appropriate for the task.")
    else:
        weaknesses.append("Model choice is under-justified or could be improved.")
        recommendations.append("Compare at least one alternative model and explain why the chosen model is preferred.")

    if scores.evaluation_strategy < 7:
        weaknesses.append("The evaluation strategy may not align with the business objective.")
        recommendations.append("Choose metrics that match the task, such as recall for churn or precision-recall for fraud.")

    if scores.business_alignment < 7:
        weaknesses.append("The decision-making process does not clearly reflect the business objective.")
        recommendations.append("Tie the evaluation back to the business goal and prioritize the most relevant metrics.")

    if scores.explainability < 7:
        weaknesses.append("The workflow lacks clear explainability steps.")
        recommendations.append("Add feature importance, permutation importance, SHAP, or LIME where appropriate.")

    if _looks_like_leakage_task(task_description) and scores.statistical_validity < 7:
        weaknesses.append("Potential leakage or weak validation strategy was detected.")
        recommendations.append("Use a proper train/test split, avoid target leakage, and validate carefully.")

    if not strengths:
        strengths.append("The workflow attempted to solve the requested analytical task.")

    if not weaknesses:
        weaknesses.append("The current decision path is generally reasonable but can still be strengthened.")

    summary = (
        "Decision critique completed with a balanced view of data understanding, preprocessing, "
        "model choice, evaluation quality, and business alignment."
    )

    confidence = min(0.98, 0.6 + (decision_score / 100.0) * 0.35)
    return DecisionCriticResult(
        decision_score=round(decision_score, 1),
        scores=scores,
        strengths=strengths,
        weaknesses=weaknesses,
        recommendations=recommendations,
        confidence=round(confidence, 2),
        summary=summary,
    )
