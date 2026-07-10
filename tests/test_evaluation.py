"""Tests for evaluation metrics and scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_agent.evaluation.decision_critic import (
    DecisionCriticResult,
    DecisionScoreBreakdown,
    criticize_decision,
)
from research_agent.evaluation.reliability_metrics import ReliabilityMetrics, compute_run_level_metrics
from research_agent.evaluation.scoring import (
    dataset_has_missing_values,
    evaluate_run,
    imputation_likely_failed,
    task_requires_preprocessing,
    task_requires_visualization,
)
from research_agent.evaluation.task_qualification import (
    execution_has_stderr_warnings,
    qualify_task,
)
from research_agent.memory.feedback_memory import FeedbackMemory

from research_agent.models.result_schema import (
    AgentRunResult,
    EvaluationResult,
    ExecutionResult,
    ExecutionStatus,
    StructuredFeedback,
)


def test_reliability_metrics_empty() -> None:
    m = ReliabilityMetrics.from_results([])
    assert m.task_completion_rate == 0.0


def test_reliability_metrics_from_results() -> None:
    results = [
        AgentRunResult(
            run_id="1",
            mode="feedback",
            query="q",
            dataset_path="d.csv",
            execution=ExecutionResult(status=ExecutionStatus.SUCCESS),
            evaluation=None,
            feedback_history=[
                StructuredFeedback(iteration=0, score=0.5, passed=False, summary=""),
                StructuredFeedback(iteration=1, score=0.8, passed=True, summary=""),
            ],
        )
    ]
    m = ReliabilityMetrics.from_results(results)
    assert m.total_tasks == 1
    assert m.feedback_improvement_score == pytest.approx(0.3, abs=0.01)


def test_reliability_metrics_summary_includes_extended_metrics() -> None:
    result = AgentRunResult(
        run_id="1",
        mode="feedback",
        query="q",
        dataset_path="d.csv",
        execution=ExecutionResult(status=ExecutionStatus.SUCCESS),
        evaluation=None,
    )
    m = ReliabilityMetrics.from_results([result])
    summary = m.summary()
    assert "DQS" in summary
    assert "BAS" in summary
    assert "ORS" in summary


def test_compute_run_level_metrics_uses_feedback_improvement_definition() -> None:
    result = AgentRunResult(
        run_id="1",
        mode="feedback",
        query="q",
        dataset_path="d.csv",
        evaluation=EvaluationResult(task_completion_score=0.8, execution_success=True, output_completeness=1.0),
        feedback_history=[
            StructuredFeedback(iteration=0, score=0.4, passed=False, summary=""),
            StructuredFeedback(iteration=1, score=0.7, passed=True, summary=""),
        ],
    )
    metrics = compute_run_level_metrics(result)
    assert metrics["fis"] == pytest.approx(0.3, abs=0.01)


def test_compute_run_level_metrics_uses_decision_critic_for_da() -> None:
    result = AgentRunResult(
        run_id="1",
        mode="feedback",
        query="q",
        dataset_path="d.csv",
        evaluation=EvaluationResult(
            task_completion_score=0.2,
            execution_success=True,
            output_completeness=1.0,
            decision_evaluation=DecisionCriticResult(
                decision_score=76.7,
                scores=DecisionScoreBreakdown(
                    data_understanding=8.0,
                    preprocessing=8.0,
                    feature_engineering=7.0,
                    model_selection=6.0,
                    evaluation_strategy=10.0,
                    statistical_validity=8.0,
                    business_alignment=6.0,
                    explainability=7.0,
                ),
                confidence=0.82,
                summary="ok",
            ),
        ),
    )
    metrics = compute_run_level_metrics(result)
    assert metrics["da"] == pytest.approx(0.767, abs=0.001)


def test_task_requires_visualization() -> None:
    assert task_requires_visualization("Plot survival rate and save bar chart")
    assert not task_requires_visualization("Print accuracy and classification report")


def test_task_requires_preprocessing_not_triggered_by_eda_missing_report() -> None:
    query = "Print shape, describe(), missing values, and correlation"
    assert not task_requires_preprocessing(query)


def test_dataset_has_missing_values_parser() -> None:
    summary = "Shape: [200, 9]\nMissing values: {'Age': 29, 'Sex': 0}\n"
    assert dataset_has_missing_values(summary)


def test_imputation_likely_failed_detection() -> None:
    stdout = "Missing 'Age' values (initial): 29\nMissing 'Age' values (after imputation): 29\n"
    assert imputation_likely_failed(stdout, "Impute missing Age with median")


def test_eda_scores_without_ml_penalty() -> None:
    code = "import pandas as pd\nprint(df.describe())"
    execution = ExecutionResult(status=ExecutionStatus.SUCCESS, stdout="shape ok")
    evaluation, _ = evaluate_run(
        code,
        execution,
        "Perform EDA: shape, describe(), correlation",
        dataset_summary="Missing values: {'a': 0}",
    )
    assert evaluation.task_completion_score >= 0.85


def test_stderr_chained_assignment_lowers_score() -> None:
    execution = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        stdout="done",
        stderr="ChainedAssignmentError: inplace method never works",
    )
    evaluation, feedback = evaluate_run(
        "impute",
        execution,
        "Handle missing Age with median imputation",
        dataset_summary="Missing values: {'Age': 5}",
    )
    assert evaluation.task_completion_score < 0.75
    assert any(i.severity == "high" for i in feedback.items)


def test_evaluate_run_no_viz_penalty_without_plot_task() -> None:
    code = "from sklearn.metrics import accuracy_score\nprint('accuracy=0.9')"
    execution = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        stdout="accuracy=0.9",
        output_files=["datasets/raw/titanic.csv"],
    )
    evaluation, feedback = evaluate_run(
        code,
        execution,
        "Predict Survived and print accuracy",
        iteration=0,
    )
    viz_items = [i for i in feedback.items if i.category == "visualization"]
    assert not viz_items
    assert evaluation.task_completion_score >= 0.7


def test_evaluate_run_success() -> None:
    code = "import matplotlib.pyplot as plt\nprint('accuracy=0.95')"
    execution = ExecutionResult(status=ExecutionStatus.SUCCESS, stdout="accuracy=0.95")
    evaluation, feedback = evaluate_run(code, execution, "Train classifier", iteration=0)
    assert evaluation.task_completion_score > 0
    assert isinstance(feedback, StructuredFeedback)


def test_decision_critic_scores_business_alignment_and_explainability() -> None:
    result = criticize_decision(
        code=(
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.inspection import permutation_importance\n"
            "model = RandomForestClassifier(random_state=42)"
        ),
        task_description="Reduce churn by predicting customer churn and explain predictions with SHAP",
        dataset_summary="Missing values: {'Age': 5}\n",
        execution_stdout="Recall=0.82\nPrecision=0.79\n",
    )
    assert result.decision_score >= 60
    assert result.scores.business_alignment >= 7
    assert result.scores.explainability >= 7
    assert result.recommendations


def test_evaluate_run_adds_decision_evaluation() -> None:
    execution = ExecutionResult(status=ExecutionStatus.SUCCESS, stdout="Recall=0.82")
    evaluation, feedback = evaluate_run(
        "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier()",
        execution,
        "Reduce churn with explainable predictions",
        iteration=0,
    )
    assert evaluation.decision_evaluation is not None
    assert evaluation.decision_quality_score is not None
    assert feedback.decision_score is not None


def test_load_results_from_benchmark_summary(tmp_path: Path) -> None:
    from research_agent.evaluation.comparison_metrics import load_results_from_file

    summary = {
        "run_id": "20260607_111119",
        "mode": "baseline",
        "task_count": 1,
        "metrics": {"TCR": 1.0},
        "results": [
            {
                "run_id": "645222bb",
                "mode": "baseline",
                "query": "Perform EDA",
                "dataset_path": "datasets/raw/telco_customer_churn.csv",
            }
        ],
    }
    path = tmp_path / "benchmark_20260607_111119.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    results = load_results_from_file(path)
    assert len(results) == 1
    assert results[0].run_id == "645222bb"
    assert results[0].query == "Perform EDA"


def _benchmark_summary(task_id: str, run_id: str, mode: str) -> dict:
    return {
        "run_id": "20260607_120000",
        "mode": mode,
        "task_count": 1,
        "metrics": {"TCR": 1.0},
        "results": [
            {
                "run_id": run_id,
                "mode": mode,
                "query": f"Task {task_id}",
                "dataset_path": "datasets/raw/data.csv",
                "metadata": {"task_id": task_id},
                "execution": {"status": "success"},
            }
        ],
    }


def test_build_output_filename() -> None:
    from research_agent.evaluation.comparison_metrics import build_output_filename

    assert build_output_filename("task_001", "task_001") == "comparison_metrics_task_001.csv"
    assert (
        build_output_filename("task_001", "task_002") == "comparison_metrics_task_001_task_002.csv"
    )
    assert build_output_filename(None, "task_001", path_hash="abcd1234") == (
        "comparison_metrics_unknown_abcd1234.csv"
    )


def test_extract_task_id_from_benchmark() -> None:
    from research_agent.evaluation.comparison_metrics import extract_task_id_from_benchmark

    data = _benchmark_summary("task_001", "abc", "baseline")
    assert extract_task_id_from_benchmark(data) == "task_001"
    assert extract_task_id_from_benchmark({"results": []}) is None


def test_find_benchmark_by_task_id(tmp_path: Path) -> None:
    from research_agent.evaluation.comparison_metrics import find_benchmark_by_task_id

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "benchmark_20260607_100000.json").write_text(
        json.dumps(_benchmark_summary("task_001", "r1", "baseline")),
        encoding="utf-8",
    )
    (baseline / "benchmark_20260607_110000.json").write_text(
        json.dumps(_benchmark_summary("task_001", "r2", "baseline")),
        encoding="utf-8",
    )
    (baseline / "benchmark_20260607_120000.json").write_text(
        json.dumps(_benchmark_summary("task_002", "r3", "baseline")),
        encoding="utf-8",
    )

    found = find_benchmark_by_task_id(baseline, "task_001")
    assert found is not None
    assert found.name == "benchmark_20260607_110000.json"
    assert find_benchmark_by_task_id(baseline, "task_999") is None


def test_compare_final_pairs_intersection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from research_agent.evaluation import comparison_metrics

    baseline = tmp_path / "experiments" / "baseline"
    feedback = tmp_path / "experiments" / "feedback_loop"
    baseline.mkdir(parents=True)
    feedback.mkdir(parents=True)

    (baseline / "benchmark_20260607_100000.json").write_text(
        json.dumps(_benchmark_summary("task_001", "b1", "baseline")),
        encoding="utf-8",
    )
    (baseline / "benchmark_20260607_110000.json").write_text(
        json.dumps(_benchmark_summary("task_002", "b2", "baseline")),
        encoding="utf-8",
    )
    (feedback / "benchmark_20260607_100000.json").write_text(
        json.dumps(_benchmark_summary("task_001", "f1", "feedback_loop")),
        encoding="utf-8",
    )

    monkeypatch.setattr(comparison_metrics, "get_project_root", lambda: tmp_path)

    df, out_path = comparison_metrics.compare_final("experiments/baseline", "experiments/feedback_loop")
    assert out_path.name == "comparison_metrics_final.csv"
    assert len(df) == 8
    assert "TCR" in df["metric"].values


def test_task_014_like_stderr_fails_when_require_clean_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REQUIRE_CLEAN_STDERR", "true")
    execution = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        stdout="RMSE before: 1.2\nRMSE after clipping: 1.5\n",
        stderr="/path: Pandas4Warning: select_dtypes include='object' is deprecated\n",
    )
    result = qualify_task(
        code="df.select_dtypes(include='object')",
        execution=execution,
        task_description="Compare RMSE before vs after clipping outliers",
    )
    assert execution_has_stderr_warnings(execution.stderr)
    assert result.qualified is False
    assert any(g.severity == "high" for g in result.gaps)


def test_imputation_not_qualified_when_counts_unchanged() -> None:
    stdout = "Missing 'Age' values (initial): 29\nMissing 'Age' values (after imputation): 29\n"
    execution = ExecutionResult(status=ExecutionStatus.SUCCESS, stdout=stdout)
    result = qualify_task(
        code="df['Age'].fillna(median, inplace=True)",
        execution=execution,
        task_description="Impute missing Age with median",
    )
    assert result.qualified is False
    assert any(g.category == "preprocessing" for g in result.gaps)


def test_viz_task_without_png_not_qualified() -> None:
    execution = ExecutionResult(status=ExecutionStatus.SUCCESS, stdout="accuracy=0.9")
    result = qualify_task(
        code="print('accuracy')",
        execution=execution,
        task_description="Plot churn rate and save bar chart",
        expected_outputs=["png"],
    )
    assert result.qualified is False
    assert any(g.category == "visualization" for g in result.gaps)


def test_execution_failure_not_qualified() -> None:
    execution = ExecutionResult(
        status=ExecutionStatus.FAILURE,
        stderr="ValueError: bad input",
        error_message="ValueError: bad input",
    )
    result = qualify_task(
        code="raise ValueError('bad input')",
        execution=execution,
        task_description="Train model",
    )
    assert result.qualified is False
    memory = FeedbackMemory()
    memory.add(StructuredFeedback(iteration=0, score=0.3, passed=False, summary="fail"))
    assert memory.should_continue(max_iterations=3) is True


def test_task_014_rmse_worsened_fails_feedback_qualification() -> None:
    """task_014-like: high heuristic but RMSE worsened should not pass feedback loop."""
    code = (
        "df['median_income'].quantile(0.25)\n"
        "df_clipped = df.copy()\n"
        "df_clipped['median_income'] = df_clipped['median_income'].clip(0, 8)\n"
        "train_test_split(df_clipped, y, test_size=0.2, random_state=42)"
    )
    execution = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        stdout=(
            "RMSE before median_income clipping: 48941.7003\n"
            "RMSE after median_income clipping: 49080.3682\n"
            "The RMSE worsened by 138.6678 after clipping median_income outliers.\n"
        ),
        stderr="",
    )
    evaluation, feedback = evaluate_run(
        code,
        execution,
        (
            "Detect outliers in median_income using IQR method. Clip median_income to "
            "[Q1-1.5*IQR, Q3+1.5*IQR], retrain RandomForestRegressor for median_house_value, "
            "and compare RMSE before vs after clipping."
        ),
        expected_outputs=["rmse", "stdout"],
        mode="feedback_loop",
        use_llm_qualification=False,
    )
    assert evaluation.task_completion_score >= 0.75
    assert feedback.qualified is False
    assert feedback.passed is False
    assert any(g.severity == "high" and g.category == "analysis" for g in feedback.items)
    memory = FeedbackMemory()
    memory.add(feedback)
    assert memory.should_continue(max_iterations=3) is True


def test_metric_worsened_medium_severity_in_baseline() -> None:
    execution = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        stdout="RMSE before: 1.0\nRMSE after: 1.5\nRMSE worsened by 0.5\n",
    )
    result = qualify_task(
        code="print('rmse')",
        execution=execution,
        task_description="Compare RMSE before vs after clipping",
        mode="baseline",
    )
    worsened = [g for g in result.gaps if g.category == "analysis"]
    assert worsened
    assert worsened[0].severity == "medium"
    assert result.qualified is True


def test_heuristic_high_qualification_fail_feedback_not_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REQUIRE_CLEAN_STDERR", "true")
    code = "import pandas as pd\nprint('accuracy=0.95')"
    execution = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        stdout="accuracy=0.95",
        stderr="Pandas4Warning: deprecated API",
    )
    evaluation, feedback = evaluate_run(
        code,
        execution,
        "Train classifier and print accuracy",
        expected_outputs=["accuracy"],
        mode="feedback_loop",
        use_llm_qualification=False,
    )
    assert evaluation.task_completion_score >= 0.75
    assert feedback.qualified is False
    assert feedback.passed is False


def test_compare_two_files_writes_task_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from research_agent.evaluation import comparison_metrics

    baseline = tmp_path / "baseline"
    feedback = tmp_path / "feedback"
    baseline.mkdir()
    feedback.mkdir()
    tables = tmp_path / "reports" / "tables"
    tables.mkdir(parents=True)

    baseline_result = AgentRunResult(
        run_id="b1",
        mode="baseline",
        query="Task task_001",
        dataset_path="datasets/raw/data.csv",
        metadata={"task_id": "task_001"},
        execution=ExecutionResult(status=ExecutionStatus.SUCCESS),
        evaluation={
            "task_completion_score": 0.8,
            "execution_success": True,
            "output_completeness": 1.0,
            "visualization_present": False,
            "ml_metrics_valid": False,
            "notes": "",
            "decision_quality_score": 78.0,
            "decision_evaluation": {
                "decision_score": 78.0,
                "scores": {
                    "data_understanding": 8.0,
                    "preprocessing": 7.0,
                    "feature_engineering": 7.5,
                    "model_selection": 8.0,
                    "evaluation_strategy": 7.5,
                    "statistical_validity": 8.0,
                    "business_alignment": 8.0,
                    "explainability": 7.0,
                },
                "strengths": [],
                "weaknesses": [],
                "recommendations": [],
                "confidence": 0.7,
                "summary": "ok",
            },
        },
    )
    feedback_result = AgentRunResult(
        run_id="f1",
        mode="feedback_loop",
        query="Task task_001",
        dataset_path="datasets/raw/data.csv",
        metadata={"task_id": "task_001"},
        execution=ExecutionResult(status=ExecutionStatus.SUCCESS),
        evaluation={
            "task_completion_score": 0.9,
            "execution_success": True,
            "output_completeness": 1.0,
            "visualization_present": False,
            "ml_metrics_valid": False,
            "notes": "",
            "decision_quality_score": 86.0,
            "decision_evaluation": {
                "decision_score": 86.0,
                "scores": {
                    "data_understanding": 9.0,
                    "preprocessing": 8.5,
                    "feature_engineering": 8.0,
                    "model_selection": 8.5,
                    "evaluation_strategy": 8.5,
                    "statistical_validity": 8.0,
                    "business_alignment": 8.5,
                    "explainability": 8.0,
                },
                "strengths": [],
                "weaknesses": [],
                "recommendations": [],
                "confidence": 0.8,
                "summary": "ok",
            },
        },
    )

    baseline_summary = {
        "run_id": "20260607_120000",
        "mode": "baseline",
        "task_count": 1,
        "metrics": {"TCR": 1.0},
        "results": [baseline_result.model_dump(mode="json")],
    }
    feedback_summary = {
        "run_id": "20260607_120000",
        "mode": "feedback_loop",
        "task_count": 1,
        "metrics": {"TCR": 1.0},
        "results": [feedback_result.model_dump(mode="json")],
    }
    b_file = baseline / "benchmark_20260607_111119.json"
    f_file = feedback / "benchmark_20260607_123222.json"
    b_file.write_text(json.dumps(baseline_summary), encoding="utf-8")
    f_file.write_text(json.dumps(feedback_summary), encoding="utf-8")

    monkeypatch.setattr(comparison_metrics, "get_project_root", lambda: tmp_path)

    df, out_path = comparison_metrics.compare_two_files(b_file, f_file)
    assert out_path.name == "comparison_metrics_task_001.csv"
    assert len(df) == 8
    details_path = tables / "comparison_details_task_001.csv"
    assert details_path.exists()
    breakdown_path = tables / "decision_breakdown_task_001.csv"
    assert breakdown_path.exists()
    report_path = tables / "comparison_report_task_001.md"
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Feedback Impact Analysis" in report_text
    assert "Decision Critic Confidence" in report_text
