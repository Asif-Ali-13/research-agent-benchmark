"""Compare baseline vs feedback-loop experiment results."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from research_agent.evaluation.reliability_metrics import ReliabilityMetrics

DECISION_COMPONENTS = [
    ("Data Understanding", "data_understanding"),
    ("Preprocessing", "preprocessing"),
    ("Feature Engineering", "feature_engineering"),
    ("Model Selection", "model_selection"),
    ("Evaluation Strategy", "evaluation_strategy"),
    ("Statistical Validity", "statistical_validity"),
    ("Business Alignment", "business_alignment"),
    ("Explainability", "explainability"),
]
from research_agent.models.result_schema import AgentRunResult
from research_agent.utils.helpers import get_project_root
from research_agent.utils.logger import setup_logger

logger = setup_logger("comparison_metrics")


def parse_run_records(data: Any) -> list[dict[str, Any]]:
    """Extract per-task run dicts from benchmark summary or legacy JSON."""
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            return [r for r in data["results"] if isinstance(r, dict)]
        if "query" in data and "dataset_path" in data:
            return [data]
    return []


def load_benchmark_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_task_id_from_benchmark(data: Any) -> str | None:
    """Return task_id from the first result's metadata, if present."""
    records = parse_run_records(data)
    if not records:
        return None
    metadata = records[0].get("metadata") or {}
    task_id = metadata.get("task_id")
    return str(task_id) if task_id else None


def load_results_from_file(path: Path) -> list[AgentRunResult]:
    """Load AgentRunResult list from a single JSON artifact."""
    data = load_benchmark_json(path)
    return [AgentRunResult.model_validate(r) for r in parse_run_records(data)]


def _result_for_task_id(results: list[AgentRunResult], task_id: str | None) -> AgentRunResult | None:
    if not results:
        return None
    if task_id:
        for r in results:
            if r.metadata.get("task_id") == task_id:
                return r
    return results[0]


def find_benchmark_by_task_id(directory: Path, task_id: str) -> Path | None:
    """Return the latest benchmark file whose results match ``task_id``."""
    matches: list[Path] = []
    if not directory.exists():
        return None
    for path in sorted(directory.glob("benchmark_*.json")):
        data = load_benchmark_json(path)
        if extract_task_id_from_benchmark(data) == task_id:
            matches.append(path)
    return matches[-1] if matches else None


def index_benchmarks_by_task_id(directory: Path) -> dict[str, Path]:
    """Map each task_id to its latest benchmark file (by sorted filename)."""
    index: dict[str, Path] = {}
    if not directory.exists():
        return index
    for path in sorted(directory.glob("benchmark_*.json")):
        task_id = extract_task_id_from_benchmark(load_benchmark_json(path))
        if task_id:
            index[task_id] = path
    return index


def build_output_filename(
    baseline_task_id: str | None,
    feedback_task_id: str | None,
    *,
    path_hash: str | None = None,
) -> str:
    """Build comparison CSV filename from task ids."""
    if baseline_task_id and feedback_task_id:
        if baseline_task_id == feedback_task_id:
            return f"comparison_metrics_{baseline_task_id}.csv"
        return f"comparison_metrics_{baseline_task_id}_{feedback_task_id}.csv"
    suffix = path_hash or "00000000"
    return f"comparison_metrics_unknown_{suffix}.csv"


def _latest_feedback(result: AgentRunResult):
    if result.feedback_history:
        return result.feedback_history[-1]
    if result.evaluation and result.evaluation.structured_feedback:
        return result.evaluation.structured_feedback
    return None


def _qualification_gap_count(result: AgentRunResult) -> int:
    fb = _latest_feedback(result)
    return len(fb.items) if fb else 0


def _build_comparison_details_row(
    baseline_result: AgentRunResult,
    feedback_result: AgentRunResult,
    task_id: str,
) -> dict[str, object]:
    b_fb = _latest_feedback(baseline_result)
    f_fb = _latest_feedback(feedback_result)
    return {
        "task_id": task_id,
        "baseline_score": baseline_result.evaluation.task_completion_score if baseline_result.evaluation else 0.0,
        "feedback_score": feedback_result.evaluation.task_completion_score if feedback_result.evaluation else 0.0,
        "baseline_passed": b_fb.passed if b_fb else False,
        "feedback_passed": f_fb.passed if f_fb else False,
        "baseline_iterations": baseline_result.iterations,
        "feedback_iterations": feedback_result.iterations,
        "baseline_qualified": (
            baseline_result.evaluation.task_qualified
            if baseline_result.evaluation and baseline_result.evaluation.task_qualified is not None
            else (b_fb.qualified if b_fb and b_fb.qualified is not None else False)
        ),
        "feedback_qualified": (
            feedback_result.evaluation.task_qualified
            if feedback_result.evaluation and feedback_result.evaluation.task_qualified is not None
            else (f_fb.qualified if f_fb and f_fb.qualified is not None else False)
        ),
        "feedback_recovered": feedback_result.recovered_from_failure,
        "qualification_gap_count_baseline": _qualification_gap_count(baseline_result),
        "qualification_gap_count_feedback": _qualification_gap_count(feedback_result),
    }


def build_comparison_details_filename(task_id: str | None) -> str:
    suffix = task_id or "unknown"
    return f"comparison_details_{suffix}.csv"


def _build_comparison_dataframe(
    baseline_results: list[AgentRunResult],
    feedback_results: list[AgentRunResult],
) -> pd.DataFrame:
    baseline_metrics = ReliabilityMetrics.from_results(baseline_results).summary()
    feedback_metrics = ReliabilityMetrics.from_results(feedback_results).summary()

    rows = []
    for key in ["TCR", "ESR", "RR", "FIS", "DA", "DQS", "BAS", "ORS"]:
        b = baseline_metrics.get(key, 0.0)
        f = feedback_metrics.get(key, 0.0)
        rows.append({
            "metric": key,
            "baseline": b,
            "feedback_loop": f,
            "delta": round(f - b, 4),
            "improved": f > b,
        })
    return pd.DataFrame(rows)


def _decision_component_scores(result: AgentRunResult) -> dict[str, float]:
    if result.evaluation and result.evaluation.decision_evaluation is not None:
        scores = result.evaluation.decision_evaluation.scores.model_dump()
        return {
            display_name: round(float(scores.get(attr, 0.0)), 4)
            for display_name, attr in DECISION_COMPONENTS
        }
    return {display_name: 0.0 for display_name, _ in DECISION_COMPONENTS}


def build_decision_breakdown_filename(task_id: str | None) -> str:
    suffix = task_id or "unknown"
    return f"decision_breakdown_{suffix}.csv"


def _build_decision_breakdown_dataframe(
    baseline_result: AgentRunResult,
    feedback_result: AgentRunResult,
) -> pd.DataFrame:
    baseline_scores = _decision_component_scores(baseline_result)
    feedback_scores = _decision_component_scores(feedback_result)

    rows = []
    for component, _ in DECISION_COMPONENTS:
        b = baseline_scores.get(component, 0.0)
        f = feedback_scores.get(component, 0.0)
        rows.append({
            "component": component,
            "baseline": b,
            "feedback_loop": f,
            "delta": round(f - b, 4),
            "improved": f > b,
        })
    return pd.DataFrame(rows)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if not df.empty:
        headers = [str(col) for col in df.columns]
        rows = [headers]
        for _, row in df.iterrows():
            rows.append([str(row[col]) for col in df.columns])
        widths = [max(len(str(r[i])) for r in rows) for i in range(len(headers))]
        lines = []
        for row in rows:
            lines.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + " |")
            if row is rows[0]:
                lines.append("| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |")
        return "\n".join(lines)
    return ""


def _build_markdown_report(
    comparison_df: pd.DataFrame,
    breakdown_df: pd.DataFrame,
    task_id: str | None,
    chart_path: Path,
    report_path: Path,
    confidence: float | None = None,
) -> str:
    comparison_table = _dataframe_to_markdown(comparison_df)
    breakdown_table = _dataframe_to_markdown(breakdown_df)
    improved_rows = breakdown_df[breakdown_df["delta"] > 0]
    regressed_rows = breakdown_df[breakdown_df["delta"] < 0]
    most_improved = improved_rows.sort_values("delta", ascending=False).iloc[0] if not improved_rows.empty else None
    largest_regression = regressed_rows.sort_values("delta", ascending=True).iloc[0] if not regressed_rows.empty else None
    highest_scoring = breakdown_df.sort_values("feedback_loop", ascending=False).iloc[0] if not breakdown_df.empty else None
    lowest_scoring = breakdown_df.sort_values("feedback_loop", ascending=True).iloc[0] if not breakdown_df.empty else None
    avg_improvement = round(float(breakdown_df["delta"].mean()), 4) if not breakdown_df.empty else 0.0
    needing_attention = [
        row["component"]
        for _, row in breakdown_df.iterrows()
        if row["feedback_loop"] < 7.0 or row["delta"] < 1.0
    ]

    positive = [
        f"✓ {row['component']} (+{row['delta']:.1f})"
        for _, row in breakdown_df.iterrows()
        if row["delta"] > 0
    ]
    neutral = [
        row["component"]
        for _, row in breakdown_df.iterrows()
        if abs(row["delta"]) < 1e-9
    ]
    negative = [
        f"✗ {row['component']} ({row['delta']:.1f})"
        for _, row in breakdown_df.iterrows()
        if row["delta"] < 0
    ]

    if breakdown_df.empty:
        interpretation = "No decision breakdown available."
    elif comparison_df.loc[comparison_df["metric"] == "FIS", "delta"].iloc[0] > 0:
        interpretation = (
            "The feedback improved analytical decision quality, especially where component scores increased."
        )
    elif comparison_df.loc[comparison_df["metric"] == "FIS", "delta"].iloc[0] < 0:
        interpretation = (
            "The feedback did not improve the highest-impact analytical decisions, so the overall feedback improvement score remained negative."
        )
    else:
        interpretation = "Feedback had no net effect on the analytical decision quality profile."

    chart_link = os.path.relpath(chart_path, report_path.parent)
    chart_md = f"![Decision Quality Radar]({chart_link})"
    return f"""# Comparison Report for {task_id or 'unknown'}

## Summary Metrics

{comparison_table}

## Decision Quality Breakdown

{breakdown_table}

{chart_md}

## Feedback Impact Analysis

### Positive Contributions
- {"\n- ".join(positive) if positive else 'None'}

### Neutral Components
- {", ".join(neutral) if neutral else 'None'}

### Negative Contributions
- {"\n- ".join(negative) if negative else 'None'}

### Overall Interpretation
- {interpretation}

### Additional Summary
- Highest scoring component: {highest_scoring['component'] if highest_scoring is not None else 'n/a'} ({highest_scoring['feedback_loop'] if highest_scoring is not None else 0.0})
- Lowest scoring component: {lowest_scoring['component'] if lowest_scoring is not None else 'n/a'} ({lowest_scoring['feedback_loop'] if lowest_scoring is not None else 0.0})
- Largest improvement: {most_improved['component'] if most_improved is not None else 'n/a'} ({most_improved['delta'] if most_improved is not None else 0.0})
- Largest regression: {largest_regression['component'] if largest_regression is not None else 'n/a'} ({largest_regression['delta'] if largest_regression is not None else 0.0})
- Average improvement: {avg_improvement}
- Components requiring further refinement: {', '.join(needing_attention) if needing_attention else 'none'}
- Decision Critic Confidence: {confidence if confidence is not None else 'n/a'}
"""


def _write_decision_quality_radar(
    baseline_result: AgentRunResult,
    feedback_result: AgentRunResult,
    task_id: str | None,
    output_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    categories = [name for name, _ in DECISION_COMPONENTS]
    baseline_scores = [
        _decision_component_scores(baseline_result).get(component, 0.0)
        for component, _ in DECISION_COMPONENTS
    ]
    feedback_scores = [
        _decision_component_scores(feedback_result).get(component, 0.0)
        for component, _ in DECISION_COMPONENTS
    ]
    values = np.array([baseline_scores, feedback_scores])
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "decision_quality_radar.png"
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_rlabel_position(90)

    for idx, label in enumerate(["Baseline", "Feedback"]):
        vals = values[idx].tolist() + [values[idx, 0].tolist()]
        ax.plot(angles, vals, linewidth=2, label=label)
        ax.fill(angles, vals, alpha=0.1)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_comparison_df(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def _path_hash(baseline_path: Path, feedback_path: Path) -> str:
    digest = hashlib.sha256(f"{baseline_path}|{feedback_path}".encode()).hexdigest()
    return digest[:8]


def append_comparison_log(
    *,
    baseline_path: Path,
    feedback_path: Path,
    output_path: Path,
    baseline_task_id: str | None,
    feedback_task_id: str | None,
    df: pd.DataFrame,
    baseline_result: AgentRunResult | None = None,
    feedback_result: AgentRunResult | None = None,
) -> None:
    """Append one row to reports/tables/comparison_log.csv for per-task runs."""
    log_path = get_project_root() / "reports" / "tables" / "comparison_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {row["metric"]: row for _, row in df.iterrows()}
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_file": baseline_path.name,
        "feedback_file": feedback_path.name,
        "baseline_task_id": baseline_task_id or "",
        "feedback_task_id": feedback_task_id or "",
        "output_file": output_path.name,
        "TCR_baseline": metrics.get("TCR", {}).get("baseline", ""),
        "TCR_feedback": metrics.get("TCR", {}).get("feedback_loop", ""),
        "ESR_baseline": metrics.get("ESR", {}).get("baseline", ""),
        "ESR_feedback": metrics.get("ESR", {}).get("feedback_loop", ""),
        "FIS_baseline": metrics.get("FIS", {}).get("baseline", ""),
        "FIS_feedback": metrics.get("FIS", {}).get("feedback_loop", ""),
        "DA_baseline": metrics.get("DA", {}).get("baseline", ""),
        "DA_feedback": metrics.get("DA", {}).get("feedback_loop", ""),
        "DQS_baseline": metrics.get("DQS", {}).get("baseline", ""),
        "DQS_feedback": metrics.get("DQS", {}).get("feedback_loop", ""),
        "BAS_baseline": metrics.get("BAS", {}).get("baseline", ""),
        "BAS_feedback": metrics.get("BAS", {}).get("feedback_loop", ""),
        "ORS_baseline": metrics.get("ORS", {}).get("baseline", ""),
        "ORS_feedback": metrics.get("ORS", {}).get("feedback_loop", ""),
        "decision_critic_confidence": (
            feedback_result.evaluation.decision_evaluation.confidence
            if feedback_result and feedback_result.evaluation and feedback_result.evaluation.decision_evaluation
            else ""
        ),
    }
    header = list(row.keys())
    write_header = not log_path.exists()
    with log_path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(row[k]) for k in header) + "\n")


def compare_two_files(
    baseline_path: Path,
    feedback_path: Path,
    output_path: str | Path | None = None,
    *,
    append_log: bool = True,
) -> tuple[pd.DataFrame, Path]:
    """Compare a single baseline benchmark file against a single feedback file."""
    baseline_data = load_benchmark_json(baseline_path)
    feedback_data = load_benchmark_json(feedback_path)
    baseline_task_id = extract_task_id_from_benchmark(baseline_data)
    feedback_task_id = extract_task_id_from_benchmark(feedback_data)

    baseline_all = load_results_from_file(baseline_path)
    feedback_all = load_results_from_file(feedback_path)
    baseline_result = _result_for_task_id(baseline_all, baseline_task_id)
    feedback_result = _result_for_task_id(feedback_all, feedback_task_id)

    if baseline_result is None or feedback_result is None:
        raise ValueError("Could not load results from one or both benchmark files")

    df = _build_comparison_dataframe([baseline_result], [feedback_result])

    if output_path is None:
        filename = build_output_filename(
            baseline_task_id,
            feedback_task_id,
            path_hash=_path_hash(baseline_path, feedback_path),
        )
        output_path = get_project_root() / "reports" / "tables" / filename

    out = _save_comparison_df(df, Path(output_path))
    details_task_id = baseline_task_id or feedback_task_id
    if details_task_id:
        details_df = df.copy()
        details_path = get_project_root() / "reports" / "tables" / build_comparison_details_filename(
            details_task_id
        )
        _save_comparison_df(details_df, details_path)

        breakdown_df = _build_decision_breakdown_dataframe(baseline_result, feedback_result)
        breakdown_path = get_project_root() / "reports" / "tables" / build_decision_breakdown_filename(
            details_task_id
        )
        _save_comparison_df(breakdown_df, breakdown_path)

        chart_dir = get_project_root() / "reports" / "figures" / (details_task_id or "comparison")
        chart_path = _write_decision_quality_radar(
            baseline_result,
            feedback_result,
            details_task_id,
            chart_dir,
        )
        report_path = get_project_root() / "reports" / "tables" / f"comparison_report_{details_task_id}.md"
        confidence = (
            feedback_result.evaluation.decision_evaluation.confidence
            if feedback_result.evaluation and feedback_result.evaluation.decision_evaluation
            else None
        )
        report_path.write_text(
            _build_markdown_report(
                df,
                breakdown_df,
                details_task_id,
                chart_path,
                report_path,
                confidence=confidence,
            ),
            encoding="utf-8",
        )
    if append_log:
        append_comparison_log(
            baseline_path=baseline_path,
            feedback_path=feedback_path,
            output_path=out,
            baseline_task_id=baseline_task_id,
            feedback_task_id=feedback_task_id,
            df=df,
            baseline_result=baseline_result,
            feedback_result=feedback_result,
        )
    return df, out


def compare_final(
    baseline_dir: str | Path,
    feedback_dir: str | Path,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Aggregate paired per-task benchmark runs and write comparison_metrics_final.csv."""
    root = get_project_root()
    baseline_index = index_benchmarks_by_task_id(root / baseline_dir)
    feedback_index = index_benchmarks_by_task_id(root / feedback_dir)

    paired_ids = sorted(set(baseline_index) & set(feedback_index))
    baseline_only = sorted(set(baseline_index) - set(feedback_index))
    feedback_only = sorted(set(feedback_index) - set(baseline_index))

    if baseline_only:
        logger.warning("Tasks with baseline only (skipped in final): %s", baseline_only)
    if feedback_only:
        logger.warning("Tasks with feedback_loop only (skipped in final): %s", feedback_only)
    if not paired_ids:
        raise ValueError("No task_ids found in both baseline and feedback_loop directories")

    baseline_results: list[AgentRunResult] = []
    feedback_results: list[AgentRunResult] = []
    for task_id in paired_ids:
        b_result = _result_for_task_id(
            load_results_from_file(baseline_index[task_id]),
            task_id,
        )
        f_result = _result_for_task_id(
            load_results_from_file(feedback_index[task_id]),
            task_id,
        )
        if b_result and f_result:
            baseline_results.append(b_result)
            feedback_results.append(f_result)

    df = _build_comparison_dataframe(baseline_results, feedback_results)
    out = _save_comparison_df(
        df,
        Path(output_path) if output_path else root / "reports" / "tables" / "comparison_metrics_final.csv",
    )
    logger.info("Final comparison over %d paired task(s)", len(paired_ids))
    return df, out


def list_benchmark_files(
    baseline_dir: str | Path,
    feedback_dir: str | Path,
) -> pd.DataFrame:
    """List benchmark JSON files with mode, timestamp, and task_id."""
    root = get_project_root()
    rows: list[dict[str, str]] = []
    for mode, subdir in [("baseline", baseline_dir), ("feedback_loop", feedback_dir)]:
        directory = root / subdir
        if not directory.exists():
            continue
        for path in sorted(directory.glob("benchmark_*.json")):
            data = load_benchmark_json(path)
            rows.append({
                "mode": mode,
                "benchmark_id": path.stem.removeprefix("benchmark_"),
                "filename": path.name,
                "task_id": extract_task_id_from_benchmark(data) or "",
            })
    return pd.DataFrame(rows)


def load_results_from_dir(directory: Path) -> list[AgentRunResult]:
    """Load task results from experiment directory.

    Prefers the latest ``benchmark_*.json`` summary written by BenchmarkRunner.
    Falls back to all ``*.json`` files for legacy single-run artifacts.
    """
    if not directory.exists():
        return []

    benchmark_files = sorted(directory.glob("benchmark_*.json"))
    if benchmark_files:
        return load_results_from_file(benchmark_files[-1])

    results: list[AgentRunResult] = []
    for path in sorted(directory.glob("*.json")):
        results.extend(load_results_from_file(path))
    return results


def compare_experiments(
    baseline_dir: str | Path,
    feedback_dir: str | Path,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Compare latest benchmark file in each directory (legacy full-suite mode)."""
    root = get_project_root()
    baseline_results = load_results_from_dir(root / baseline_dir)
    feedback_results = load_results_from_dir(root / feedback_dir)

    df = _build_comparison_dataframe(baseline_results, feedback_results)
    out = _save_comparison_df(
        df,
        Path(output_path) if output_path else root / "reports" / "tables" / "comparison_metrics.csv",
    )
    return df, out
