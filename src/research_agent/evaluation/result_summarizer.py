"""Benchmark result summarization utilities.

This module scans existing benchmark outputs under `reports/tables/` and
`reports/figures/`, aggregates metrics, generates CSV summaries,
publication-quality Markdown, optional statistical tests (if scipy is
available), and matplotlib figures. It is additive only and never
overwrites existing benchmark outputs.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("result_summarizer")

METRICS = ["TCR", "ESR", "RR", "FIS", "DA", "DQS", "BAS", "ORS"]
DECISION_COMPONENTS = [
    "Data Understanding",
    "Preprocessing",
    "Feature Engineering",
    "Model Selection",
    "Evaluation Strategy",
    "Statistical Validity",
    "Business Alignment",
    "Explainability",
]


def safe_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def discover_tasks(tables_dir: Path) -> List[str]:
    tasks = set()
    for p in tables_dir.glob("comparison_metrics_task_*.csv"):
        name = p.stem  # comparison_metrics_task_026
        parts = name.split("comparison_metrics_")
        if len(parts) == 2:
            tasks.add(parts[1])
    for p in tables_dir.glob("comparison_report_task_*.md"):
        name = p.stem
        parts = name.split("comparison_report_")
        if len(parts) == 2:
            tasks.add(parts[1])
    task_list = sorted(tasks)
    logger.info("Discovered %d tasks", len(task_list))
    return task_list


def load_tasks_metadata(repo_root: Path) -> Dict[str, Dict[str, str]]:
    tasks_file = repo_root / "datasets" / "benchmark_tasks" / "tasks.json"
    mapping: Dict[str, Dict[str, str]] = {}
    if not tasks_file.exists():
        return mapping
    try:
        data = json.loads(tasks_file.read_text())
        for entry in data:
            tid = entry.get("id")
            if tid:
                mapping[tid] = {
                    "dataset": entry.get("dataset", ""),
                    "difficulty": entry.get("difficulty", ""),
                }
    except Exception:
        logger.exception("Failed to load tasks.json")
    return mapping


def read_metrics_csv(path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    # returns metric -> {baseline, feedback_loop, delta, improved}
    out: Dict[str, Dict[str, Optional[float]]] = {}
    try:
        with path.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                metric = row.get("metric") or row.get("component")
                if not metric:
                    continue
                baseline = safe_float(row.get("baseline"))
                feedback = safe_float(row.get("feedback_loop"))
                delta = safe_float(row.get("delta"))
                improved = row.get("improved") in ("True", "true", "1")
                out[metric] = {
                    "baseline": baseline,
                    "feedback": feedback,
                    "delta": delta,
                    "improved": improved,
                }
    except Exception:
        logger.exception("Failed to read metrics CSV: %s", path)
    return out


@dataclass
class TaskMetrics:
    task_id: str
    metrics: Dict[str, Dict[str, Optional[float]]]
    decision: Dict[str, Dict[str, Optional[float]]]
    dataset: Optional[str] = None
    difficulty: Optional[str] = None


class ResultSummarizer:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.tables_dir = repo_root / "reports" / "tables"
        self.figures_dir = repo_root / "reports" / "figures"
        self.out_root = repo_root / "research_artifacts"
        self.out_summaries = self.out_root / "summaries"
        self.out_stats = self.out_root / "statistics"
        self.out_figs = self.out_root / "figures"
        self.out_cases = self.out_root / "case_studies"
        for d in (self.out_root, self.out_summaries, self.out_stats, self.out_figs, self.out_cases):
            d.mkdir(parents=True, exist_ok=True)
        self.task_meta = load_tasks_metadata(repo_root)

    def summarize(self) -> None:
        tasks = discover_tasks(self.tables_dir)
        logger.info("Found %d tasks to summarize", len(tasks))

        task_metrics: List[TaskMetrics] = []
        for tid in tasks:
            cm = self.tables_dir / f"comparison_metrics_{tid}.csv"
            db = self.tables_dir / f"decision_breakdown_{tid}.csv"
            metrics = read_metrics_csv(cm) if cm.exists() else {}
            decision = read_metrics_csv(db) if db.exists() else {}
            meta = self.task_meta.get(tid, {})
            tm = TaskMetrics(task_id=tid, metrics=metrics, decision=decision, dataset=meta.get("dataset"), difficulty=meta.get("difficulty"))
            task_metrics.append(tm)

        # Generate overall_metrics.csv
        overall_rows = []
        for metric in METRICS:
            deltas = [tm.metrics.get(metric, {}).get("delta") for tm in task_metrics if tm.metrics.get(metric)]
            deltas = [d for d in deltas if d is not None]
            baseline_vals = [tm.metrics.get(metric, {}).get("baseline") for tm in task_metrics if tm.metrics.get(metric)]
            baseline_vals = [b for b in baseline_vals if b is not None]
            feedback_vals = [tm.metrics.get(metric, {}).get("feedback") for tm in task_metrics if tm.metrics.get(metric)]
            feedback_vals = [f for f in feedback_vals if f is not None]
            if deltas:
                mean_delta = statistics.mean(deltas)
                median_delta = statistics.median(deltas)
                std_delta = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
            else:
                mean_delta = median_delta = std_delta = 0.0
            baseline_mean = statistics.mean(baseline_vals) if baseline_vals else 0.0
            feedback_mean = statistics.mean(feedback_vals) if feedback_vals else 0.0
            improvement_pct = (mean_delta / baseline_mean * 100.0) if baseline_mean not in (0.0, None) else 0.0
            overall_rows.append({
                "Metric": metric,
                "Baseline Mean": f"{baseline_mean:.4f}",
                "Feedback Mean": f"{feedback_mean:.4f}",
                "Mean Delta": f"{mean_delta:.4f}",
                "Median Delta": f"{median_delta:.4f}",
                "Std Dev": f"{std_delta:.4f}",
                "Improvement %": f"{improvement_pct:.2f}",
            })

        overall_csv = self.out_summaries / "overall_metrics.csv"
        with overall_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(overall_rows[0].keys()) if overall_rows else ["Metric"])
            writer.writeheader()
            for r in overall_rows:
                writer.writerow(r)

        # Generate benchmark_statistics.csv
        stats_rows = []
        for tm in sorted(task_metrics, key=lambda t: t.task_id):
            row: Dict[str, object] = {"Task ID": tm.task_id, "Dataset": tm.dataset or "", "Difficulty": tm.difficulty or ""}
            exec_status = []
            for metric in METRICS:
                m = tm.metrics.get(metric, {})
                row[f"{metric}_Baseline"] = m.get("baseline")
                row[f"{metric}_Feedback"] = m.get("feedback")
                row[f"{metric}_Delta"] = m.get("delta")
                row[f"{metric}_Improved"] = m.get("improved")
                if m.get("baseline") is not None:
                    exec_status.append("baseline:ok")
                else:
                    exec_status.append("baseline:missing")
                if m.get("feedback") is not None:
                    exec_status.append("feedback:ok")
                else:
                    exec_status.append("feedback:missing")
            row["Execution Status"] = ";".join(sorted(set(exec_status)))
            stats_rows.append(row)

        stats_csv = self.out_stats / "benchmark_statistics.csv"
        if stats_rows:
            with stats_csv.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(stats_rows[0].keys()))
                writer.writeheader()
                for r in stats_rows:
                    writer.writerow(r)

        # overall_decision_breakdown.csv
        comp_avgs: Dict[str, Dict[str, float]] = {}
        for comp in DECISION_COMPONENTS:
            baseline_vals = []
            feedback_vals = []
            improved_count = 0
            for tm in task_metrics:
                d = tm.decision.get(comp)
                if d:
                    if d.get("baseline") is not None:
                        baseline_vals.append(d.get("baseline"))
                    if d.get("feedback") is not None:
                        feedback_vals.append(d.get("feedback"))
                    if d.get("improved"):
                        improved_count += 1
            baseline_avg = statistics.mean(baseline_vals) if baseline_vals else 0.0
            feedback_avg = statistics.mean(feedback_vals) if feedback_vals else 0.0
            comp_avgs[comp] = {
                "Baseline": baseline_avg,
                "Feedback": feedback_avg,
                "Delta": feedback_avg - baseline_avg,
                "Improved": improved_count,
            }

        decision_csv = self.out_stats / "overall_decision_breakdown.csv"
        with decision_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Component", "Baseline", "Feedback", "Delta", "ImprovedCount"])
            for comp, vals in comp_avgs.items():
                writer.writerow([comp, f"{vals['Baseline']:.4f}", f"{vals['Feedback']:.4f}", f"{vals['Delta']:.4f}", vals["Improved"]])

        # task_leaderboard.csv
        leaderboard_rows = []
        for tm in task_metrics:
            ors = tm.metrics.get("ORS", {})
            dqs = tm.metrics.get("DQS", {})
            fis = tm.metrics.get("FIS", {})
            bas = tm.metrics.get("BAS", {})
            improved_count = sum(1 for m in METRICS if tm.metrics.get(m, {}).get("improved"))
            leaderboard_rows.append({
                "Task ID": tm.task_id,
                "ORS_Delta": ors.get("delta") if ors else 0.0,
                "DQS_Delta": dqs.get("delta") if dqs else 0.0,
                "FIS_Delta": fis.get("delta") if fis else 0.0,
                "BAS_Delta": bas.get("delta") if bas else 0.0,
                "Execution_Improved_Count": improved_count,
            })

        leaderboard_csv = self.out_summaries / "task_leaderboard.csv"
        if leaderboard_rows:
            leaderboard_rows_sorted = sorted(leaderboard_rows, key=lambda r: (r["ORS_Delta"] or 0.0), reverse=True)
            with leaderboard_csv.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(leaderboard_rows_sorted[0].keys()))
                writer.writeheader()
                for r in leaderboard_rows_sorted:
                    writer.writerow(r)

        # benchmark_summary.md
        summary_md = self.out_summaries / "benchmark_summary.md"
        total_tasks = len(task_metrics)
        successful_baseline = sum(1 for tm in task_metrics if any(tm.metrics.get(m, {}).get("baseline") is not None for m in METRICS))
        successful_feedback = sum(1 for tm in task_metrics if any(tm.metrics.get(m, {}).get("feedback") is not None for m in METRICS))
        avg_values = {}
        for metric in METRICS:
            vals = [tm.metrics.get(metric, {}).get("feedback") for tm in task_metrics if tm.metrics.get(metric) and tm.metrics.get(metric).get("feedback") is not None]
            if vals:
                avg_values[metric] = statistics.mean(vals)
            else:
                avg_values[metric] = 0.0

        # top improvements
        ors_sorted = sorted(((tm.task_id, tm.metrics.get("ORS", {}).get("delta") or 0.0) for tm in task_metrics), key=lambda x: x[1], reverse=True)
        dqs_sorted = sorted(((tm.task_id, tm.metrics.get("DQS", {}).get("delta") or 0.0) for tm in task_metrics), key=lambda x: x[1], reverse=True)
        fis_sorted = sorted(((tm.task_id, tm.metrics.get("FIS", {}).get("delta") or 0.0) for tm in task_metrics), key=lambda x: x[1], reverse=True)

        most_difficult = None
        difficulty_rank = {"easy": 1, "medium": 2, "hard": 3, "very_hard": 4, "expert": 5}
        if task_metrics:
            most_difficult = max(task_metrics, key=lambda t: difficulty_rank.get((t.difficulty or ""), 0))

        # best / worst dataset by ORS
        dataset_scores: Dict[str, List[float]] = defaultdict(list)
        for tm in task_metrics:
            ds = tm.dataset or ""
            ors = tm.metrics.get("ORS", {}).get("feedback")
            if ors is not None:
                dataset_scores[ds].append(ors)
        dataset_avg = {ds: (statistics.mean(v) if v else 0.0) for ds, v in dataset_scores.items()}
        best_dataset = max(dataset_avg.items(), key=lambda x: x[1])[0] if dataset_avg else ""
        worst_dataset = min(dataset_avg.items(), key=lambda x: x[1])[0] if dataset_avg else ""

        with summary_md.open("w") as fh:
            fh.write(f"# Benchmark Summary\n\n")
            fh.write(f"Total benchmark tasks: {total_tasks}\n\n")
            fh.write(f"Successful baseline executions: {successful_baseline}\n\n")
            fh.write(f"Successful feedback executions: {successful_feedback}\n\n")
            fh.write("## Average Metrics (feedback)\n\n")
            for metric in METRICS:
                fh.write(f"- Average {metric}: {avg_values.get(metric,0.0):.4f}\n")
            fh.write("\n")
            fh.write(f"Largest ORS improvement: {ors_sorted[0] if ors_sorted else ('N/A', 0)}\n")
            fh.write(f"Largest ORS degradation: {ors_sorted[-1] if ors_sorted else ('N/A', 0)}\n")
            fh.write("\n")
            fh.write("Top 5 DQS improvements:\n")
            for tid, val in dqs_sorted[:5]:
                fh.write(f"- {tid}: {val:.4f}\n")
            fh.write("\n")
            fh.write("Top 5 ORS improvements:\n")
            for tid, val in ors_sorted[:5]:
                fh.write(f"- {tid}: {val:.4f}\n")
            fh.write("\n")
            fh.write("Top 5 FIS improvements:\n")
            for tid, val in fis_sorted[:5]:
                fh.write(f"- {tid}: {val:.4f}\n")
            fh.write("\n")
            fh.write(f"Most difficult benchmark: {most_difficult.task_id if most_difficult else 'N/A'} ({most_difficult.difficulty if most_difficult else 'N/A'})\n")
            fh.write(f"Best performing dataset: {best_dataset}\n")
            fh.write(f"Worst performing dataset: {worst_dataset}\n")
            fh.write("\n## Observations\n\n")
            fh.write("Automated summary generated programmatically. Inspect `research_artifacts/` for full CSVs, figures, and case studies.\n")

        # Optional statistical tests using scipy
        try:
            import scipy.stats as stats  # type: ignore

            stat_md = self.out_summaries / "statistical_analysis.md"
            with stat_md.open("w") as fh:
                fh.write("# Statistical Analysis\n\n")
                for metric in METRICS:
                    base = [tm.metrics.get(metric, {}).get("baseline") for tm in task_metrics if tm.metrics.get(metric) and tm.metrics.get(metric).get("baseline") is not None and tm.metrics.get(metric).get("feedback") is not None]
                    feed = [tm.metrics.get(metric, {}).get("feedback") for tm in task_metrics if tm.metrics.get(metric) and tm.metrics.get(metric).get("baseline") is not None and tm.metrics.get(metric).get("feedback") is not None]
                    if not base or not feed:
                        continue
                    try:
                        t_res = stats.ttest_rel(feed, base)
                        w_res = stats.wilcoxon(feed, base)
                        mean_diff = statistics.mean([f - b for f, b in zip(feed, base)])
                        ci_low, ci_high = stats.t.interval(0.95, len(feed)-1, loc=mean_diff, scale=statistics.pstdev([f - b for f, b in zip(feed, base)])/math.sqrt(len(feed)))
                        # Cohen's d
                        pooled_sd = statistics.pstdev([f - b for f, b in zip(feed, base)])
                        cohen_d = (statistics.mean(feed) - statistics.mean(base)) / (pooled_sd if pooled_sd else 1.0)
                        fh.write(f"## {metric}\n")
                        fh.write(f"- Paired t-test t={t_res.statistic:.4f}, p={t_res.pvalue:.4f}\n")
                        fh.write(f"- Wilcoxon W={w_res.statistic:.4f}, p={w_res.pvalue:.4f}\n")
                        fh.write(f"- 95% CI for mean diff: ({ci_low:.4f}, {ci_high:.4f})\n")
                        fh.write(f"- Mean diff: {mean_diff:.4f}\n")
                        fh.write(f"- Std dev diff: {statistics.pstdev([f - b for f, b in zip(feed, base)]):.4f}\n")
                        fh.write(f"- Cohen's d: {cohen_d:.4f}\n\n")
                    except Exception:
                        logger.exception("Failed statistical test for %s", metric)
        except Exception:
            logger.info("scipy not available, skipping statistical analysis")

        # Visualizations (matplotlib only)
        try:
            # overall metrics bar chart (means)
            labels = [r["Metric"] for r in overall_rows]
            baseline_means = [float(r["Baseline Mean"]) for r in overall_rows]
            feedback_means = [float(r["Feedback Mean"]) for r in overall_rows]
            x = range(len(labels))
            plt.figure(figsize=(10, 6))
            plt.bar([i - 0.2 for i in x], baseline_means, width=0.4, label="Baseline")
            plt.bar([i + 0.2 for i in x], feedback_means, width=0.4, label="Feedback")
            plt.xticks(x, labels, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.out_figs / "overall_metrics_bar_chart.png")
            plt.close()

            # line chart
            plt.figure(figsize=(10, 6))
            plt.plot(x, baseline_means, marker="o", label="Baseline")
            plt.plot(x, feedback_means, marker="o", label="Feedback")
            plt.xticks(x, labels, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.out_figs / "overall_metrics_line_chart.png")
            plt.close()

            # boxplot of deltas per metric
            deltas_per_metric = []
            for metric in METRICS:
                deltas = [tm.metrics.get(metric, {}).get("delta") for tm in task_metrics if tm.metrics.get(metric) and tm.metrics.get(metric).get("delta") is not None]
                deltas_per_metric.append(deltas if deltas else [0.0])
            plt.figure(figsize=(10, 6))
            plt.boxplot(deltas_per_metric, labels=METRICS)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.out_figs / "overall_metrics_boxplot.png")
            plt.close()

            # heatmap: correlation of mean metrics (use simple imshow)
            import numpy as np

            arr = np.array([[float(r["Baseline Mean"]) for r in overall_rows], [float(r["Feedback Mean"]) for r in overall_rows]])
            corr = np.corrcoef(arr)
            plt.figure(figsize=(6, 6))
            plt.imshow(corr, cmap="viridis", vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks([0, 1], ["Baseline", "Feedback"])
            plt.yticks([0, 1], ["Baseline", "Feedback"])
            plt.title("Metric means correlation")
            plt.tight_layout()
            plt.savefig(self.out_figs / "overall_metrics_heatmap.png")
            plt.close()

            # distributions & histograms
            ors_deltas = [tm.metrics.get("ORS", {}).get("delta") for tm in task_metrics if tm.metrics.get("ORS") and tm.metrics.get("ORS").get("delta") is not None]
            dqs_deltas = [tm.metrics.get("DQS", {}).get("delta") for tm in task_metrics if tm.metrics.get("DQS") and tm.metrics.get("DQS").get("delta") is not None]
            fis_deltas = [tm.metrics.get("FIS", {}).get("delta") for tm in task_metrics if tm.metrics.get("FIS") and tm.metrics.get("FIS").get("delta") is not None]
            plt.figure()
            plt.hist([d for d in ors_deltas], bins=20)
            plt.title("ORS_distribution")
            plt.savefig(self.out_figs / "ORS_distribution.png")
            plt.close()

            plt.figure()
            plt.hist([d for d in dqs_deltas], bins=20)
            plt.title("DQS_distribution")
            plt.savefig(self.out_figs / "DQS_distribution.png")
            plt.close()

            plt.figure()
            plt.hist([d for d in fis_deltas], bins=20)
            plt.title("FIS_distribution")
            plt.savefig(self.out_figs / "FIS_distribution.png")
            plt.close()

            # improvement histogram
            all_deltas = [d for tm in task_metrics for m in METRICS for d in [tm.metrics.get(m, {}).get("delta")] if d is not None]
            plt.figure()
            plt.hist(all_deltas, bins=50)
            plt.title("Improvement_histogram")
            plt.savefig(self.out_figs / "Improvement_histogram.png")
            plt.close()

        except Exception:
            logger.exception("Failed to generate some figures")

        # Case studies
        try:
            # pick representatives
            def pick_task_with_metric(metric: str, best: bool = True) -> Optional[TaskMetrics]:
                lst = [(tm, tm.metrics.get(metric, {}).get("delta") or 0.0) for tm in task_metrics if tm.metrics.get(metric)]
                if not lst:
                    return None
                lst_sorted = sorted(lst, key=lambda x: x[1], reverse=best)
                return lst_sorted[0][0]

            categories = {
                "highest_ors_improvement": pick_task_with_metric("ORS", True),
                "highest_dqs_improvement": pick_task_with_metric("DQS", True),
                "highest_fis_improvement": pick_task_with_metric("FIS", True),
                "largest_degradation": pick_task_with_metric("ORS", False),
            }
            for cat, tm in categories.items():
                if not tm:
                    continue
                dest = self.out_cases / cat
                dest.mkdir(parents=True, exist_ok=True)
                # copy files
                src_base = self.tables_dir
                for fname in [f"comparison_report_{tm.task_id}.md", f"comparison_metrics_{tm.task_id}.csv", f"decision_breakdown_{tm.task_id}.csv"]:
                    src = src_base / fname
                    if src.exists():
                        try:
                            import shutil

                            shutil.copy2(src, dest / src.name)
                        except Exception:
                            logger.exception("Failed copying %s", src)
                # copy radar image if exists
                radar = self.figures_dir / f"task_{tm.task_id.split('_')[-1]}" / "decision_quality_radar.png"
                # also accept reports/figures/task_026/decision_quality_radar.png pattern
                # fallback: search for any matching radar png under reports/figures/*/decision_quality_radar.png
                if not radar.exists():
                    for p in self.figures_dir.glob("**/decision_quality_radar.png"):
                        if tm.task_id.split("_")[-1] in str(p):
                            radar = p
                            break
                if radar.exists():
                    try:
                        import shutil

                        shutil.copy2(radar, dest / radar.name)
                    except Exception:
                        logger.exception("Failed copying radar image %s", radar)
        except Exception:
            logger.exception("Failed to create case studies")

        logger.info("Summarization completed. Artifacts saved to %s", str(self.out_root))
