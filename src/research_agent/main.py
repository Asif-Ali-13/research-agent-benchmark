"""CLI entry point for research agent system."""

from __future__ import annotations

import argparse
import json
import sys

from research_agent.evaluation.benchmark_runner import BenchmarkRunner
from research_agent.evaluation.comparison_metrics import (
    compare_experiments,
    compare_final,
    compare_two_files,
    find_benchmark_by_task_id,
    list_benchmark_files,
)
from research_agent.flows.feedback_flow import FeedbackFlow
from research_agent.flows.reliability_flow import ReliabilityFlow
from research_agent.models.request_schema import BenchmarkMode, RunMode
from research_agent.utils.file_handler import FileHandler
from research_agent.utils.helpers import get_project_root, load_env
from research_agent.utils.logger import setup_logger
from research_agent.evaluation.result_summarizer import ResultSummarizer

logger = setup_logger("main")

BASELINE_DIR = "experiments/baseline"
FEEDBACK_DIR = "experiments/feedback_loop"


def run_analysis(query: str, dataset_path: str, mode: RunMode) -> dict:
    """Run analytics workflow in selected mode."""
    if mode == RunMode.BASELINE:
        result = ReliabilityFlow().run(query=query, dataset_path=dataset_path)
    elif mode == RunMode.FEEDBACK:
        result = FeedbackFlow().run(query=query, dataset_path=dataset_path)
    else:
        from research_agent.crews.analytics_crew.analytics_crew import run_analytics_crew
        from research_agent.tools.csv_tool import load_dataset_summary

        summary = load_dataset_summary(dataset_path)
        output = run_analytics_crew({
            "query": query,
            "dataset_path": dataset_path,
            "dataset_summary": summary,
            "plan": "",
            "feedback_context": "",
            "code_preview": "",
            "stdout": "",
            "stderr": "",
            "execution_status": "pending",
            "output_files": "[]",
            "evaluation_summary": "",
        })
        return {"mode": "full", "crew_output": output}

    return result.to_dict()


def run_benchmark(mode_str: str, limit: int | None, task_ids: list[str] | None) -> dict:
    mode = BenchmarkMode(mode_str)
    return BenchmarkRunner().run_suite(mode=mode, task_ids=task_ids, limit=limit)


def run_compare(args: argparse.Namespace) -> None:
    root = get_project_root()
    baseline_path = root / BASELINE_DIR
    feedback_path = root / FEEDBACK_DIR

    if args.list:
        df = list_benchmark_files(BASELINE_DIR, FEEDBACK_DIR)
        print(df.to_string(index=False))
        return

    if args.final:
        df, out_path = compare_final(BASELINE_DIR, FEEDBACK_DIR)
        print(df.to_string(index=False))
        print(f"Saved to {out_path}")
        return

    if args.task_id:
        b_file = find_benchmark_by_task_id(baseline_path, args.task_id)
        f_file = find_benchmark_by_task_id(feedback_path, args.task_id)
        if b_file is None:
            raise FileNotFoundError(
                f"No baseline benchmark found for task_id={args.task_id} in {baseline_path}"
            )
        if f_file is None:
            raise FileNotFoundError(
                f"No feedback_loop benchmark found for task_id={args.task_id} in {feedback_path}"
            )
        df, out_path = compare_two_files(b_file, f_file)
        print(df.to_string(index=False))
        print(f"Saved to {out_path}")
        return

    if args.baseline_id or args.feedback_id:
        if not args.baseline_id or not args.feedback_id:
            raise ValueError("Both --baseline-id and --feedback-id are required together")
        b_file = baseline_path / f"benchmark_{args.baseline_id}.json"
        f_file = feedback_path / f"benchmark_{args.feedback_id}.json"
        if not b_file.exists():
            raise FileNotFoundError(f"Baseline benchmark not found: {b_file}")
        if not f_file.exists():
            raise FileNotFoundError(f"Feedback benchmark not found: {f_file}")
        df, out_path = compare_two_files(b_file, f_file)
        print(df.to_string(index=False))
        print(f"Saved to {out_path}")
        return

    df, out_path = compare_experiments(BASELINE_DIR, FEEDBACK_DIR)
    print(df.to_string(index=False))
    print(f"Saved to {out_path}")


def cli_main() -> None:
    load_env()
    FileHandler().ensure_dirs()

    parser = argparse.ArgumentParser(description="LLM Data Science Agents Research System")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a single analytics task")
    run_p.add_argument("--query", required=True, help="Analytics task description")
    run_p.add_argument("--dataset", required=True, help="Path to CSV dataset")
    run_p.add_argument(
        "--mode",
        choices=[m.value for m in RunMode],
        default=RunMode.FEEDBACK.value,
        help="baseline | feedback | full",
    )

    bench_p = sub.add_parser("benchmark", help="Run benchmark task suite")
    bench_p.add_argument(
        "--mode",
        choices=[m.value for m in BenchmarkMode],
        default=BenchmarkMode.FEEDBACK_LOOP.value,
    )
    bench_p.add_argument("--limit", type=int, default=None)
    bench_p.add_argument("--task-ids", nargs="*", default=None)

    compare_p = sub.add_parser("compare", help="Compare baseline vs feedback-loop metrics")
    compare_mode = compare_p.add_mutually_exclusive_group()
    compare_mode.add_argument(
        "--task-id",
        help="Find latest benchmark files matching this task_id in both directories",
    )
    compare_mode.add_argument(
        "--final",
        action="store_true",
        help="Aggregate all paired per-task benchmarks into comparison_metrics_final.csv",
    )
    compare_mode.add_argument(
        "--list",
        action="store_true",
        help="List benchmark files with mode, benchmark_id, and task_id",
    )
    compare_p.add_argument(
        "--baseline-id",
        help="Benchmark timestamp id for baseline file (use with --feedback-id)",
    )
    compare_p.add_argument(
        "--feedback-id",
        help="Benchmark timestamp id for feedback_loop file (use with --baseline-id)",
    )

    init_p = sub.add_parser("init", help="Create directories and generate datasets")
    init_p.add_argument("--skip-datasets", action="store_true")

    summarize_p = sub.add_parser(
        "summarize-results", help="Aggregate benchmark results and generate research artifacts"
    )
    summarize_p.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory for artifacts (defaults to research_artifacts)",
    )
    summarize_p.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating matplotlib figures",
    )

    args = parser.parse_args()

    try:
        if args.command == "run":
            out = run_analysis(args.query, args.dataset, RunMode(args.mode))
            print(json.dumps(out, indent=2, default=str))
        elif args.command == "benchmark":
            out = run_benchmark(args.mode, args.limit, args.task_ids)
            print(json.dumps(out.get("metrics", out), indent=2))
        elif args.command == "compare":
            run_compare(args)
        elif args.command == "init":
            FileHandler().ensure_dirs()
            if not args.skip_datasets:
                from scripts.generate_datasets import main as gen_main

                gen_main()
            print("Project initialized.")
        elif args.command == "summarize-results":
            root = get_project_root()
            summarizer = ResultSummarizer(root)
            summarizer.summarize()
            print(f"Summaries and artifacts saved to {root / 'research_artifacts'}")
    except Exception as exc:
        logger.exception("Command failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
