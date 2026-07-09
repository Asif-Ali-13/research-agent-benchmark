"""Run benchmark task suite and persist results."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_agent.evaluation.reliability_metrics import ReliabilityMetrics
from research_agent.flows.feedback_flow import FeedbackFlow
from research_agent.flows.reliability_flow import ReliabilityFlow
from research_agent.models.request_schema import BenchmarkMode
from research_agent.models.result_schema import AgentRunResult
from research_agent.utils.file_handler import FileHandler
from research_agent.utils.helpers import get_env_float, get_project_root, resolve_dataset_path
from research_agent.utils.llm_retry import is_rate_limit_error
from research_agent.utils.logger import setup_logger

logger = setup_logger("benchmark_runner")


class BenchmarkRunner:
    """Execute 20-30 structured analytics benchmark tasks."""

    def __init__(self) -> None:
        self.root = get_project_root()
        self.file_handler = FileHandler()
        self.tasks_path = self.root / "datasets" / "benchmark_tasks" / "tasks.json"

    def load_tasks(self, task_ids: list[str] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        tasks = self.file_handler.read_json(self.tasks_path)
        if task_ids:
            tasks = [t for t in tasks if t["id"] in task_ids]
        if limit:
            tasks = tasks[:limit]
        return tasks

    def _output_dir(self, mode: BenchmarkMode) -> Path:
        sub = "baseline" if mode == BenchmarkMode.BASELINE else "feedback_loop"
        out = self.root / "experiments" / sub
        out.mkdir(parents=True, exist_ok=True)
        return out

    def run_task(self, task: dict[str, Any], mode: BenchmarkMode) -> AgentRunResult:
        dataset_path = resolve_dataset_path(task["dataset"])
        query = task["query"]

        if mode == BenchmarkMode.BASELINE:
            flow = ReliabilityFlow()
            result = flow.run(
                query=query,
                dataset_path=dataset_path,
                task_id=task["id"],
                expected_outputs=task.get("expected_outputs"),
            )
        else:
            flow = FeedbackFlow()
            result = flow.run(
                query=query,
                dataset_path=dataset_path,
                task_id=task["id"],
                expected_outputs=task.get("expected_outputs"),
            )
        return result

    def run_suite(
        self,
        mode: BenchmarkMode = BenchmarkMode.FEEDBACK_LOOP,
        task_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        tasks = self.load_tasks(task_ids=task_ids, limit=limit)
        results: list[AgentRunResult] = []

        logger.info("Running %d benchmark tasks in mode=%s", len(tasks), mode.value)

        delay = get_env_float("BENCHMARK_TASK_DELAY_SECONDS", 12.0)

        for i, task in enumerate(tasks, 1):
            logger.info("[%d/%d] Task %s", i, len(tasks), task["id"])
            result: AgentRunResult | None = None
            last_exc: Exception | None = None
            for attempt in range(1, 4):
                try:
                    result = self.run_task(task, mode)
                    break
                except Exception as exc:
                    last_exc = exc
                    if is_rate_limit_error(exc) and attempt < 3:
                        wait = 35.0 * attempt
                        logger.warning(
                            "Task %s rate limited, retry %d/3 after %.0fs",
                            task["id"],
                            attempt,
                            wait,
                        )
                        time.sleep(wait)
                        continue
                    logger.exception("Task %s failed: %s", task["id"], exc)
                    result = AgentRunResult(
                        run_id=str(uuid.uuid4())[:8],
                        mode=mode.value,
                        query=task["query"],
                        dataset_path=resolve_dataset_path(task["dataset"]),
                        metadata={"task_id": task["id"], "error": str(exc)},
                    )
                    break

            if result is None and last_exc is not None:
                result = AgentRunResult(
                    run_id=str(uuid.uuid4())[:8],
                    mode=mode.value,
                    query=task["query"],
                    dataset_path=resolve_dataset_path(task["dataset"]),
                    metadata={"task_id": task["id"], "error": str(last_exc)},
                )
            results.append(result)

            if i < len(tasks) and delay > 0:
                logger.info("Waiting %.0fs before next task (rate-limit spacing)", delay)
                time.sleep(delay)

        metrics = ReliabilityMetrics.from_results(results)
        summary = {
            "run_id": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            "mode": mode.value,
            "task_count": len(tasks),
            "metrics": metrics.summary(),
            "results": [r.to_dict() for r in results],
        }

        out_dir = self._output_dir(mode)
        out_path = out_dir / f"benchmark_{summary['run_id']}.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.info("Saved benchmark results to %s", out_path)
        return summary
