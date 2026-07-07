"""File I/O helpers for datasets, code artifacts, and reports."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_agent.utils.helpers import get_project_root, sanitize_filename


class FileHandler:
    """Manage project paths and artifact persistence."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or get_project_root()
        self.datasets_raw = self.root / "datasets" / "raw"
        self.datasets_processed = self.root / "datasets" / "processed"
        self.benchmark_tasks = self.root / "datasets" / "benchmark_tasks"
        self.reports = self.root / "reports"
        self.experiments = self.root / "experiments" / "results"
        self.execution_workspace = self.root / "execution_workspace"

    def ensure_dirs(self) -> None:
        for path in [
            self.datasets_raw,
            self.datasets_processed,
            self.benchmark_tasks,
            self.reports / "figures",
            self.reports / "tables",
            self.reports / "paper_draft",
            self.experiments,
            self.execution_workspace,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def read_text(self, path: Path | str) -> str:
        return Path(path).read_text(encoding="utf-8")

    def write_text(self, path: Path | str, content: str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def read_json(self, path: Path | str) -> Any:
        return json.loads(self.read_text(path))

    def write_json(self, path: Path | str, data: Any) -> Path:
        return self.write_text(path, json.dumps(data, indent=2, default=str))

    def create_run_workspace(self, run_id: str) -> Path:
        """Isolated directory for code execution."""
        ws = self.execution_workspace / sanitize_filename(run_id)
        if ws.exists():
            shutil.rmtree(ws)
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    def save_generated_code(self, run_id: str, code: str, iteration: int = 0) -> Path:
        ws = self.execution_workspace / sanitize_filename(run_id)
        ws.mkdir(parents=True, exist_ok=True)
        path = ws / f"analysis_iter_{iteration}.py"
        path.write_text(code, encoding="utf-8")
        return path

    def save_report(self, content: str, name: str | None = None) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = name or f"report_{ts}.md"
        path = self.reports / filename
        return self.write_text(path, content)

    def list_datasets(self) -> list[Path]:
        self.datasets_raw.mkdir(parents=True, exist_ok=True)
        return sorted(self.datasets_raw.glob("*.csv"))
