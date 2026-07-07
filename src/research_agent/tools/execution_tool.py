"""Safe isolated code execution via subprocess."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from research_agent.models.result_schema import ExecutionResult, ExecutionStatus
from research_agent.utils.helpers import get_env_int, get_project_root
from research_agent.utils.logger import setup_logger

logger = setup_logger("execution_tool")

BLOCKED_PATTERNS = [
    r"\bos\.system\b",
    r"\bsubprocess\b",
    r"\bshutil\.rmtree\b",
    r"\bos\.remove\b",
    r"\bos\.unlink\b",
    r"\bsocket\b",
    r"\brequests\.(get|post|put|delete)\b",
    r"\burllib\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\b__import__\s*\(",
    r"\bopen\s*\([^)]*['\"]w",
]

BLOCKED_IMPORTS = {"subprocess", "socket", "requests", "urllib", "ftplib", "telnetlib"}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".html", ".pdf"}


def _base_run_id(run_id: str) -> str:
    """Strip iteration suffix (e.g. abc123_i2 -> abc123) for stable artifact folders."""
    return re.sub(r"_i\d+$", "", run_id)


class ExecutionToolInput(BaseModel):
    code: str = Field(..., description="Python code to execute")
    run_id: str = Field(default="run", description="Run identifier for workspace")
    dataset_path: str | None = Field(default=None, description="Optional dataset to copy into workspace")


class SafeCodeExecutor:
    """Execute Python in an isolated workspace with safety checks."""

    def __init__(self, timeout: int | None = None) -> None:
        self.timeout = timeout or get_env_int("EXECUTION_TIMEOUT_SECONDS", 120)
        self.root = get_project_root() / "execution_workspace"

    def validate_code(self, code: str) -> tuple[bool, str | None]:
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Blocked pattern detected: {pattern}"
        for imp in BLOCKED_IMPORTS:
            if re.search(rf"^\s*import\s+{imp}\b|^\s*from\s+{imp}\b", code, re.MULTILINE):
                return False, f"Blocked import: {imp}"
        return True, None

    def _resolve_dataset_source(self, dataset_path: str) -> Path:
        """Resolve dataset path, treating relative paths as project-root relative."""
        src = Path(dataset_path)
        if not src.is_absolute():
            src = get_project_root() / src
        return src.resolve()

    def _copy_dataset_to_workspace(self, workspace: Path, dataset_path: str) -> str | None:
        """
        Copy dataset into workspace, preserving project-relative layout.
        Returns the workspace-relative path (for DATASET_PATH), or None if missing.
        """
        src = self._resolve_dataset_source(dataset_path)
        if not src.exists():
            logger.warning("Dataset not found at %s", src)
            return None

        project_root = get_project_root()
        try:
            rel = src.relative_to(project_root)
        except ValueError:
            rel = Path(src.name)

        dest = workspace / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        return str(rel)

    def _setup_artifact_dirs(self, run_id: str, artifacts_run_id: str | None) -> Path:
        """Project-level directory for figures produced by this run."""
        base_id = artifacts_run_id or _base_run_id(run_id)
        figures_dir = get_project_root() / "reports" / "figures" / base_id
        figures_dir.mkdir(parents=True, exist_ok=True)
        return figures_dir

    def _figures_relative_paths(self, figures_dir: Path) -> list[str]:
        """List image files already under the project figures directory."""
        project_root = get_project_root()
        return [
            str(p.relative_to(project_root))
            for p in sorted(figures_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]

    def _promote_artifacts(self, workspace: Path, figures_dir: Path) -> list[str]:
        """Copy image outputs from workspace into project reports/figures/."""
        project_root = get_project_root()
        promoted: list[str] = []
        seen_names: set[str] = {p.name for p in figures_dir.iterdir() if p.is_file()}

        for src in workspace.rglob("*"):
            if not src.is_file() or src.name == "analysis.py":
                continue
            if src.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            name = src.name
            if name in seen_names:
                stem, suffix = src.stem, src.suffix
                name = f"{stem}_{len(seen_names)}{suffix}"
            seen_names.add(name)

            dest = figures_dir / name
            shutil.copy2(src, dest)
            promoted.append(str(dest.relative_to(project_root)))

        return promoted

    def run(
        self,
        code: str,
        run_id: str,
        dataset_path: str | None = None,
        artifacts_run_id: str | None = None,
    ) -> ExecutionResult:
        valid, reason = self.validate_code(code)
        if not valid:
            logger.warning("Code blocked: %s", reason)
            return ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                stderr=reason or "blocked",
                error_message=reason,
            )

        workspace = self.root / run_id
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        script_path = workspace / "analysis.py"
        script_path.write_text(code, encoding="utf-8")

        figures_dir = self._setup_artifact_dirs(run_id, artifacts_run_id)
        project_root = get_project_root()

        env = os.environ.copy()
        env["PROJECT_ROOT"] = str(project_root)
        env["FIGURES_DIR"] = str(figures_dir)
        if dataset_path:
            workspace_dataset = self._copy_dataset_to_workspace(workspace, dataset_path)
            if workspace_dataset:
                env["DATASET_PATH"] = workspace_dataset

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            duration = time.perf_counter() - start
            workspace_files = [
                str(p.relative_to(workspace))
                for p in workspace.rglob("*")
                if p.is_file() and p.name != "analysis.py"
            ]
            promoted = self._promote_artifacts(workspace, figures_dir)
            figure_outputs = self._figures_relative_paths(figures_dir)
            output_files = list(
                dict.fromkeys(workspace_files + promoted + figure_outputs)
            )
            status = ExecutionStatus.SUCCESS if proc.returncode == 0 else ExecutionStatus.FAILURE
            return ExecutionResult(
                status=status,
                stdout=proc.stdout,
                stderr=proc.stderr,
                return_code=proc.returncode,
                duration_seconds=duration,
                output_files=output_files,
                error_message=proc.stderr if proc.returncode != 0 else None,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                duration_seconds=duration,
                error_message=f"Execution timed out after {self.timeout}s",
            )
        except Exception as exc:
            duration = time.perf_counter() - start
            return ExecutionResult(
                status=ExecutionStatus.FAILURE,
                duration_seconds=duration,
                error_message=str(exc),
            )


class CodeExecutionTool(BaseTool):
    name: str = "code_execution_tool"
    description: str = (
        "Execute Python data science code safely in an isolated workspace. "
        "Returns stdout, stderr, status, and generated output files."
    )
    args_schema: type[BaseModel] = ExecutionToolInput

    def _run(
        self,
        code: str,
        run_id: str = "run",
        dataset_path: str | None = None,
        artifacts_run_id: str | None = None,
    ) -> str:
        executor = SafeCodeExecutor()
        result = executor.run(
            code=code,
            run_id=run_id,
            dataset_path=dataset_path,
            artifacts_run_id=artifacts_run_id,
        )
        return result.model_dump_json()


def execute_code(
    code: str,
    run_id: str,
    dataset_path: str | None = None,
    artifacts_run_id: str | None = None,
) -> ExecutionResult:
    """Programmatic execution helper (non-CrewAI)."""
    return SafeCodeExecutor().run(
        code=code,
        run_id=run_id,
        dataset_path=dataset_path,
        artifacts_run_id=artifacts_run_id,
    )
