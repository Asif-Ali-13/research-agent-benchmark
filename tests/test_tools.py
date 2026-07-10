"""Tests for research agent tools."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from research_agent.tools.csv_tool import load_dataset_summary
from research_agent.tools.execution_tool import SafeCodeExecutor, execute_code
from research_agent.tools.model_tool import train_baseline_model
from research_agent.utils.helpers import get_project_root


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "target": [0, 1, 0]})
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


def test_csv_summary(sample_csv: Path) -> None:
    summary = load_dataset_summary(str(sample_csv))
    assert "Shape" in summary
    assert "target" in summary


def test_safe_execution_success(sample_csv: Path) -> None:
    code = f"""
import pandas as pd
df = pd.read_csv('{sample_csv}')
print(df.shape)
"""
    result = execute_code(code=code, run_id="test_success", dataset_path=str(sample_csv))
    assert result.status.value == "success"
    assert "3" in result.stdout


def test_safe_execution_project_relative_dataset_path() -> None:
    dataset = get_project_root() / "datasets" / "raw" / "iris.csv"
    if not dataset.exists():
        pytest.skip("iris.csv not available")

    code = """
import pandas as pd
df = pd.read_csv("datasets/raw/iris.csv")
print(df.shape)
"""
    result = execute_code(
        code=code,
        run_id="test_rel_path",
        dataset_path="datasets/raw/iris.csv",
    )
    assert result.status.value == "success"
    assert "150" in result.stdout


def test_safe_execution_promotes_figures_to_reports() -> None:
    code = """
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig_dir = os.environ["FIGURES_DIR"]
os.makedirs(fig_dir, exist_ok=True)
plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
plt.savefig(os.path.join(fig_dir, "line_plot.png"))
plt.close()
print("saved")
"""
    result = execute_code(
        code=code,
        run_id="test_fig_i0",
        dataset_path="datasets/raw/iris.csv",
        artifacts_run_id="test_fig",
    )
    assert result.status.value == "success"
    dest = get_project_root() / "reports" / "figures" / "test_fig" / "line_plot.png"
    assert dest.exists()
    assert any("reports/figures/test_fig/line_plot.png" in f for f in result.output_files)


def test_safe_execution_blocks_dangerous_code() -> None:
    executor = SafeCodeExecutor()
    valid, reason = executor.validate_code("import subprocess\nsubprocess.run(['ls'])")
    assert valid is False
    assert reason is not None


def test_model_training(sample_csv: Path) -> None:
    metrics = train_baseline_model(str(sample_csv), "target", "classification")
    assert "accuracy" in metrics


def test_benchmark_tasks_file_exists() -> None:
    path = get_project_root() / "datasets" / "benchmark_tasks" / "tasks.json"
    assert path.exists()
    tasks = json.loads(path.read_text())
    assert len(tasks) >= 20
