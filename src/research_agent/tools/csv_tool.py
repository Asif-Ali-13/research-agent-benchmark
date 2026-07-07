"""CSV dataset inspection and preprocessing tools."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from research_agent.utils.helpers import get_project_root


class CSVToolInput(BaseModel):
    dataset_path: str = Field(..., description="Path to CSV file")
    n_rows: int = Field(default=5, description="Sample rows to preview")


class CSVInspectionTool(BaseTool):
    name: str = "csv_inspection_tool"
    description: str = (
        "Load a CSV dataset and return schema, missing values, dtypes, and sample rows."
    )
    args_schema: type[BaseModel] = CSVToolInput

    def _run(self, dataset_path: str, n_rows: int = 5) -> str:
        path = Path(dataset_path)
        if not path.is_absolute():
            path = get_project_root() / path
        if not path.exists():
            return f"Error: dataset not found at {dataset_path}"

        df = pd.read_csv(path)
        info = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "missing": df.isnull().sum().to_dict(),
            "numeric_summary": df.describe(include="number").to_dict()
            if df.select_dtypes("number").shape[1] > 0
            else {},
            "sample": df.head(n_rows).to_dict(orient="records"),
        }
        lines = [
            f"Shape: {info['shape']}",
            f"Columns: {info['columns']}",
            f"Missing values: {info['missing']}",
            f"Dtypes: {info['dtypes']}",
            f"Sample rows: {info['sample']}",
        ]
        return "\n".join(lines)


def load_dataset_summary(dataset_path: str, n_rows: int = 5) -> str:
    """Non-tool helper for flows."""
    return CSVInspectionTool()._run(dataset_path=dataset_path, n_rows=n_rows)
