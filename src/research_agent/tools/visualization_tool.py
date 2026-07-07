"""Visualization helpers for agents and evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from research_agent.utils.helpers import get_project_root


class VizToolInput(BaseModel):
    dataset_path: str
    output_dir: str = "reports/figures"
    chart_type: str = Field(default="pairplot", description="histogram | correlation | pairplot")


class VisualizationTool(BaseTool):
    name: str = "visualization_tool"
    description: str = "Generate EDA charts from a CSV and save to reports/figures."
    args_schema: type[BaseModel] = VizToolInput

    def _run(self, dataset_path: str, output_dir: str = "reports/figures", chart_type: str = "pairplot") -> str:
        root = get_project_root()
        df = pd.read_csv(dataset_path)
        out = root / output_dir
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(dataset_path).stem

        saved: list[str] = []
        numeric = df.select_dtypes(include="number")

        if chart_type == "histogram" and not numeric.empty:
            col = numeric.columns[0]
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            path = out / f"{stem}_hist.png"
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            saved.append(str(path))

        elif chart_type == "correlation" and numeric.shape[1] >= 2:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            path = out / f"{stem}_corr.png"
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            saved.append(str(path))

        elif chart_type == "pairplot" and numeric.shape[1] >= 2:
            cols = list(numeric.columns[:4])
            g = sns.pairplot(df[cols].dropna())
            path = out / f"{stem}_pairplot.png"
            g.savefig(path)
            plt.close("all")
            saved.append(str(path))

        if not saved:
            return "No numeric columns available for visualization."
        return f"Saved charts: {saved}"
