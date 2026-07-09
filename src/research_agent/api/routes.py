"""FastAPI routes for the research agent system."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from research_agent.evaluation.benchmark_runner import BenchmarkRunner
from research_agent.evaluation.comparison_metrics import compare_experiments
from research_agent.main import run_analysis
from research_agent.models.request_schema import (
    AnalysisRequest,
    BenchmarkRequest,
    CompareExperimentsRequest,
)
from research_agent.utils.file_handler import FileHandler
from research_agent.utils.helpers import load_env

load_env()
FileHandler().ensure_dirs()

app = FastAPI(
    title="Research Agent System API",
    description="Multi-agent LLM data science system with structured feedback loops",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalysisRequest) -> dict[str, Any]:
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")
    try:
        return run_analysis(request.query, request.dataset_path, request.mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/benchmark")
def benchmark(request: BenchmarkRequest) -> dict[str, Any]:
    try:
        return BenchmarkRunner().run_suite(
            mode=request.mode,
            task_ids=request.task_ids,
            limit=request.limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/compare")
def compare(request: CompareExperimentsRequest) -> dict[str, Any]:
    try:
        df, out_path = compare_experiments(
            request.baseline_dir,
            request.feedback_dir,
            request.output_path,
        )
        return {"metrics": df.to_dict(orient="records"), "output_path": str(out_path)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/datasets")
def list_datasets() -> dict[str, list[str]]:
    paths = FileHandler().list_datasets()
    return {"datasets": [str(p) for p in paths]}


def run_server() -> None:
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("research_agent.api.routes:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
