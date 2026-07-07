"""SQLite-backed experiment and run memory."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_agent.utils.helpers import get_project_root


class MemoryManager:
    """Store experiment runs in SQLite for tracking and comparison."""

    def __init__(self, db_path: Path | None = None) -> None:
        root = get_project_root()
        self.db_path = db_path or (root / "experiments" / "results" / "experiments.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    task_id TEXT,
                    query TEXT,
                    dataset_path TEXT,
                    completed INTEGER DEFAULT 0,
                    execution_success INTEGER DEFAULT 0,
                    score REAL,
                    recovered INTEGER DEFAULT 0,
                    iterations INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    payload TEXT
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    score REAL,
                    passed INTEGER,
                    payload TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                """
            )

    def save_run(self, run_id: str, mode: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, mode, task_id, query, dataset_path, completed,
                 execution_success, score, recovered, iterations, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    mode,
                    payload.get("task_id"),
                    payload.get("query"),
                    payload.get("dataset_path"),
                    int(payload.get("completed", False)),
                    int(payload.get("execution_success", False)),
                    payload.get("score"),
                    int(payload.get("recovered_from_failure", False)),
                    payload.get("iterations", 1),
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(payload, default=str),
                ),
            )

    def save_feedback(self, run_id: str, iteration: int, feedback: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (run_id, iteration, score, passed, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    iteration,
                    feedback.get("score"),
                    int(feedback.get("passed", False)),
                    json.dumps(feedback, default=str),
                ),
            )

    def list_runs(self, mode: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if mode:
                rows = conn.execute("SELECT * FROM runs WHERE mode = ?", (mode,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM runs").fetchall()
        return [dict(r) for r in rows]
