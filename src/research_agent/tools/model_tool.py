"""Sklearn model training and evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class ModelToolInput(BaseModel):
    dataset_path: str
    target_column: str
    task_type: str = Field(default="auto", description="classification | regression | auto")


class ModelTrainingTool(BaseTool):
    name: str = "model_training_tool"
    description: str = (
        "Train a baseline RandomForest model on a CSV with target column. "
        "Returns JSON metrics."
    )
    args_schema: type[BaseModel] = ModelToolInput

    def _run(self, dataset_path: str, target_column: str, task_type: str = "auto") -> str:
        df = pd.read_csv(dataset_path)
        if target_column not in df.columns:
            return json.dumps({"error": f"Target column '{target_column}' not found"})

        y = df[target_column]
        X = df.drop(columns=[target_column])

        if task_type == "auto":
            task_type = "classification" if y.dtype == "object" or y.nunique() <= 20 else "regression"

        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical = [c for c in X.columns if c not in numeric]

        transformers = []
        if numeric:
            transformers.append((
                "num", 
                Pipeline([
                    ("imputer", SimpleImputer()), 
                    ("scaler", StandardScaler())
                ]),
                numeric
            ))
        if categorical:
            transformers.append((
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ))

        preprocessor = ColumnTransformer(transformers)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(preprocessor.fit_transform(X_train), y_train)
            preds = model.predict(preprocessor.transform(X_test))
            metrics = {
                "task_type": "classification",
                "accuracy": float(accuracy_score(y_test, preds)),
                "f1_macro": float(f1_score(y_test, preds, average="macro")),
            }
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(preprocessor.fit_transform(X_train), y_train)
            preds = model.predict(preprocessor.transform(X_test))
            metrics = {
                "task_type": "regression",
                "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
                "r2": float(r2_score(y_test, preds)),
            }

        return json.dumps(metrics)


def train_baseline_model(dataset_path: str, target_column: str, task_type: str = "auto") -> dict:
    raw = ModelTrainingTool()._run(
        dataset_path=dataset_path,
        target_column=target_column,
        task_type=task_type,
    )
    return json.loads(raw)
