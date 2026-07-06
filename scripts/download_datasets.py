#!/usr/bin/env python3
"""Download real-world CSV datasets (1500+ rows) into datasets/raw/."""

from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
MIN_ROWS = 1500

# Verified public sources (UCI ML Repository, IBM Open Data, Hands-On ML, etc.)
DATASETS: list[dict] = [
    {
        "filename": "telco_customer_churn.csv",
        "url": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "source": "IBM Telco Customer Churn (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)",
        "loader": "csv",
    },
    {
        "filename": "bank_marketing.csv",
        "url": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
        "source": "UCI Bank Marketing (https://archive.ics.uci.edu/dataset/222/bank+marketing)",
        "loader": "nested_zip_csv",
        "outer_zip": "bank-additional.zip",
        "inner_path": "bank-additional/bank-additional-full.csv",
        "read_kwargs": {"sep": ";"},
    },
    {
        "filename": "california_housing.csv",
        "url": "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",
        "source": "California Housing 1990 census (Hands-On Machine Learning)",
        "loader": "csv",
    },
    {
        "filename": "adult_income.csv",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "source": "UCI Adult Census Income (https://archive.ics.uci.edu/dataset/2/adult)",
        "loader": "csv",
        "read_kwargs": {"header": None, "skipinitialspace": True},
        "columns": [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education_num",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "native_country",
            "income",
        ],
    },
    {
        "filename": "credit_card_default.csv",
        "url": "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        "source": "UCI Default of Credit Card Clients (https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)",
        "loader": "zip_xls",
        "inner_path": "default of credit card clients.xls",
    },
    {
        "filename": "online_shoppers.csv",
        "url": "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
        "source": "UCI Online Shoppers Purchasing Intention (https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)",
        "loader": "zip_csv",
        "inner_path": "online_shoppers_intention.csv",
    },
]


def _fetch_bytes(url: str) -> bytes:
    result = subprocess.run(
        ["curl", "-fsSL", "--max-time", "180", url],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"Download failed for {url}: {err}")
    return result.stdout


def _load_dataset(spec: dict) -> pd.DataFrame:
    raw = _fetch_bytes(spec["url"])
    loader = spec["loader"]
    read_kwargs = dict(spec.get("read_kwargs", {}))

    if loader == "csv":
        text = raw.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(text), **read_kwargs)
    elif loader == "zip_csv":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            with zf.open(spec["inner_path"]) as f:
                df = pd.read_csv(f, **read_kwargs)
    elif loader == "nested_zip_csv":
        with zipfile.ZipFile(io.BytesIO(raw)) as outer:
            inner_bytes = outer.read(spec["outer_zip"])
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
            with inner.open(spec["inner_path"]) as f:
                df = pd.read_csv(f, **read_kwargs)
    elif loader == "zip_xls":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            with zf.open(spec["inner_path"]) as f:
                df = pd.read_excel(f, header=1)
    else:
        raise ValueError(f"Unknown loader: {loader}")

    if "columns" in spec:
        df.columns = spec["columns"]

    df = df.dropna(how="all")
    if len(df.columns) > 0:
        df = df[~df.iloc[:, 0].astype(str).str.match(r"^\s*$", na=False)]

    # Telco: TotalCharges sometimes stored as blank string
    if spec["filename"] == "telco_customer_churn.csv" and "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Credit card: use readable column names from UCI header row
    if spec["filename"] == "credit_card_default.csv":
        df = df.rename(columns={"default payment next month": "default_payment_next_month"})

    return df


def download_all() -> list[Path]:
    RAW.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for spec in DATASETS:
        name = spec["filename"]
        path = RAW / name
        print(f"Downloading {name}...")
        print(f"  Source: {spec['source']}")
        df = _load_dataset(spec)
        if len(df) < MIN_ROWS:
            raise ValueError(f"{name}: only {len(df)} rows (need >={MIN_ROWS})")
        df.to_csv(path, index=False)
        saved.append(path)
        print(f"  Saved {path} ({len(df):,} rows, {len(df.columns)} columns)")

    return saved


def main() -> None:
    try:
        paths = download_all()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"\nDownloaded {len(paths)} datasets to {RAW}")


if __name__ == "__main__":
    main()
