"""Shared pytest fixtures: isolated MLflow tracking + synthetic dataset."""

import mlflow
import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def tmp_mlflow_uri(tmp_path, monkeypatch):
    """Isolate MLflow tracking per test using tmp_path."""
    uri = f"file:{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test-experiment")
    return uri


@pytest.fixture
def synthetic_data():
    """Small deterministic dataset for unit tests (sklearn built-in)."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=6,
        random_state=42,
    )
    return X, y


@pytest.fixture
def synthetic_dataframe():
    """Synthetic dataset matching Churn schema (for data.py tests)."""
    return pd.DataFrame(
        {
            "CreditScore": [650, 720, 580, 680] * 25,
            "Geography": ["France", "Spain", "Germany", "France"] * 25,
            "Gender": ["Male", "Female"] * 50,
            "Age": [40, 28, 62, 42] * 25,
            "Tenure": [5, 3, 8, 5] * 25,
            "Balance": [50000.0, 150000.0, 5000.0, 75000.0] * 25,
            "NumOfProducts": [1, 2, 1, 2] * 25,
            "HasCrCard": [1, 1, 0, 1] * 25,
            "IsActiveMember": [1, 1, 0, 1] * 25,
            "EstimatedSalary": [80000.0, 95000.0, 45000.0, 110000.0] * 25,
            "Exited": [0, 0, 1, 0] * 25,
        }
    )
