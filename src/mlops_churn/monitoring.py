"""Drift detection (KS test) + batch metric logging to MLflow."""

import numpy as np
import pandas as pd
from mlflow import MlflowClient
from scipy.stats import ks_2samp

from mlops_churn import config


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_cols: list[str],
) -> float:
    """Mean Kolmogorov-Smirnov statistic across numeric columns. 0=identical, ~1=very different.

    Uses scipy.stats.ks_2samp (existing standard test).
    """
    if not numeric_cols:
        return 0.0
    scores = [ks_2samp(reference[col], current[col]).statistic for col in numeric_cols]
    return float(sum(scores) / len(scores))


def log_batch_metrics(
    run_id: str,
    step: int,
    prediction_count: int,
    latency_ms_list: list[float],
    labels: list[int],
    drift_score: float,
) -> None:
    """Log 5 monitoring metrics for one batch + set alert tag if drift exceeds threshold."""
    client = MlflowClient()

    p50 = float(np.percentile(latency_ms_list, 50))
    p95 = float(np.percentile(latency_ms_list, 95))
    churn_rate = float(sum(labels) / len(labels)) if labels else 0.0

    metrics = {
        "prediction_count": float(prediction_count),
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "churn_rate": churn_rate,
        "drift_score": drift_score,
    }
    for name, value in metrics.items():
        client.log_metric(run_id, name, value, step=step)

    if drift_score > config.DRIFT_SCORE_THRESHOLD:
        client.set_tag(run_id, "alert", "true")
