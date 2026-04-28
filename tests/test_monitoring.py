import mlflow
import numpy as np
import pandas as pd
import pytest

from mlops_churn import config, monitoring


@pytest.fixture
def reference_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "CreditScore": rng.normal(650, 80, 500),
            "Age": rng.normal(40, 10, 500),
            "Balance": rng.normal(50000, 20000, 500),
        }
    )


def test_drift_score_zero_for_identical(reference_df):
    """Identical distributions → drift score near 0."""
    score = monitoring.compute_drift(
        reference_df, reference_df.copy(), reference_df.columns.tolist()
    )
    assert score < 0.05


def test_drift_score_high_for_shifted(reference_df):
    """Significantly shifted distributions → drift score > 0.5."""
    shifted = reference_df.copy()
    shifted["CreditScore"] += 200
    shifted["Age"] += 25
    shifted["Balance"] += 50000
    score = monitoring.compute_drift(reference_df, shifted, reference_df.columns.tolist())
    assert score > 0.5


def test_log_batch_metrics_writes_5_metrics(tmp_mlflow_uri):
    """log_batch_metrics writes 5 expected metrics."""
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        monitoring.log_batch_metrics(
            run_id=run_id,
            step=0,
            prediction_count=50,
            latency_ms_list=[40.0, 45.0, 50.0, 55.0, 60.0] * 10,
            labels=[0, 1, 0, 0, 1] * 10,
            drift_score=0.05,
        )

    fetched = mlflow.get_run(run_id)
    metrics = fetched.data.metrics
    for m in ["prediction_count", "latency_p50_ms", "latency_p95_ms", "churn_rate", "drift_score"]:
        assert m in metrics, f"missing metric {m}"


def test_log_batch_metrics_sets_alert_tag_when_drift_exceeds_threshold(tmp_mlflow_uri):
    """When drift_score > threshold, run gets tag alert=true."""
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        monitoring.log_batch_metrics(
            run_id=run_id,
            step=0,
            prediction_count=50,
            latency_ms_list=[40.0] * 50,
            labels=[0] * 50,
            drift_score=config.DRIFT_SCORE_THRESHOLD + 0.05,
        )
    fetched = mlflow.get_run(run_id)
    assert fetched.data.tags.get("alert") == "true"
