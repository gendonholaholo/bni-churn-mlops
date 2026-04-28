import mlflow
import pytest
from sklearn.linear_model import LogisticRegression

from mlops_churn import config, registry, serving


@pytest.fixture
def production_model(tmp_mlflow_uri, synthetic_data, numeric_only_config):
    """Train + register + alias 'production'. Returns features_dict.

    Fits on a DataFrame with named columns to ensure sklearn stores feature names,
    avoiding warnings during pyfunc inference downstream.
    """
    import pandas as pd

    X, y = synthetic_data
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    with mlflow.start_run() as run:
        model = LogisticRegression(max_iter=200).fit(X_df, y)
        mlflow.sklearn.log_model(model, name="model", input_example=X_df.head())
    v = registry.register_run(run.info.run_id)
    registry.set_alias(config.ALIAS_PRODUCTION, v)
    serving._cache = None  # reset cache between tests
    return {name: float(X_df.iloc[0][name]) for name in feature_names}


def test_predict_returns_dict_with_required_keys(production_model):
    """predict returns dict with prob, label, latency_ms keys."""
    out = serving.predict(production_model, alias=config.ALIAS_PRODUCTION)
    assert set(out.keys()) >= {"prob", "label", "latency_ms"}
    assert 0.0 <= out["prob"] <= 1.0
    assert out["label"] in (0, 1)
    assert out["latency_ms"] >= 0.0


def test_predict_ab_returns_both_versions(
    production_model, tmp_mlflow_uri, synthetic_data, numeric_only_config
):
    """predict_ab returns dict with production, staging, agreement keys."""
    import pandas as pd

    X, y = synthetic_data
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    with mlflow.start_run() as run:
        model = LogisticRegression(max_iter=200, C=0.1).fit(X_df, y)
        mlflow.sklearn.log_model(model, name="model", input_example=X_df.head())
    v2 = registry.register_run(run.info.run_id)
    registry.set_alias(config.ALIAS_STAGING, v2)
    serving._cache = None

    out = serving.predict_ab(production_model)
    assert set(out.keys()) == {"production", "staging", "agreement"}
    assert isinstance(out["agreement"], bool)
