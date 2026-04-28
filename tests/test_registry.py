import mlflow
import pytest
from mlflow.exceptions import MlflowException

from mlops_churn import config, registry


@pytest.fixture
def two_logged_models(tmp_mlflow_uri, synthetic_data):
    """Helper: log 2 simple models to MLflow + register both. Returns (v1, v2)."""
    from sklearn.linear_model import LogisticRegression

    X, y = synthetic_data
    versions = []
    for _ in range(2):
        with mlflow.start_run() as run:
            model = LogisticRegression(max_iter=200).fit(X, y)
            mlflow.sklearn.log_model(model, name="model")
        v = registry.register_run(run.info.run_id)
        versions.append(v)
    return versions


def test_set_alias_assigns_correctly(two_logged_models):
    v1, _v2 = two_logged_models
    registry.set_alias(config.ALIAS_PRODUCTION, v1)
    mv = registry.get_version_by_alias(config.ALIAS_PRODUCTION)
    assert mv.version == v1


def test_set_alias_atomic_move(two_logged_models):
    v1, v2 = two_logged_models
    registry.set_alias(config.ALIAS_PRODUCTION, v1)
    registry.set_alias(config.ALIAS_PRODUCTION, v2)  # move
    mv = registry.get_version_by_alias(config.ALIAS_PRODUCTION)
    assert mv.version == v2


def test_remove_alias(two_logged_models):
    v1, _v2 = two_logged_models
    registry.set_alias(config.ALIAS_STAGING, v1)
    registry.remove_alias(config.ALIAS_STAGING)
    with pytest.raises(MlflowException):
        registry.get_version_by_alias(config.ALIAS_STAGING)
