import mlflow

from mlops_churn import config, train


def test_train_one_returns_valid_run_id(tmp_mlflow_uri, monkeypatch, synthetic_data):
    """train_one returns run_id that can be fetched from MLflow."""
    X, y = synthetic_data
    monkeypatch.setattr(train, "_load_train_val", lambda: _split_synthetic(X, y))

    run_id = train.train_one("logistic_regression", {"C": 1.0, "max_iter": 200})

    assert isinstance(run_id, str)
    fetched = mlflow.get_run(run_id)
    assert fetched.info.run_id == run_id


def test_train_one_logs_5_metrics(tmp_mlflow_uri, monkeypatch, synthetic_data):
    """train_one logs accuracy, f1, roc_auc, precision, recall."""
    X, y = synthetic_data
    monkeypatch.setattr(train, "_load_train_val", lambda: _split_synthetic(X, y))

    run_id = train.train_one(
        "random_forest",
        {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2},
    )

    fetched = mlflow.get_run(run_id)
    metrics = fetched.data.metrics
    for m in ["accuracy", "f1", "roc_auc", "precision", "recall"]:
        assert m in metrics, f"missing metric {m}"


def _split_synthetic(X, y):
    """Helper to mimic _load_train_val for tests."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df[config.TARGET] = y
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[config.TARGET]
    )
    return train_df, val_df
