"""Train one model with given algo + params, log to MLflow."""

from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from mlops_churn import config, data

_MODEL_CLASS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}


def _load_train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train/val. Override in tests via monkeypatch."""
    train, val, _test = data.load_processed()
    return train, val


def _compact_param_str(params: dict[str, Any]) -> str:
    """e.g., {'C': 0.1, 'max_iter': 500} -> 'C0.10-max_iter500'."""
    parts = []
    for k, v in params.items():
        v_str = f"{v:.2f}" if isinstance(v, float) else str(v)
        parts.append(f"{k}{v_str}")
    return "-".join(parts)


def _build_pipeline(algo: str, params: dict[str, Any]) -> Pipeline:
    """Build a sklearn Pipeline: preprocessing + model.

    Preprocessing handles raw Churn features:
    - Numeric columns (CreditScore, Age, Tenure, Balance, NumOfProducts,
      EstimatedSalary): StandardScaler
    - Categorical columns (Geography, Gender): OneHotEncoder
    - Binary columns (HasCrCard, IsActiveMember): pass-through (already 0/1)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",  # binary cols pass through
    )
    model = _MODEL_CLASS[algo](**params, random_state=42)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_one(algo: str, params: dict[str, Any], source: str = "gradio-lab") -> str:
    """Train one model + log to MLflow. Returns run_id."""
    if algo not in _MODEL_CLASS:
        raise ValueError(f"Unknown algo {algo!r}. Choices: {list(_MODEL_CLASS)}")

    train_df, val_df = _load_train_val()
    X_train = train_df.drop(columns=[config.TARGET])
    y_train = train_df[config.TARGET]
    X_val = val_df.drop(columns=[config.TARGET])
    y_val = val_df[config.TARGET]

    mlflow.set_experiment(config.EXPERIMENT_TRAINING)
    run_name = f"{source}-{algo}-{_compact_param_str(params)}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("source", source)
        mlflow.set_tag("algo", algo)
        mlflow.log_params(params)

        model = _build_pipeline(algo, params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_prob),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
        }
        mlflow.log_metrics(metrics)

        # Log feature schema artifact
        schema = {c: str(X_train[c].dtype) for c in X_train.columns}
        mlflow.log_dict(schema, "feature_schema.json")

        # Log confusion matrix plot
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_val, y_pred, ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Log model — Pipeline is sklearn, so always sklearn flavor
        mlflow.sklearn.log_model(model, name="model", input_example=X_train.iloc[:5])

        return run.info.run_id
