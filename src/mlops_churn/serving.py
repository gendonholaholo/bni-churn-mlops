"""Inference using MLflow registered model by alias.

Single-threaded demo cache; no lock by design. Don't refactor for thread-safety
unless Gradio workload becomes concurrent.
"""

import time
from typing import Any

import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow import MlflowClient

from mlops_churn import config

_cache: dict | None = None


def _resolve_version(alias: str) -> str:
    return MlflowClient().get_model_version_by_alias(config.REGISTERED_MODEL_NAME, alias).version


def _load_or_use_cache(alias: str):
    global _cache
    current_version = _resolve_version(alias)
    if _cache is None or _cache["alias"] != alias or _cache["version"] != current_version:
        _cache = {
            "model": mlflow.pyfunc.load_model(f"models:/{config.REGISTERED_MODEL_NAME}@{alias}"),
            "version": current_version,
            "alias": alias,
        }
    return _cache["model"], _cache["version"]


def _get_probability(features: dict, alias: str, label: int) -> float:
    """Load underlying model in its native flavor and call predict_proba.

    Tries sklearn first (works for LR + RF), then xgboost flavor.
    Falls back to label if neither provides predict_proba.
    """
    import pandas as pd

    X = pd.DataFrame([features])
    model_uri = f"models:/{config.REGISTERED_MODEL_NAME}@{alias}"

    # Try sklearn flavor first
    try:
        sk_model = mlflow.sklearn.load_model(model_uri)
        return float(sk_model.predict_proba(X)[0, 1])
    except Exception:
        pass

    # Try xgboost flavor
    try:
        xgb_model = mlflow.xgboost.load_model(model_uri)
        return float(xgb_model.predict_proba(X)[0, 1])
    except Exception:
        pass

    # Last resort: degenerate label-based prob
    return float(label)


def predict(features: dict[str, Any], alias: str = "production") -> dict[str, Any]:
    """Run inference. Returns {prob, label, latency_ms, version}."""
    model, version = _load_or_use_cache(alias)
    X = pd.DataFrame([features])

    t0 = time.perf_counter()
    raw = model.predict(X)
    latency_ms = (time.perf_counter() - t0) * 1000

    label = int(raw[0]) if hasattr(raw, "__getitem__") else int(raw)

    # Get probability — try flavor-specific loaders
    prob = _get_probability(features, alias, label)

    return {
        "prob": prob,
        "label": label,
        "latency_ms": latency_ms,
        "version": version,
    }


def predict_ab(features: dict[str, Any]) -> dict[str, Any]:
    """Compare production vs staging predictions for the same input."""
    prod = predict(features, alias=config.ALIAS_PRODUCTION)
    stag = predict(features, alias=config.ALIAS_STAGING)
    return {
        "production": prod,
        "staging": stag,
        "agreement": prod["label"] == stag["label"],
    }
