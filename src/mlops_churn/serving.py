"""Inference using MLflow registered model by alias.

Single-threaded demo cache; no lock by design. Don't refactor for thread-safety
unless Gradio workload becomes concurrent.
"""

import time
from typing import Any

import mlflow
import mlflow.sklearn
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


def predict(features: dict[str, Any], alias: str = "production") -> dict[str, Any]:
    """Run inference. Returns {prob, label, latency_ms, version}.

    The registered model is a sklearn Pipeline that includes preprocessing.
    Pass raw features directly — Pipeline handles scaling + one-hot internally.
    """
    model, version = _load_or_use_cache(alias)
    X = pd.DataFrame([features])

    t0 = time.perf_counter()
    raw = model.predict(X)
    latency_ms = (time.perf_counter() - t0) * 1000

    label = int(raw[0]) if hasattr(raw, "__getitem__") else int(raw)

    # Pipeline always supports predict_proba (all 3 algos do)
    try:
        sk_model = mlflow.sklearn.load_model(f"models:/{config.REGISTERED_MODEL_NAME}@{alias}")
        prob = float(sk_model.predict_proba(X)[0, 1])
    except Exception:
        prob = float(label)

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
