"""MLflow Model Registry alias management (post-stages-deprecation API)."""

from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from mlops_churn import config


def _client() -> MlflowClient:
    return MlflowClient()


def register_run(run_id: str) -> str:
    """Register the model artifact from a run. Returns version (string)."""
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, config.REGISTERED_MODEL_NAME)
    return mv.version


def set_alias(alias: str, version: str) -> None:
    """Assign or move alias to version. Atomic — if alias existed, it moves."""
    _client().set_registered_model_alias(config.REGISTERED_MODEL_NAME, alias, version)


def get_version_by_alias(alias: str) -> ModelVersion:
    """Get ModelVersion by alias. Raises if alias does not exist."""
    return _client().get_model_version_by_alias(config.REGISTERED_MODEL_NAME, alias)


def remove_alias(alias: str) -> None:
    """Delete alias. MLflow raises if not exists."""
    _client().delete_registered_model_alias(config.REGISTERED_MODEL_NAME, alias)


def list_versions() -> list[ModelVersion]:
    """List all versions of the registered model."""
    return _client().search_model_versions(f"name='{config.REGISTERED_MODEL_NAME}'")


def transition_history(version: str) -> list[dict[str, Any]]:
    """Return MLflow's audit history for a version (timestamps + alias changes)."""
    mv = _client().get_model_version(config.REGISTERED_MODEL_NAME, version)
    return [
        {
            "version": mv.version,
            "creation_ts": mv.creation_timestamp,
            "last_updated_ts": mv.last_updated_timestamp,
            "current_aliases": mv.aliases,
        }
    ]
