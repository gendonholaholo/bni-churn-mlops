"""Optuna hyperparameter tuning for XGBoost.

Each Optuna trial = 1 MLflow run, logged via reuse of `train.train_one`.
Zero duplication with seed_runs and Gradio Training Lab.

Usage:
    uv run python -m scripts.tune                # 100 trials default
    uv run python -m scripts.tune --n-trials 50  # quick mode

After completion, all runs visible in MLflow UI:
    Filter: tags.source = 'optuna-tune'
    Sort by: metrics.f1 (descending)
"""

import argparse

import mlflow
import optuna

from mlops_churn import config, train


def _suggest_params(trial: optuna.Trial) -> dict:
    """Sample one set of XGBoost hyperparameters from the search space."""
    space = config.OPTUNA_SEARCH_SPACE
    params = {}
    for name, spec in space.items():
        if name in ("n_estimators", "max_depth", "min_child_weight"):
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        else:
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
    return params


def make_objective(algo: str = "xgboost"):
    """Build objective fn that calls train.train_one and returns F1.

    Each trial logs a full MLflow run (params + 5 metrics + model artifact)
    via the same `train.train_one` used by seed_runs and Gradio Lab.
    """
    client = mlflow.MlflowClient()

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        run_id = train.train_one(algo, params, source="optuna-tune")
        client.set_tag(run_id, "trial", str(trial.number))
        return mlflow.get_run(run_id).data.metrics["f1"]

    return objective


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for XGBoost (logs each trial to MLflow)"
    )
    parser.add_argument("--n-trials", type=int, default=config.OPTUNA_N_TRIALS)
    parser.add_argument("--algo", default="xgboost", choices=["xgboost"])
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_TRAINING)

    print(
        f"Optuna tuning: {args.n_trials} trials, algo={args.algo}, "
        f"search_space={len(config.OPTUNA_SEARCH_SPACE)} hyperparameters"
    )
    print(
        f"All runs logged to MLflow experiment '{config.EXPERIMENT_TRAINING}' "
        f"with tag source=optuna-tune."
    )
    print()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.OPTUNA_RANDOM_SEED),
        study_name=f"optuna-{args.algo}",
    )
    study.optimize(make_objective(args.algo), n_trials=args.n_trials)

    print()
    print("=" * 60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best F1:    {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k:20s} = {v}")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. Open MLflow UI: {config.MLFLOW_UI_URL}")
    print(f"  2. Experiment '{config.EXPERIMENT_TRAINING}' → filter `tags.source = 'optuna-tune'`")
    print("  3. Sort by F1, pick winner, then promote:")
    print("     uv run python -m scripts.promote --run-id <RUN_ID> --alias staging")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
