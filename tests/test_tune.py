"""Smoke test for scripts/tune.py — verify objective fn runs end-to-end."""

import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_churn import config, train
from scripts.tune import _suggest_params, make_objective


def test_suggest_params_covers_all_search_space_keys():
    """_suggest_params returns dict with all 6 keys from OPTUNA_SEARCH_SPACE."""
    study = optuna.create_study()
    trial = study.ask()
    params = _suggest_params(trial)
    assert set(params.keys()) == set(config.OPTUNA_SEARCH_SPACE.keys())
    # int params produce ints, float params produce floats
    assert isinstance(params["n_estimators"], int)
    assert isinstance(params["max_depth"], int)
    assert isinstance(params["learning_rate"], float)


def test_objective_runs_end_to_end_with_synthetic_data(
    tmp_mlflow_uri, monkeypatch, synthetic_data, numeric_only_config
):
    """make_objective produces a callable that runs 1 trial and returns F1 in [0, 1]."""
    X, y = synthetic_data
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df[config.TARGET] = y
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[config.TARGET]
    )
    monkeypatch.setattr(train, "_load_train_val", lambda: (train_df, val_df))

    objective = make_objective("xgboost")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)

    assert len(study.trials) == 2
    for t in study.trials:
        assert t.value is not None
        assert 0.0 <= t.value <= 1.0
