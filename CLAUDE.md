# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Hands-on MLOps **simulation/demo** of customer-churn classification. The audience is product/engineering stakeholders watching a ≤15-minute walkthrough — design choices favor demo legibility (one Gradio button, one MLflow tab) over production hardening. Treat the README's "Demo Flow" as the canonical end-to-end scenario.

Stack: Python 3.11 · `uv` · `ruff` · MLflow 3.1.4 · Gradio 6.0.1 · scikit-learn 1.7.1 · xgboost · scipy · optuna.

## Commands

```bash
uv sync                                            # install deps
uv run python -m scripts.seed_runs                 # preprocess + 9 baseline runs + register top 2
uv run python -m scripts.tune --n-trials 50        # Optuna XGBoost search (each trial = 1 MLflow run)
uv run python -m scripts.promote --list            # see registered versions + aliases
uv run python -m scripts.promote --run-id <ID> --alias staging
uv run python -m scripts.simulate_traffic --mode drifted    # generate monitoring run with KS drift
./run.sh                                           # MLflow server (:5001) + Gradio (:7860)

uv run pytest                                      # full suite (~30s, ~14 tests)
uv run pytest tests/test_train.py::test_train_one_returns_valid_run_id   # single test
uv run ruff check . && uv run ruff format .
```

The Kaggle dataset (`Churn_Modelling.csv`) must be placed at `data/raw/` before `seed_runs`. It is gitignored.

**Port 5001, not 5000** — the run script and `config.MLFLOW_UI_URL` deliberately use 5001 because macOS Control Center reserves 5000 (AirPlay Receiver). Don't "fix" this back to 5000.

## Architecture

The repo splits into a **library** (`src/mlops_churn/`), **callable scripts** (`scripts/`), a **single Gradio app** (`app/gradio_app.py`), and **tests**.

### Single source of truth: `config.py`

`src/mlops_churn/config.py` owns every path, MLflow name, alias string, hyperparameter spec, drift threshold, customer feature schema, and demo sample customer. **It has zero internal imports** so every other module can depend on it freely. When adding a new tunable knob, demo sample, or alias, put it here first — Gradio sliders, seed variants, the inference form, and tests all read from this module.

### Shared training entry: `train.train_one`

`train.train_one(algo, params, source)` is the **only** place a model gets trained and logged. Three callers share it:

- `scripts/seed_runs.py` (source=`"seed"`) — 9 fixed variants from `config.SEED_VARIANTS`
- `app/gradio_app.py` Training Lab (source=`"gradio-lab"`) — sliders pulled from `config.HYPERPARAM_SPEC`
- `scripts/tune.py` (source=`"optuna-tune"`) — Optuna trials over `config.OPTUNA_SEARCH_SPACE`

The `source` tag is how MLflow runs are filtered in the UI (`tags.source = 'optuna-tune'`). Don't introduce a parallel training path; extend `train_one` and route through it.

### Models are sklearn Pipelines (preprocessing inside the artifact)

`train._build_pipeline` wraps the chosen classifier with a `ColumnTransformer` (StandardScaler on numerics, OneHotEncoder on categoricals, pass-through on binaries). Consequences:

- `data.preprocess` only drops leak columns (`RowNumber`, `CustomerId`, `Surname`). It does **not** scale or encode.
- Inference (`serving.predict`) and traffic simulation pass **raw features** directly to the loaded model — no separate scaler artifact to manage. If you change the column lists in `config.NUMERIC_FEATURES` / `CATEGORICAL_FEATURES` / `BINARY_FEATURES`, `_build_pipeline` and the Gradio form both follow automatically.
- All three algorithms (`logistic_regression`, `random_forest`, `xgboost`) support `predict_proba`, so `serving.predict` always returns a probability.

### MLflow registry: aliases, not stages

This project uses the post-deprecation alias API (`@production`, `@staging`, `@archived`) — never the old `Stage="Production"` strings. `registry.set_alias` is atomic (moves an existing alias). Two experiments:

- `churn-prediction` — training runs (used by `seed_runs`, Gradio Lab, `tune.py`)
- `production-monitoring` — one run per `simulate_traffic` invocation, logging 5 metrics per step (`prediction_count`, `latency_p50_ms`, `latency_p95_ms`, `churn_rate`, `drift_score`) and tagging `alert=true` when `drift_score > config.DRIFT_SCORE_THRESHOLD` (0.2).

Drift uses `scipy.stats.ks_2samp` averaged across `NUMERIC_FEATURES`. The `--mode drifted` simulation in `scripts/simulate_traffic.py` shifts Age / CreditScore / Balance in raw units to push KS above the threshold.

### Inference cache (intentionally single-threaded)

`serving._cache` is a module-level dict reloaded only when alias→version mapping changes. The docstring explicitly notes "no lock by design" — do **not** add threading primitives unless the Gradio workload becomes concurrent (it isn't).

## Testing patterns

`tests/conftest.py` provides three fixtures used by every test:

- `tmp_mlflow_uri` — points MLflow at `tmp_path` so tests never touch `mlflow.db`
- `synthetic_data` — `make_classification` (200 samples, 10 numeric features `f0..f9`)
- `numeric_only_config` — monkeypatches `config.NUMERIC_FEATURES = ["f0".."f9"]` and `CATEGORICAL_FEATURES = []` so the `ColumnTransformer` works against synthetic data

Tests that exercise `train.train_one` use **all three** fixtures plus `monkeypatch.setattr(train, "_load_train_val", ...)` to inject a synthetic split. Follow that pattern when adding training-related tests; do not require the real Kaggle CSV.

## Ruff configuration to be aware of

`ignore = ["N803", "N806"]` — `X`, `y` uppercase identifiers are intentionally allowed (sklearn canonical). Don't rename them. `notebooks/` is excluded from lint/format. Line length is 100, double quotes.
