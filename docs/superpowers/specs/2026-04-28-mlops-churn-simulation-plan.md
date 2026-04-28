# MLOps Churn Simulation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hands-on MLOps simulation (BNI churn use case) using MLflow 3.1.4 + Gradio 6.0.1, demonstrating training tracking, model registry with aliases, inference, A/B test, and drift monitoring — for self-learning + PM demo.

**Architecture:** Modular Python package (`src/mlops_churn/`) with thin script wrappers and a single Gradio app (3 tabs). MLflow serves as observation/governance surface; Gradio serves as action surface. No tool overlap. Local execution only.

**Tech Stack:** Python 3.11 · uv (package mgr) · ruff (lint+format) · MLflow 3.1.4 · Gradio 6.0.1 · scikit-learn 1.7.1 · xgboost · scipy · pytest

**Reference docs (READ BEFORE STARTING):**
- Design: [`2026-04-28-mlops-churn-simulation-design.md`](2026-04-28-mlops-churn-simulation-design.md)
- SoW + DoD: [`2026-04-28-mlops-churn-simulation-sow-dod.md`](2026-04-28-mlops-churn-simulation-sow-dod.md)

**Working directory:** `/Users/ghawsshafadonia/Documents/Pekerjaan/BNI/test/`

**Critical principles (from spec, BINDING):**
- No deprecated MLflow stages — use aliases via `MlflowClient.set_registered_model_alias`
- No custom math when library exists — `scipy.stats.ks_2samp` for drift, `sklearn` for everything else
- Single Source of Truth: all constants in `config.py`
- KISS / YAGNI: no features beyond 20 in-scope items
- Use `uv run` for all Python commands; `uv add` for deps; commit `uv.lock`

---

## Phase 0 — Project Setup (4 tasks)

### Task 1: Initialize uv project + add dependencies

**Files:**
- Create: `test/pyproject.toml` (via uv)
- Create: `test/uv.lock` (auto)
- Create: `test/.venv/` (auto, gitignored)

- [ ] **Step 1: Initialize uv project**

```bash
cd /Users/ghawsshafadonia/Documents/Pekerjaan/BNI/test
uv init --no-readme --python 3.11 --package --name mlops-churn
```
Expected: creates `pyproject.toml` with `name = "mlops-churn"`, `.python-version`, and **`src/mlops_churn/__init__.py`** (uv auto-creates the src layout). After subsequent `uv add`, the package is editable-installed in `.venv` so `from mlops_churn import ...` works directly in tests.

- [ ] **Step 2: Add runtime dependencies**

```bash
uv add mlflow==3.1.4 gradio==6.0.1 scikit-learn==1.7.1 xgboost pandas numpy matplotlib scipy
```
Expected: `pyproject.toml` updated with `[project.dependencies]`, `uv.lock` written.

- [ ] **Step 3: Add dev dependencies**

```bash
uv add --dev pytest ruff
```
Expected: dev group in `pyproject.toml`.

- [ ] **Step 4: Verify environment + package importable**

```bash
uv run python -c "import mlflow, gradio, sklearn, xgboost, scipy, pandas; print('libs OK')"
uv run python -c "from mlops_churn import __init__; print('package OK')"
```
Expected: stdout `libs OK` and `package OK`. If second line fails with `ModuleNotFoundError`, run `uv sync` to ensure editable install completed.

- [ ] **Step 5: Commit**

```bash
git add test/pyproject.toml test/uv.lock test/.python-version
git commit -m "chore(setup): init uv project + add deps (mlflow 3.1.4, gradio 6.0.1, sklearn, xgboost, scipy)"
```

---

### Task 2: Create folder skeleton + .gitignore

**Files:** (note: `src/mlops_churn/__init__.py` already created by `uv init --package` in Task 1 — do not recreate)
- Create: `test/.gitignore`
- Create: `test/scripts/__init__.py`
- Create: `test/tests/__init__.py`
- Create: `test/app/`
- Create: `test/data/raw/.gitkeep`
- Create: `test/data/processed/.gitkeep`
- Create: `test/notebooks/`

- [ ] **Step 1: Make directories (skip src/mlops_churn — uv created it)**

```bash
cd /Users/ghawsshafadonia/Documents/Pekerjaan/BNI/test
mkdir -p scripts app tests data/raw data/processed notebooks
touch scripts/__init__.py tests/__init__.py
touch data/raw/.gitkeep data/processed/.gitkeep
```

- [ ] **Step 2: Write .gitignore**

```bash
cat > .gitignore <<'EOF'
# uv / venv
.venv/
__pycache__/
*.py[cod]

# MLflow runtime
mlruns/
mlartifacts/
mlflow.db
mlflow.db-journal

# Data (raw + processed are gitignored except .gitkeep)
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# IDE
.vscode/
.idea/
.DS_Store
EOF
```

- [ ] **Step 3: Verify uv.lock + .python-version still tracked**

```bash
git status
```
Expected: `uv.lock`, `.python-version`, `pyproject.toml` already tracked from Task 1; new files visible.

- [ ] **Step 4: Commit**

```bash
git add test/.gitignore test/src test/scripts test/tests test/app test/data test/notebooks
git commit -m "chore(setup): scaffold folder skeleton + .gitignore"
```

---

### Task 3: Configure ruff in pyproject.toml

**Files:**
- Modify: `test/pyproject.toml`

- [ ] **Step 1: Append ruff + pytest config to pyproject.toml**

```bash
cat >> pyproject.toml <<'EOF'

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N"]

[tool.ruff.format]
quote-style = "double"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
EOF
```

**Catatan:** `pythonpath` config tidak diperlukan karena `uv init --package` membuat package editable-installed di `.venv`, sehingga `from mlops_churn import ...` bekerja langsung di tests.

- [ ] **Step 2: Verify ruff runs cleanly on empty project**

```bash
uv run ruff check .
uv run ruff format --check .
```
Expected: `All checks passed!` and no format issues.

- [ ] **Step 3: Commit**

```bash
git add test/pyproject.toml
git commit -m "chore(setup): configure ruff (lint + format) in pyproject.toml"
```

---

### Task 4: Write README skeleton

**Files:**
- Create: `test/README.md`

- [ ] **Step 1: Write README with quickstart**

```markdown
# BNI Churn — MLOps Simulation

Hands-on MLOps simulation: customer churn classification using MLflow 3.x (tracking + registry + aliases) and Gradio 6.x (training trigger + inference + A/B test).

## Tech Stack
Python 3.11 · uv · ruff · MLflow 3.1.4 · Gradio 6.0.1 · scikit-learn 1.7.1 · xgboost · scipy

## Setup

1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
2. Sync dependencies:
   ```bash
   uv sync
   ```
3. Download the Churn Modelling dataset from Kaggle:
   https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
   Place `Churn_Modelling.csv` at `data/raw/Churn_Modelling.csv`.

## Quickstart Demo

```bash
# 1. Initial seed (preprocess + 9 baseline runs + register top 2)
uv run python -m scripts.seed_runs

# 2. Launch MLflow + Gradio
./run.sh

# Open in browser:
#  - Gradio app: http://localhost:7860
#  - MLflow UI: http://localhost:5000
```

## Demo Flow (≤ 15 min)
See [docs/superpowers/specs/2026-04-28-mlops-churn-simulation-design.md](docs/superpowers/specs/2026-04-28-mlops-churn-simulation-design.md) Section 7.

## Development

```bash
uv run pytest           # 14 tests, < 30s
uv run ruff check .     # lint
uv run ruff format .    # format
```
```

- [ ] **Step 2: Commit**

```bash
git add test/README.md
git commit -m "docs(setup): add README skeleton with quickstart"
```

---

## Phase 1 — Core SSoT: config.py (1 task)

### Task 5: Implement config.py (no test — pure constants)

**Files:**
- Create: `test/src/mlops_churn/config.py`

- [ ] **Step 1: Write config.py with all SSoT constants**

```python
"""Single source of truth for paths, MLflow naming, hyperparameters, schemas, thresholds.

This module has NO internal imports — anyone can import from here without circular deps.
"""
from pathlib import Path

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # test/
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Churn_Modelling.csv"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---- MLflow ----
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_BACKEND = "sqlite:///mlflow.db"
MLFLOW_ARTIFACT_ROOT = "./mlartifacts"

EXPERIMENT_TRAINING = "churn-prediction"
EXPERIMENT_MONITORING = "production-monitoring"
REGISTERED_MODEL_NAME = "churn-model"

ALIAS_PRODUCTION = "production"
ALIAS_STAGING = "staging"
ALIAS_ARCHIVED = "archived"

# ---- Drift ----
DRIFT_SCORE_THRESHOLD = 0.2  # KS D-statistic; >0.2 = noticeable drift

# ---- Hyperparameter spec per algorithm (used by seed, Gradio Lab, tests) ----
HYPERPARAM_SPEC = {
    "logistic_regression": {
        "C":        {"min": 0.01, "max": 10.0,  "default": 1.0},
        "max_iter": {"min": 100,  "max": 2000,  "default": 500},
    },
    "random_forest": {
        "n_estimators":      {"min": 10, "max": 500, "default": 100},
        "max_depth":         {"min": 3,  "max": 30,  "default": 10},
        "min_samples_split": {"min": 2,  "max": 20,  "default": 2},
    },
    "xgboost": {
        "n_estimators":  {"min": 10,   "max": 500, "default": 100},
        "learning_rate": {"min": 0.01, "max": 0.5, "default": 0.1},
        "max_depth":     {"min": 3,    "max": 15,  "default": 6},
    },
}

# ---- 3 hyperparameter variants per algo for seed_runs ----
SEED_VARIANTS = {
    "logistic_regression": [
        {"C": 0.1, "max_iter": 500},
        {"C": 1.0, "max_iter": 500},
        {"C": 10.0, "max_iter": 2000},
    ],
    "random_forest": [
        {"n_estimators": 50, "max_depth": 10, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5},
        {"n_estimators": 500, "max_depth": 30, "min_samples_split": 10},
    ],
    "xgboost": [
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 8},
        {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 10},
    ],
}

# ---- Customer feature schema (for Gradio inference form, SSoT) ----
CUSTOMER_FEATURE_SCHEMA = {
    "CreditScore":     {"type": "int",   "min": 350,    "max": 850,    "default": 650},
    "Geography":       {"type": "categorical", "choices": ["France", "Spain", "Germany"], "default": "France"},
    "Gender":          {"type": "categorical", "choices": ["Male", "Female"], "default": "Male"},
    "Age":             {"type": "int",   "min": 18,     "max": 95,     "default": 40},
    "Tenure":          {"type": "int",   "min": 0,      "max": 10,     "default": 5},
    "Balance":         {"type": "float", "min": 0,      "max": 300000, "default": 50000},
    "NumOfProducts":   {"type": "int",   "min": 1,      "max": 4,      "default": 1},
    "HasCrCard":       {"type": "binary","default": 1},  # 1=Yes, 0=No
    "IsActiveMember":  {"type": "binary","default": 1},
    "EstimatedSalary": {"type": "float", "min": 10000,  "max": 200000, "default": 80000},
}

NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "EstimatedSalary",
]
CATEGORICAL_FEATURES = ["Geography", "Gender"]
BINARY_FEATURES = ["HasCrCard", "IsActiveMember"]
TARGET = "Exited"

# ---- A/B Test sample customers (3 presets) ----
AB_TEST_SAMPLES = {
    "Young high-balance": {
        "CreditScore": 720, "Geography": "France", "Gender": "Female", "Age": 28,
        "Tenure": 3, "Balance": 150000, "NumOfProducts": 2,
        "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 95000,
    },
    "Senior low-balance": {
        "CreditScore": 580, "Geography": "Germany", "Gender": "Male", "Age": 62,
        "Tenure": 8, "Balance": 5000, "NumOfProducts": 1,
        "HasCrCard": 0, "IsActiveMember": 0, "EstimatedSalary": 45000,
    },
    "Mid-age active": {
        "CreditScore": 680, "Geography": "Spain", "Gender": "Male", "Age": 42,
        "Tenure": 5, "Balance": 75000, "NumOfProducts": 2,
        "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 110000,
    },
}
```

- [ ] **Step 2: Lint + format**

```bash
uv run ruff check src/mlops_churn/config.py
uv run ruff format src/mlops_churn/config.py
```
Expected: clean.

- [ ] **Step 3: Verify importable**

```bash
uv run python -c "from mlops_churn import config; print(config.REGISTERED_MODEL_NAME, config.DRIFT_SCORE_THRESHOLD)"
```
Expected: `churn-model 0.2`

- [ ] **Step 4: Commit**

```bash
git add test/src/mlops_churn/config.py
git commit -m "feat(config): add SSoT constants module (paths, MLflow naming, hyperparam spec, schemas)"
```

---

## Phase 2 — Conftest + Fixtures (1 task)

### Task 6: Create tests/conftest.py with shared fixtures

**Files:**
- Create: `test/tests/conftest.py`

- [ ] **Step 1: Write conftest.py**

```python
"""Shared pytest fixtures: isolated MLflow tracking + synthetic dataset."""
import mlflow
import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def tmp_mlflow_uri(tmp_path, monkeypatch):
    """Isolate MLflow tracking per test using tmp_path."""
    uri = f"file:{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test-experiment")
    return uri


@pytest.fixture
def synthetic_data():
    """Small deterministic dataset for unit tests (sklearn built-in)."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=6,
        random_state=42,
    )
    return X, y


@pytest.fixture
def synthetic_dataframe():
    """Synthetic dataset matching Churn schema (for data.py tests)."""
    return pd.DataFrame({
        "CreditScore": [650, 720, 580, 680] * 25,
        "Geography": ["France", "Spain", "Germany", "France"] * 25,
        "Gender": ["Male", "Female"] * 50,
        "Age": [40, 28, 62, 42] * 25,
        "Tenure": [5, 3, 8, 5] * 25,
        "Balance": [50000.0, 150000.0, 5000.0, 75000.0] * 25,
        "NumOfProducts": [1, 2, 1, 2] * 25,
        "HasCrCard": [1, 1, 0, 1] * 25,
        "IsActiveMember": [1, 1, 0, 1] * 25,
        "EstimatedSalary": [80000.0, 95000.0, 45000.0, 110000.0] * 25,
        "Exited": [0, 0, 1, 0] * 25,
    })
```

- [ ] **Step 2: Verify pytest discovers fixtures**

```bash
uv run pytest tests/ --collect-only
```
Expected: 0 tests collected (no test files yet) but no errors.

- [ ] **Step 3: Lint**

```bash
uv run ruff check tests/conftest.py
uv run ruff format tests/conftest.py
```

- [ ] **Step 4: Commit**

```bash
git add test/tests/conftest.py
git commit -m "test(conftest): add shared fixtures (tmp_mlflow_uri, synthetic_data, synthetic_dataframe)"
```

---

## Phase 3 — data.py (3 tasks, TDD)

### Task 7: TDD — `data.load_raw()`

**Files:**
- Create: `test/tests/test_data.py`
- Create: `test/src/mlops_churn/data.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_data.py
import pandas as pd
import pytest
from mlops_churn import data, config


def test_load_raw_returns_dataframe(tmp_path, monkeypatch, synthetic_dataframe):
    """load_raw reads CSV at config.DATA_RAW_PATH and returns DataFrame."""
    csv_path = tmp_path / "Churn_Modelling.csv"
    synthetic_dataframe.to_csv(csv_path, index=False)
    monkeypatch.setattr(config, "DATA_RAW_PATH", csv_path)

    df = data.load_raw()

    assert isinstance(df, pd.DataFrame)
    assert "Exited" in df.columns
    assert len(df) == 100
```

- [ ] **Step 2: Run test (expect fail)**

```bash
uv run pytest tests/test_data.py::test_load_raw_returns_dataframe -v
```
Expected: FAIL — `module 'mlops_churn' has no attribute 'data'`.

- [ ] **Step 3: Implement load_raw**

```python
# src/mlops_churn/data.py
"""Data ingestion + preprocessing + train/val/test split."""
import pandas as pd

from mlops_churn import config


def load_raw() -> pd.DataFrame:
    """Load Churn Modelling CSV from config.DATA_RAW_PATH."""
    if not config.DATA_RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.DATA_RAW_PATH}. "
            "Download from https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling "
            "and place at data/raw/Churn_Modelling.csv"
        )
    return pd.read_csv(config.DATA_RAW_PATH)
```

- [ ] **Step 4: Run test (expect pass)**

```bash
uv run pytest tests/test_data.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add test/tests/test_data.py test/src/mlops_churn/data.py
git commit -m "feat(data): add load_raw() with FileNotFoundError + tests"
```

---

### Task 8: TDD — `data.preprocess()`

**Files:**
- Modify: `test/tests/test_data.py`
- Modify: `test/src/mlops_churn/data.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_data.py
def test_preprocess_output_schema(synthetic_dataframe):
    """preprocess encodes categoricals + scales numerics, drops leak columns."""
    out = data.preprocess(synthetic_dataframe)

    # No identifier columns leaking into features
    assert "RowNumber" not in out.columns
    assert "CustomerId" not in out.columns
    assert "Surname" not in out.columns

    # Categoricals encoded (one-hot expands Geography + Gender)
    assert "Geography_Germany" in out.columns or "Geography_France" in out.columns
    assert "Gender_Male" in out.columns or "Gender_Female" in out.columns

    # Numerics still present
    assert "CreditScore" in out.columns

    # Target preserved
    assert "Exited" in out.columns

    # No NaN
    assert out.isna().sum().sum() == 0
```

- [ ] **Step 2: Run test (expect fail)**

```bash
uv run pytest tests/test_data.py::test_preprocess_output_schema -v
```
Expected: FAIL — `data.preprocess` not defined.

- [ ] **Step 3: Implement preprocess**

```python
# Append to src/mlops_churn/data.py
from sklearn.preprocessing import StandardScaler


# Columns to drop (identifiers/leak)
_LEAK_COLUMNS = ["RowNumber", "CustomerId", "Surname"]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals + scale numerics. Returns model-ready DataFrame.

    - Drops identifier columns (RowNumber, CustomerId, Surname) if present
    - One-hot encodes categorical features
    - Standard-scales numeric features
    - Preserves Exited target
    """
    df = df.copy()

    # Drop leak columns if present
    drop = [c for c in _LEAK_COLUMNS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)

    # One-hot encode categoricals
    df = pd.get_dummies(
        df,
        columns=config.CATEGORICAL_FEATURES,
        drop_first=False,
        dtype=int,
    )

    # Standard-scale numerics (binary cols stay 0/1)
    scaler = StandardScaler()
    df[config.NUMERIC_FEATURES] = scaler.fit_transform(df[config.NUMERIC_FEATURES])

    return df
```

- [ ] **Step 4: Run tests (both pass)**

```bash
uv run pytest tests/test_data.py -v
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add test/tests/test_data.py test/src/mlops_churn/data.py
git commit -m "feat(data): add preprocess() with one-hot + scaling + leak-column drop"
```

---

### Task 9: TDD — `data.get_splits()`

**Files:**
- Modify: `test/tests/test_data.py`
- Modify: `test/src/mlops_churn/data.py`

- [ ] **Step 1: Write failing test**

```python
def test_get_splits_ratios(tmp_path, monkeypatch, synthetic_dataframe):
    """get_splits writes train/val/test in approximately 70/15/15 ratio."""
    monkeypatch.setattr(config, "DATA_PROCESSED_DIR", tmp_path)

    # Stage processed file as if preprocess already ran
    processed = data.preprocess(synthetic_dataframe)
    train, val, test = data.get_splits(processed)

    total = len(train) + len(val) + len(test)
    assert total == len(synthetic_dataframe)
    assert 0.65 < len(train) / total < 0.75
    assert 0.10 < len(val) / total < 0.20
    assert 0.10 < len(test) / total < 0.20

    # All splits have target column
    assert "Exited" in train.columns
    assert "Exited" in val.columns
    assert "Exited" in test.columns
```

- [ ] **Step 2: Run test (expect fail)**

```bash
uv run pytest tests/test_data.py::test_get_splits_ratios -v
```
Expected: FAIL — `data.get_splits` not defined.

- [ ] **Step 3: Implement get_splits**

```python
# Append to src/mlops_churn/data.py
from sklearn.model_selection import train_test_split


def get_splits(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split preprocessed DataFrame into train/val/test (default 70/15/15)."""
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[config.TARGET],
    )
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val[config.TARGET],
    )
    return train, val, test


def write_splits_to_disk(train, val, test) -> None:
    """Write train/val/test CSVs to config.DATA_PROCESSED_DIR."""
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.DATA_PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(config.DATA_PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(config.DATA_PROCESSED_DIR / "test.csv", index=False)


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read pre-split CSVs from disk."""
    d = config.DATA_PROCESSED_DIR
    return (
        pd.read_csv(d / "train.csv"),
        pd.read_csv(d / "val.csv"),
        pd.read_csv(d / "test.csv"),
    )
```

- [ ] **Step 4: Run all data tests (3 pass)**

```bash
uv run pytest tests/test_data.py -v
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add test/tests/test_data.py test/src/mlops_churn/data.py
git commit -m "feat(data): add get_splits + write_splits_to_disk + load_processed (70/15/15 stratified)"
```

---

## Phase 4 — train.py (1 task, 2 tests)

### Task 10: TDD — `train.train_one()`

**Files:**
- Create: `test/tests/test_train.py`
- Create: `test/src/mlops_churn/train.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_train.py
import mlflow
from mlops_churn import train, config


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

    run_id = train.train_one("random_forest", {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2})

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
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[config.TARGET])
    return train_df, val_df
```

- [ ] **Step 2: Run tests (expect fail)**

```bash
uv run pytest tests/test_train.py -v
```
Expected: FAIL — `train.train_one` not defined.

- [ ] **Step 3: Implement train.py**

```python
# src/mlops_churn/train.py
"""Train one model with given algo + params, log to MLflow."""
import json
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
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
from xgboost import XGBClassifier

from mlops_churn import config, data


_MODEL_CLASS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}

_LOG_MODEL_FN = {
    "logistic_regression": mlflow.sklearn.log_model,
    "random_forest": mlflow.sklearn.log_model,
    "xgboost": mlflow.xgboost.log_model,
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

        model = _MODEL_CLASS[algo](**params, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy":  accuracy_score(y_val, y_pred),
            "f1":        f1_score(y_val, y_pred),
            "roc_auc":   roc_auc_score(y_val, y_prob),
            "precision": precision_score(y_val, y_pred),
            "recall":    recall_score(y_val, y_pred),
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

        # Log model with appropriate flavor
        _LOG_MODEL_FN[algo](model, name="model", input_example=X_train.iloc[:5])

        return run.info.run_id
```

- [ ] **Step 4: Run tests (2 pass)**

```bash
uv run pytest tests/test_train.py -v
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add test/tests/test_train.py test/src/mlops_churn/train.py
git commit -m "feat(train): add train_one() with 3 algo dispatch, 5 metrics, model+schema+confmat artifacts"
```

---

## Phase 5 — registry.py (1 task, 3 tests)

### Task 11: TDD — `registry` alias management

**Files:**
- Create: `test/tests/test_registry.py`
- Create: `test/src/mlops_churn/registry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_registry.py
import mlflow
import pytest
from mlops_churn import registry, config


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
    with pytest.raises(Exception):
        registry.get_version_by_alias(config.ALIAS_STAGING)
```

- [ ] **Step 2: Run tests (expect fail)**

```bash
uv run pytest tests/test_registry.py -v
```
Expected: FAIL — `registry.register_run` not defined.

- [ ] **Step 3: Implement registry.py**

```python
# src/mlops_churn/registry.py
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
    _client().set_registered_model_alias(
        config.REGISTERED_MODEL_NAME, alias, version
    )


def get_version_by_alias(alias: str) -> ModelVersion:
    """Get ModelVersion by alias. Raises if alias does not exist."""
    return _client().get_model_version_by_alias(config.REGISTERED_MODEL_NAME, alias)


def remove_alias(alias: str) -> None:
    """Delete alias. Idempotent? — MLflow raises if not exists; caller wraps if needed."""
    _client().delete_registered_model_alias(config.REGISTERED_MODEL_NAME, alias)


def list_versions() -> list[ModelVersion]:
    """List all versions of the registered model."""
    return _client().search_model_versions(f"name='{config.REGISTERED_MODEL_NAME}'")


def transition_history(version: str) -> list[dict[str, Any]]:
    """Return MLflow's audit history for a version (timestamps + alias changes)."""
    mv = _client().get_model_version(config.REGISTERED_MODEL_NAME, version)
    # MLflow ModelVersion has creation_timestamp + last_updated_timestamp + aliases
    return [{
        "version": mv.version,
        "creation_ts": mv.creation_timestamp,
        "last_updated_ts": mv.last_updated_timestamp,
        "current_aliases": mv.aliases,
    }]
```

- [ ] **Step 4: Run tests (3 pass)**

```bash
uv run pytest tests/test_registry.py -v
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add test/tests/test_registry.py test/src/mlops_churn/registry.py
git commit -m "feat(registry): add alias mgmt (register_run/set_alias/get_version_by_alias/remove_alias) using MLflow 3.x API"
```

---

## Phase 6 — serving.py (1 task, 2 tests)

### Task 12: TDD — `serving.predict()` + `predict_ab()`

**Files:**
- Create: `test/tests/test_serving.py`
- Create: `test/src/mlops_churn/serving.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_serving.py
import mlflow
import pytest
from sklearn.linear_model import LogisticRegression
from mlops_churn import serving, registry, config


@pytest.fixture
def production_model(tmp_mlflow_uri, synthetic_data):
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


def test_predict_ab_returns_both_versions(production_model, tmp_mlflow_uri, synthetic_data):
    """predict_ab returns dict with production, staging, agreement keys."""
    # Set staging to a separate version
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
```

- [ ] **Step 2: Run tests (expect fail)**

```bash
uv run pytest tests/test_serving.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement serving.py**

```python
# src/mlops_churn/serving.py
"""Inference using MLflow registered model by alias.

Single-threaded demo cache; no lock by design. Don't refactor for thread-safety
unless Gradio workload becomes concurrent.
"""
import time
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient

from mlops_churn import config


_cache: dict | None = None


def _resolve_version(alias: str) -> str:
    return MlflowClient().get_model_version_by_alias(
        config.REGISTERED_MODEL_NAME, alias
    ).version


def _load_or_use_cache(alias: str):
    global _cache
    current_version = _resolve_version(alias)
    if (
        _cache is None
        or _cache["alias"] != alias
        or _cache["version"] != current_version
    ):
        _cache = {
            "model": mlflow.pyfunc.load_model(
                f"models:/{config.REGISTERED_MODEL_NAME}@{alias}"
            ),
            "version": current_version,
            "alias": alias,
        }
    return _cache["model"], _cache["version"]


def predict(features: dict[str, Any], alias: str = "production") -> dict[str, Any]:
    """Run inference. Returns {prob, label, latency_ms, version}."""
    model, version = _load_or_use_cache(alias)
    # Convert dict to single-row DataFrame
    X = pd.DataFrame([features])

    t0 = time.perf_counter()
    raw = model.predict(X)
    latency_ms = (time.perf_counter() - t0) * 1000

    # raw is array-like; for binary classification we expect prob (if predict_proba was used)
    # MLflow pyfunc wraps sklearn classifiers and returns predict() output (label)
    label = int(raw[0]) if hasattr(raw, "__getitem__") else int(raw)

    # Try to get probability if model supports it
    try:
        underlying = model.unwrap_python_model() if hasattr(model, "unwrap_python_model") else None
        if underlying and hasattr(underlying, "predict_proba"):
            prob = float(underlying.predict_proba(X)[0, 1])
        else:
            # Fallback: use sklearn loader directly
            sk_model = mlflow.sklearn.load_model(
                f"models:/{config.REGISTERED_MODEL_NAME}@{alias}"
            )
            prob = float(sk_model.predict_proba(X)[0, 1])
    except Exception:
        prob = float(label)  # degenerate fallback

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
```

- [ ] **Step 4: Run tests (2 pass)**

```bash
uv run pytest tests/test_serving.py -v
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add test/tests/test_serving.py test/src/mlops_churn/serving.py
git commit -m "feat(serving): add predict + predict_ab with version-cache invalidation by alias"
```

---

## Phase 7 — monitoring.py (1 task, 4 tests)

### Task 13: TDD — `monitoring` drift + batch metrics + alert

**Files:**
- Create: `test/tests/test_monitoring.py`
- Create: `test/src/mlops_churn/monitoring.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_monitoring.py
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlops_churn import monitoring, config


@pytest.fixture
def reference_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "CreditScore": rng.normal(650, 80, 500),
        "Age": rng.normal(40, 10, 500),
        "Balance": rng.normal(50000, 20000, 500),
    })


def test_drift_score_zero_for_identical(reference_df):
    """Identical distributions → drift score near 0."""
    score = monitoring.compute_drift(reference_df, reference_df.copy(), reference_df.columns.tolist())
    assert score < 0.05


def test_drift_score_high_for_shifted(reference_df):
    """Significantly shifted distributions → drift score > 0.5."""
    shifted = reference_df.copy()
    shifted["CreditScore"] += 200  # 2.5 sigma shift
    shifted["Age"] += 25
    shifted["Balance"] += 50000
    score = monitoring.compute_drift(reference_df, shifted, reference_df.columns.tolist())
    assert score > 0.5


def test_log_batch_metrics_writes_5_metrics(tmp_mlflow_uri):
    """log_batch_metrics writes prediction_count, latency_p50_ms, latency_p95_ms, churn_rate, drift_score."""
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        monitoring.log_batch_metrics(
            run_id=run_id,
            step=0,
            prediction_count=50,
            latency_ms_list=[40.0, 45.0, 50.0, 55.0, 60.0] * 10,
            labels=[0, 1, 0, 0, 1] * 10,
            drift_score=0.05,
        )

    fetched = mlflow.get_run(run_id)
    metrics = fetched.data.metrics
    for m in ["prediction_count", "latency_p50_ms", "latency_p95_ms", "churn_rate", "drift_score"]:
        assert m in metrics, f"missing metric {m}"


def test_log_batch_metrics_sets_alert_tag_when_drift_exceeds_threshold(tmp_mlflow_uri):
    """When drift_score > config.DRIFT_SCORE_THRESHOLD, run gets tag alert=true."""
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        monitoring.log_batch_metrics(
            run_id=run_id, step=0, prediction_count=50,
            latency_ms_list=[40.0] * 50, labels=[0] * 50,
            drift_score=config.DRIFT_SCORE_THRESHOLD + 0.05,
        )
    fetched = mlflow.get_run(run_id)
    assert fetched.data.tags.get("alert") == "true"
```

- [ ] **Step 2: Run tests (expect fail)**

```bash
uv run pytest tests/test_monitoring.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement monitoring.py**

```python
# src/mlops_churn/monitoring.py
"""Drift detection (KS test) + batch metric logging to MLflow."""
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from scipy.stats import ks_2samp

from mlops_churn import config


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_cols: list[str],
) -> float:
    """Mean Kolmogorov-Smirnov statistic across numeric columns. 0=identical, ~1=very different.

    Uses scipy.stats.ks_2samp (existing standard test).
    """
    if not numeric_cols:
        return 0.0
    scores = [
        ks_2samp(reference[col], current[col]).statistic for col in numeric_cols
    ]
    return float(sum(scores) / len(scores))


def log_batch_metrics(
    run_id: str,
    step: int,
    prediction_count: int,
    latency_ms_list: list[float],
    labels: list[int],
    drift_score: float,
) -> None:
    """Log 5 monitoring metrics for one batch + set alert tag if drift exceeds threshold."""
    client = MlflowClient()

    p50 = float(np.percentile(latency_ms_list, 50))
    p95 = float(np.percentile(latency_ms_list, 95))
    churn_rate = float(sum(labels) / len(labels)) if labels else 0.0

    metrics = {
        "prediction_count": float(prediction_count),
        "latency_p50_ms":   p50,
        "latency_p95_ms":   p95,
        "churn_rate":       churn_rate,
        "drift_score":      drift_score,
    }
    for name, value in metrics.items():
        client.log_metric(run_id, name, value, step=step)

    if drift_score > config.DRIFT_SCORE_THRESHOLD:
        client.set_tag(run_id, "alert", "true")
```

- [ ] **Step 4: Run tests (4 pass)**

```bash
uv run pytest tests/test_monitoring.py -v
```
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add test/tests/test_monitoring.py test/src/mlops_churn/monitoring.py
git commit -m "feat(monitoring): drift via scipy.stats.ks_2samp + log_batch_metrics with alert tag"
```

---

## Phase 8 — Scripts (3 tasks)

### Task 14: `scripts/seed_runs.py`

**Files:**
- Create: `test/scripts/seed_runs.py`

- [ ] **Step 1: Implement seed_runs.py**

```python
# scripts/seed_runs.py
"""Initial setup: preprocess + 9 baseline runs + register top 2 with aliases."""
import argparse

import mlflow

from mlops_churn import config, data, registry, train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-register", action="store_true")
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    # Preprocess
    if not args.skip_preprocess:
        print("[Preprocess] Loading", config.DATA_RAW_PATH)
        raw = data.load_raw()
        print(f"[Preprocess] {len(raw)} rows. Encoding + scaling...")
        processed = data.preprocess(raw)
        train_df, val_df, test_df = data.get_splits(processed)
        data.write_splits_to_disk(train_df, val_df, test_df)
        print(f"[Preprocess] Wrote splits to {config.DATA_PROCESSED_DIR}.")

    # Train 9 runs
    run_ids = []
    total = sum(len(v) for v in config.SEED_VARIANTS.values())
    i = 0
    for algo, variants in config.SEED_VARIANTS.items():
        for params in variants:
            i += 1
            print(f"[{i}/{total}] {algo}  {params}", end="  ")
            run_id = train.train_one(algo, params, source="seed")
            metrics = mlflow.get_run(run_id).data.metrics
            print(f"F1={metrics['f1']:.3f}  run={run_id[:6]}")
            run_ids.append((run_id, metrics["f1"], algo))

    # Register top 2 by F1
    if not args.skip_register:
        run_ids.sort(key=lambda x: x[1], reverse=True)
        top = run_ids[:2]
        v1 = registry.register_run(top[0][0])
        registry.set_alias(config.ALIAS_PRODUCTION, v1)
        v2 = registry.register_run(top[1][0])
        registry.set_alias(config.ALIAS_STAGING, v2)
        print(f"\n✅ Top 2 registered:")
        print(f"   v{v1} (run {top[0][0][:6]}, {top[0][2]}) → @production  F1={top[0][1]:.3f}")
        print(f"   v{v2} (run {top[1][0][:6]}, {top[1][2]}) → @staging     F1={top[1][1]:.3f}")

    print("\n🎯 Next: ./run.sh untuk buka Gradio + MLflow UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke test (will fail without data; tolerate that)**

```bash
uv run python -m scripts.seed_runs --skip-preprocess --skip-register 2>&1 | head -5
```
Expected: error about loading processed data — acceptable (no data yet). The script structure is what we test.

- [ ] **Step 3: Lint + format**

```bash
uv run ruff check scripts/seed_runs.py
uv run ruff format scripts/seed_runs.py
```

- [ ] **Step 4: Commit**

```bash
git add test/scripts/seed_runs.py
git commit -m "feat(scripts): add seed_runs.py — preprocess + 9 baseline runs + register top 2"
```

---

### Task 15: `scripts/promote.py`

**Files:**
- Create: `test/scripts/promote.py`

- [ ] **Step 1: Implement promote.py**

```python
# scripts/promote.py
"""Alias management CLI: register, move, list, remove."""
import argparse
import mlflow

from mlops_churn import config, registry


def cmd_list() -> int:
    versions = registry.list_versions()
    print(f"{config.REGISTERED_MODEL_NAME}:")
    if not versions:
        print("  (no versions registered)")
        return 0
    for mv in versions:
        aliases_str = ", ".join(mv.aliases) if mv.aliases else "(none)"
        print(f"  v{mv.version} (run {mv.run_id[:6]}) → aliases: {aliases_str}")
    return 0


def cmd_register_and_set(run_id: str, alias: str) -> int:
    print(f"Registering run {run_id} → {config.REGISTERED_MODEL_NAME}...")
    v = registry.register_run(run_id)
    print(f"✅ Registered as v{v}.")
    print(f"Setting alias '{alias}' → v{v}.")
    registry.set_alias(alias, v)
    print("✅ Done.")
    return 0


def cmd_move(version: str, alias: str) -> int:
    print(f"Setting alias '{alias}' → v{version}.")
    registry.set_alias(alias, version)
    print("✅ Done.")
    return 0


def cmd_remove(alias: str) -> int:
    print(f"Removing alias '{alias}'...")
    registry.remove_alias(alias)
    print("✅ Done.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Alias management for churn-model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List versions + aliases")
    group.add_argument("--run-id", help="Register run + set alias")
    group.add_argument("--version", help="Move existing alias to version")
    group.add_argument("--remove", action="store_true", help="Delete alias")
    parser.add_argument("--alias", help="Alias name (required for --run-id, --version, --remove)")
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    if args.list:
        return cmd_list()
    if args.run_id:
        if not args.alias:
            parser.error("--run-id requires --alias")
        return cmd_register_and_set(args.run_id, args.alias)
    if args.version:
        if not args.alias:
            parser.error("--version requires --alias")
        return cmd_move(args.version, args.alias)
    if args.remove:
        if not args.alias:
            parser.error("--remove requires --alias")
        return cmd_remove(args.alias)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Lint + format**

```bash
uv run ruff check scripts/promote.py
uv run ruff format scripts/promote.py
```

- [ ] **Step 3: Verify --help works**

```bash
uv run python -m scripts.promote --help
```
Expected: help text with 4 mutually-exclusive options.

- [ ] **Step 4: Commit**

```bash
git add test/scripts/promote.py
git commit -m "feat(scripts): add promote.py CLI (list/register/move/remove modes)"
```

---

### Task 16: `scripts/simulate_traffic.py`

**Files:**
- Create: `test/scripts/simulate_traffic.py`

- [ ] **Step 1: Implement simulate_traffic.py**

```python
# scripts/simulate_traffic.py
"""Generate inference batches + log monitoring metrics. 1 run per invocation."""
import argparse
import datetime as dt
import time

import mlflow
import numpy as np
import pandas as pd

from mlops_churn import config, data, monitoring, registry, serving


def _shift_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply distribution shift to numeric features (drift simulation).

    NOTE: data/processed/ contains z-scored numerics, so values are in stdev units
    (~ -3 to +3 typical). Shift +1.5 stdev = noticeable drift (KS detects).
    """
    out = df.copy()
    for col in ("Age", "CreditScore", "Balance"):
        if col in out.columns:
            out[col] = out[col] + 1.5  # +1.5 sigma shift in scaled space
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["normal", "drifted"], default="normal")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--alias", default=config.ALIAS_PRODUCTION)
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_MONITORING)

    # Load reference (training set) and pool (test set)
    train_df, _val, test_df = data.load_processed()
    target_col = config.TARGET
    feature_cols = [c for c in train_df.columns if c != target_col]
    numeric_cols = [c for c in config.NUMERIC_FEATURES if c in feature_cols]

    # Resolve current production version for tagging
    version = registry.get_version_by_alias(args.alias).version

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"monitoring-{args.mode}-{timestamp}"

    print(f"🌊 Traffic sim: mode={args.mode}, batches={args.batches}, batch_size={args.batch_size}")
    print(f"   Run name: {run_name}")
    print(f"   Serving model: {config.REGISTERED_MODEL_NAME} v{version} (@{args.alias})")
    print()
    print("step  count  p50ms  p95ms  churn_rate  drift_score  alert")

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.set_tag("mode", args.mode)
        mlflow.set_tag("model_version", str(version))
        mlflow.set_tag("alias_used", args.alias)

        rng = np.random.default_rng(42)
        for step in range(args.batches):
            sampled = test_df.sample(n=args.batch_size, replace=True, random_state=rng.integers(0, 10000))
            X_batch = sampled.drop(columns=[target_col])
            if args.mode == "drifted":
                X_batch = _shift_features(X_batch)

            latencies = []
            labels = []
            for _, row in X_batch.iterrows():
                features = row.to_dict()
                out = serving.predict(features, alias=args.alias)
                latencies.append(out["latency_ms"])  # use predict's measurement
                labels.append(out["label"])

            drift_score = monitoring.compute_drift(train_df[numeric_cols], X_batch[numeric_cols], numeric_cols)
            monitoring.log_batch_metrics(
                run_id=run_id,
                step=step,
                prediction_count=len(labels),
                latency_ms_list=latencies,
                labels=labels,
                drift_score=drift_score,
            )

            alert = "🚨" if drift_score > config.DRIFT_SCORE_THRESHOLD else "  "
            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))
            churn = sum(labels) / len(labels)
            print(f"{step+1:<4}  {len(labels):<5}  {p50:<5.1f}  {p95:<5.1f}  {churn:<10.3f}  {drift_score:<11.3f} {alert}")

        print(f"\n✅ Run {run_id[:6]} logged.")
        print(f"🔗 {config.MLFLOW_TRACKING_URI}/#/experiments/{run.info.experiment_id}/runs/{run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Lint + format**

```bash
uv run ruff check scripts/simulate_traffic.py
uv run ruff format scripts/simulate_traffic.py
```

- [ ] **Step 3: Verify --help works**

```bash
uv run python -m scripts.simulate_traffic --help
```
Expected: help with --mode, --batches, --batch-size, --alias.

- [ ] **Step 4: Commit**

```bash
git add test/scripts/simulate_traffic.py
git commit -m "feat(scripts): add simulate_traffic.py — 1 run per invocation, drift via KS, alert via tag"
```

---

## Phase 9 — Launcher (1 task)

### Task 17: `run.sh` launcher

**Files:**
- Create: `test/run.sh`

- [ ] **Step 1: Write run.sh**

```bash
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "🚀 Booting MLflow tracking server (port 5000)..."
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 --port 5000 &
MLFLOW_PID=$!

cleanup() {
  echo "🛑 Stopping MLflow (PID $MLFLOW_PID)..."
  kill $MLFLOW_PID 2>/dev/null || true
}
trap cleanup EXIT

sleep 3
echo "🎨 Booting Gradio app (port 7860)..."
echo "   - MLflow UI: http://localhost:5000"
echo "   - Gradio:    http://localhost:7860"
uv run python app/gradio_app.py
```

- [ ] **Step 2: Make executable**

```bash
chmod +x run.sh
```

- [ ] **Step 3: Commit**

```bash
git add test/run.sh
git commit -m "feat(launcher): add run.sh — boot MLflow + Gradio with single command"
```

---

## Phase 10 — Gradio App (4 tasks)

### Task 18: Gradio app skeleton + bootstrap

**Files:**
- Create: `test/app/gradio_app.py`

- [ ] **Step 1: Write app skeleton with empty tabs**

```python
# app/gradio_app.py
"""Single Gradio app: 3 tabs (Training Lab / Inference / A/B Test).

Action surface only — observation is in MLflow UI (port 5000).
"""
import gradio as gr
import mlflow

from mlops_churn import config


def build_training_lab() -> None:
    gr.Markdown("## 🎛️ Training Lab\n_(diisi di Task 19)_")


def build_inference() -> None:
    gr.Markdown("## 🚀 Inference\n_(diisi di Task 20)_")


def build_ab_test() -> None:
    gr.Markdown("## 🔀 A/B Test\n_(diisi di Task 21)_")


def build_app() -> gr.Blocks:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    with gr.Blocks(title="BNI Churn MLOps Simulation") as demo:
        gr.Markdown("# 🏦 BNI Churn — MLOps Simulation")
        with gr.Tabs():
            with gr.Tab("🎛️ Training Lab"):
                build_training_lab()
            with gr.Tab("🚀 Inference"):
                build_inference()
            with gr.Tab("🔀 A/B Test"):
                build_ab_test()
        gr.Markdown(
            f"📊 MLflow UI: [{config.MLFLOW_TRACKING_URI}]({config.MLFLOW_TRACKING_URI})"
        )
    return demo


if __name__ == "__main__":
    build_app().launch(server_port=7860, share=False)
```

- [ ] **Step 2: Verify launchable (Ctrl+C immediately)**

```bash
timeout 5 uv run python app/gradio_app.py 2>&1 | head -10
```
Expected: Gradio starts, prints `Running on local URL: http://0.0.0.0:7860`. Timeout kills it after 5s.

- [ ] **Step 3: Lint + format**

```bash
uv run ruff check app/gradio_app.py
uv run ruff format app/gradio_app.py
```

- [ ] **Step 4: Commit**

```bash
git add test/app/gradio_app.py
git commit -m "feat(app): Gradio skeleton with 3 empty tabs"
```

---

### Task 19: Training Lab tab

**Files:**
- Modify: `test/app/gradio_app.py`

- [ ] **Step 1: Replace `build_training_lab()` with full implementation**

```python
# Replace the placeholder build_training_lab() with:
from mlops_churn import train


def _params_for_algo(algo: str) -> dict:
    return config.HYPERPARAM_SPEC[algo]


def _slider_components(algo: str):
    """Build sliders for a given algo. Returns dict {param_name: gr.Slider}."""
    spec = _params_for_algo(algo)
    sliders = {}
    for param_name, p in spec.items():
        sliders[param_name] = gr.Slider(
            minimum=p["min"],
            maximum=p["max"],
            value=p["default"],
            label=param_name,
            step=0.01 if isinstance(p["default"], float) else 1,
        )
    return sliders


def build_training_lab() -> None:
    gr.Markdown("## 🎛️ Training Lab — atur knob → train → MLflow merekam")

    algo = gr.Dropdown(
        choices=list(config.HYPERPARAM_SPEC.keys()),
        value="random_forest",
        label="Algoritma",
    )

    # Build per-algo column blocks; only 1 visible at a time
    col_groups = {}
    slider_refs = {}
    for a in config.HYPERPARAM_SPEC:
        with gr.Column(visible=(a == "random_forest")) as col:
            slider_refs[a] = _slider_components(a)
        col_groups[a] = col

    tag = gr.Textbox(label="Tag opsional", placeholder="demo-pm-april")
    btn = gr.Button("🚀 Train & Log to MLflow", variant="primary")

    status = gr.Markdown("", visible=False)
    metrics_md = gr.Markdown("", visible=False)
    run_link = gr.Markdown("", visible=False)

    def _toggle_columns(selected_algo):
        return [gr.update(visible=(a == selected_algo)) for a in config.HYPERPARAM_SPEC]

    algo.change(_toggle_columns, algo, list(col_groups.values()))

    def _run_training(algo_value, *all_params):
        """Receive algo + all sliders flattened; pick relevant params + train.

        Inputs are passed positionally by Gradio in the order defined in `inputs=[...]`.
        Order: [algo, *all_sliders] — algo is first, sliders flattened across all algos follow.
        """
        all_specs = list(config.HYPERPARAM_SPEC.items())
        offset = 0
        per_algo = {}
        for a, spec in all_specs:
            per_algo[a] = dict(zip(spec.keys(), all_params[offset : offset + len(spec)], strict=True))
            offset += len(spec)
        params = per_algo[algo_value]

        try:
            run_id = train.train_one(algo_value, params, source="gradio-lab")
            metrics = mlflow.get_run(run_id).data.metrics
            metrics_table = (
                "| Metric    | Value |\n|-----------|-------|\n"
                f"| Accuracy  | {metrics['accuracy']:.3f} |\n"
                f"| F1        | {metrics['f1']:.3f} |\n"
                f"| ROC AUC   | {metrics['roc_auc']:.3f} |\n"
                f"| Precision | {metrics['precision']:.3f} |\n"
                f"| Recall    | {metrics['recall']:.3f} |"
            )
            return (
                gr.update(value="✅ Selesai", visible=True),
                gr.update(value=metrics_table, visible=True),
                gr.update(
                    value=f"🔗 Run ID: `{run_id}` — [Buka di MLflow UI]({config.MLFLOW_TRACKING_URI}/#/experiments/0/runs/{run_id})",
                    visible=True,
                ),
            )
        except Exception as e:
            return (
                gr.update(value=f"❌ Error: {e}", visible=True),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
            )

    # Flatten all sliders for input
    all_sliders = []
    for a in config.HYPERPARAM_SPEC:
        all_sliders.extend(slider_refs[a].values())

    btn.click(
        _run_training,
        inputs=[algo, *all_sliders],
        outputs=[status, metrics_md, run_link],
    )
```

- [ ] **Step 2: Smoke launch (Ctrl+C after seeing UI)**

```bash
timeout 5 uv run python app/gradio_app.py 2>&1 | head -10
```
Expected: clean launch, no errors.

- [ ] **Step 3: Lint + format**

```bash
uv run ruff check app/gradio_app.py && uv run ruff format app/gradio_app.py
```

- [ ] **Step 4: Commit**

```bash
git add test/app/gradio_app.py
git commit -m "feat(app): implement Training Lab tab — dynamic sliders + MLflow training trigger"
```

---

### Task 20: Inference tab + reusable customer form

**Files:**
- Modify: `test/app/gradio_app.py`

- [ ] **Step 1: Add `render_customer_form()` helper + Inference tab**

```python
# Add this helper before build_inference:
from mlops_churn import serving


def render_customer_form() -> dict:
    """Render the 10-field customer form. Returns dict of components."""
    components = {}
    schema = config.CUSTOMER_FEATURE_SCHEMA
    with gr.Row():
        with gr.Column():
            gr.Markdown("**Demografi**")
            components["CreditScore"] = gr.Slider(
                schema["CreditScore"]["min"], schema["CreditScore"]["max"],
                value=schema["CreditScore"]["default"], step=1, label="CreditScore"
            )
            components["Age"] = gr.Slider(
                schema["Age"]["min"], schema["Age"]["max"],
                value=schema["Age"]["default"], step=1, label="Age"
            )
            components["Tenure"] = gr.Slider(
                schema["Tenure"]["min"], schema["Tenure"]["max"],
                value=schema["Tenure"]["default"], step=1, label="Tenure (years)"
            )
            components["Geography"] = gr.Dropdown(
                choices=schema["Geography"]["choices"],
                value=schema["Geography"]["default"], label="Geography"
            )
            components["Gender"] = gr.Radio(
                choices=schema["Gender"]["choices"],
                value=schema["Gender"]["default"], label="Gender"
            )
        with gr.Column():
            gr.Markdown("**Akun**")
            components["Balance"] = gr.Number(
                value=schema["Balance"]["default"], label="Balance",
                minimum=schema["Balance"]["min"], maximum=schema["Balance"]["max"]
            )
            components["NumOfProducts"] = gr.Slider(
                schema["NumOfProducts"]["min"], schema["NumOfProducts"]["max"],
                value=schema["NumOfProducts"]["default"], step=1, label="NumOfProducts"
            )
            components["HasCrCard"] = gr.Radio(
                choices=[("Yes", 1), ("No", 0)],
                value=schema["HasCrCard"]["default"], label="HasCrCard"
            )
            components["IsActiveMember"] = gr.Radio(
                choices=[("Yes", 1), ("No", 0)],
                value=schema["IsActiveMember"]["default"], label="IsActiveMember"
            )
            components["EstimatedSalary"] = gr.Number(
                value=schema["EstimatedSalary"]["default"], label="EstimatedSalary",
                minimum=schema["EstimatedSalary"]["min"], maximum=schema["EstimatedSalary"]["max"]
            )
    return components


def _features_dict_from_components(component_values: list, keys: list[str]) -> dict:
    return dict(zip(keys, component_values, strict=True))


def build_inference() -> None:
    gr.Markdown("## 🚀 Inference — Prediksi Churn")

    status = gr.Markdown("📦 Model serving: _(klik Refresh)_")
    refresh_btn = gr.Button("🔄 Refresh", size="sm")

    components = render_customer_form()
    feature_keys = list(components.keys())

    predict_btn = gr.Button("🔮 Predict", variant="primary")
    result_label = gr.Markdown("", visible=False)
    result_meta = gr.Markdown("", visible=False)

    def _refresh_status():
        try:
            mv = registry.get_version_by_alias(config.ALIAS_PRODUCTION)
            return f"📦 Model serving: `{config.REGISTERED_MODEL_NAME}` **v{mv.version}** (@{config.ALIAS_PRODUCTION})"
        except Exception:
            return "⚠️ Belum ada model `@production`. Jalankan `uv run python -m scripts.seed_runs` dulu."

    refresh_btn.click(_refresh_status, outputs=status)

    def _do_predict(*values):
        features = _features_dict_from_components(values, feature_keys)
        try:
            serving._cache = None  # force fresh load
            out = serving.predict(features, alias=config.ALIAS_PRODUCTION)
            label_text = "🔴 **LIKELY TO CHURN**" if out["label"] == 1 else "🟢 **LIKELY TO STAY**"
            return (
                gr.update(value=f"{label_text} — probability: {out['prob']*100:.1f}%", visible=True),
                gr.update(value=f"Latency: {out['latency_ms']:.1f} ms · Model v{out['version']}", visible=True),
            )
        except Exception as e:
            return (
                gr.update(value=f"❌ Error: {e}", visible=True),
                gr.update(value="", visible=False),
            )

    predict_btn.click(_do_predict, inputs=list(components.values()), outputs=[result_label, result_meta])
```

- [ ] **Step 2: Import registry at top of file**

Edit: ensure `from mlops_churn import config, registry, serving, train` is consolidated at imports.

- [ ] **Step 3: Smoke launch**

```bash
timeout 5 uv run python app/gradio_app.py 2>&1 | head -10
```

- [ ] **Step 4: Lint + format**

```bash
uv run ruff check app/gradio_app.py && uv run ruff format app/gradio_app.py
```

- [ ] **Step 5: Commit**

```bash
git add test/app/gradio_app.py
git commit -m "feat(app): implement Inference tab + reusable render_customer_form helper"
```

---

### Task 21: A/B Test tab

**Files:**
- Modify: `test/app/gradio_app.py`

- [ ] **Step 1: Replace `build_ab_test()`**

```python
def build_ab_test() -> None:
    gr.Markdown("## 🔀 A/B Test — Bandingkan Production vs Staging")

    status = gr.Markdown("📦 _(klik Refresh)_")
    refresh_btn = gr.Button("🔄 Refresh", size="sm")

    sample_choices = ["Custom (isi manual)"] + list(config.AB_TEST_SAMPLES.keys())
    sample_dropdown = gr.Dropdown(
        choices=sample_choices, value="Custom (isi manual)", label="Pilih customer contoh"
    )

    components = render_customer_form()
    feature_keys = list(components.keys())

    compare_btn = gr.Button("🔀 Compare A/B", variant="primary")
    result_md = gr.Markdown("", visible=False)
    agreement_md = gr.Markdown("", visible=False)

    def _refresh_status():
        try:
            prod = registry.get_version_by_alias(config.ALIAS_PRODUCTION).version
        except Exception:
            prod = "—"
        try:
            stag = registry.get_version_by_alias(config.ALIAS_STAGING).version
        except Exception:
            stag = "—"
        return f"🅰️ Production: **v{prod}**  |  🅱️ Staging: **v{stag}**"

    refresh_btn.click(_refresh_status, outputs=status)

    def _load_sample(name: str):
        if name == "Custom (isi manual)":
            return [components[k].value for k in feature_keys]
        sample = config.AB_TEST_SAMPLES[name]
        return [sample[k] for k in feature_keys]

    sample_dropdown.change(_load_sample, sample_dropdown, list(components.values()))

    def _do_compare(*values):
        features = _features_dict_from_components(values, feature_keys)
        try:
            serving._cache = None
            out = serving.predict_ab(features)
            prod = out["production"]
            stag = out["staging"]
            prod_lab = "🔴 CHURN" if prod["label"] == 1 else "🟢 STAY"
            stag_lab = "🔴 CHURN" if stag["label"] == 1 else "🟢 STAY"

            md = (
                f"| | Production v{prod['version']} | Staging v{stag['version']} |\n"
                f"|---|---|---|\n"
                f"| Verdict   | {prod_lab} | {stag_lab} |\n"
                f"| Probability | {prod['prob']*100:.1f}% | {stag['prob']*100:.1f}% |\n"
                f"| Latency   | {prod['latency_ms']:.1f} ms | {stag['latency_ms']:.1f} ms |\n"
            )
            agreement = "🟢 Kedua model SETUJU" if out["agreement"] else "🟡 Model BERBEDA pendapat — kandidat case yang menarik"
            return (
                gr.update(value=md, visible=True),
                gr.update(value=agreement, visible=True),
            )
        except Exception as e:
            return (
                gr.update(value=f"❌ Error: {e}", visible=True),
                gr.update(value="", visible=False),
            )

    compare_btn.click(_do_compare, inputs=list(components.values()), outputs=[result_md, agreement_md])
```

- [ ] **Step 2: Smoke launch**

```bash
timeout 5 uv run python app/gradio_app.py 2>&1 | head -10
```

- [ ] **Step 3: Lint + format**

```bash
uv run ruff check app/gradio_app.py && uv run ruff format app/gradio_app.py
```

- [ ] **Step 4: Commit**

```bash
git add test/app/gradio_app.py
git commit -m "feat(app): implement A/B Test tab with sample preset loader + agreement banner"
```

---

## Phase 11 — Polish (2 tasks)

### Task 22: Run full test suite + ruff check

**Files:** (no changes; verification only)

- [ ] **Step 1: Run pytest**

```bash
uv run pytest tests/ -v
```
Expected: 14 PASS, < 30 seconds total.

- [ ] **Step 2: Run ruff lint**

```bash
uv run ruff check .
```
Expected: All checks passed!

- [ ] **Step 3: Run ruff format check**

```bash
uv run ruff format --check .
```
Expected: No format issues.

- [ ] **Step 4: If any failures, fix + re-commit per affected file**

Iterate until all 3 above are clean.

---

### Task 23: Add EDA notebook stub + finalize README

**Files:**
- Create: `test/notebooks/01_eda.ipynb`
- Modify: `test/README.md`

- [ ] **Step 1: Create minimal EDA notebook**

```bash
cat > notebooks/01_eda.ipynb <<'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Churn Modelling — Exploratory Data Analysis\n\nQuick exploration of the dataset before training. Not part of the pipeline."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["from mlops_churn import data\ndf = data.load_raw()\ndf.head()"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["df.describe()"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["df['Exited'].value_counts(normalize=True)"]
  }
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
```

- [ ] **Step 2: Append demo flow to README**

Open `test/README.md` and append:

```markdown

## Demo Flow (≤ 15 menit)

1. Pre-demo: `uv run python -m scripts.seed_runs` → 9 baseline runs + register top 2
2. Step 1 (2m): Buka MLflow UI `:5000` — show 2 experiments
3. Step 2 (3m): MLflow → `churn-prediction` → Compare 3 runs side-by-side
4. Step 3 (3m): Gradio Training Lab → adjust slider → train → run baru tercatat di MLflow
5. Step 4 (2m): MLflow Models → set alias `staging` ke versi yang baru di-train
6. Step 5 (2m): Gradio Inference → predict customer pakai @production
7. Step 6 (2m): Gradio A/B Test → compare prod vs staging
8. Step 7 (1m): Terminal — `uv run python -m scripts.simulate_traffic --mode drifted` → MLflow `production-monitoring` → drift_score chart + alert tag
```

- [ ] **Step 3: Commit**

```bash
git add test/notebooks/01_eda.ipynb test/README.md
git commit -m "docs: add EDA notebook stub + demo flow section in README"
```

---

## Final Verification Checklist (after all 23 tasks)

- [ ] `uv run pytest -v` → 14 tests PASS in < 30s
- [ ] `uv run ruff check .` → clean
- [ ] `uv run ruff format --check .` → clean
- [ ] User downloads Churn_Modelling.csv → `data/raw/`
- [ ] `uv run python -m scripts.seed_runs` → 9 runs + 2 aliases
- [ ] `./run.sh` → MLflow on :5000, Gradio on :7860
- [ ] Manual smoke: each Gradio tab works (Training Lab trains, Inference predicts, A/B compares)
- [ ] `uv run python -m scripts.simulate_traffic --mode drifted --batches 5` → run logged with alert tag

---

## Appendix — File Manifest (final state)

```
test/
├── pyproject.toml              ✅ Task 1, 3
├── uv.lock                     ✅ Task 1
├── .python-version             ✅ Task 1
├── .gitignore                  ✅ Task 2
├── README.md                   ✅ Task 4, 23
├── run.sh                      ✅ Task 17
├── src/mlops_churn/
│   ├── __init__.py             ✅ Task 2
│   ├── config.py               ✅ Task 5
│   ├── data.py                 ✅ Task 7-9
│   ├── train.py                ✅ Task 10
│   ├── registry.py             ✅ Task 11
│   ├── serving.py              ✅ Task 12
│   └── monitoring.py           ✅ Task 13
├── scripts/
│   ├── __init__.py             ✅ Task 2
│   ├── seed_runs.py            ✅ Task 14
│   ├── promote.py              ✅ Task 15
│   └── simulate_traffic.py     ✅ Task 16
├── app/
│   └── gradio_app.py           ✅ Task 18-21
├── tests/
│   ├── __init__.py             ✅ Task 2
│   ├── conftest.py             ✅ Task 6
│   ├── test_data.py            ✅ Task 7-9
│   ├── test_train.py           ✅ Task 10
│   ├── test_registry.py        ✅ Task 11
│   ├── test_serving.py         ✅ Task 12
│   └── test_monitoring.py      ✅ Task 13
├── notebooks/01_eda.ipynb      ✅ Task 23
├── data/{raw,processed}/.gitkeep  ✅ Task 2
└── docs/superpowers/specs/     (already exists pre-plan)
```

**Total: 23 tasks, ~115 steps, 14 test cases.**
