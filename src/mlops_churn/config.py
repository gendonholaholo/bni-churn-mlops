"""Single source of truth for paths, MLflow naming, hyperparameters, schemas, thresholds.

This module has NO internal imports — anyone can import from here without circular deps.
"""

import os
from pathlib import Path

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # test/
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Churn_Modelling.csv"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---- MLflow ----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
# Public URL for the MLflow UI used in Gradio links/prints. Mirrors the tracking
# URI when it's HTTP; otherwise falls back to the local server started by run.sh.
MLFLOW_UI_URL = os.getenv(
    "MLFLOW_UI_URL",
    MLFLOW_TRACKING_URI if MLFLOW_TRACKING_URI.startswith("http") else "http://localhost:5001",
)
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

# ---- Optuna (XGBoost only — see SoW item 21) ----
OPTUNA_N_TRIALS = 100
OPTUNA_RANDOM_SEED = 42
OPTUNA_SEARCH_SPACE = {
    "learning_rate": {"low": 0.001, "high": 0.3, "log": True},
    "n_estimators": {"low": 50, "high": 1000, "log": False},
    "max_depth": {"low": 2, "high": 12, "log": False},
    "min_child_weight": {"low": 1, "high": 10, "log": False},
    "subsample": {"low": 0.5, "high": 1.0, "log": False},
    "colsample_bytree": {"low": 0.5, "high": 1.0, "log": False},
}

# ---- Hyperparameter spec per algorithm (used by seed, Gradio Lab, tests) ----
HYPERPARAM_SPEC = {
    "logistic_regression": {
        "C": {"min": 0.01, "max": 10.0, "default": 1.0},
        "max_iter": {"min": 100, "max": 2000, "default": 500},
    },
    "random_forest": {
        "n_estimators": {"min": 10, "max": 500, "default": 100},
        "max_depth": {"min": 3, "max": 30, "default": 10},
        "min_samples_split": {"min": 2, "max": 20, "default": 2},
    },
    "xgboost": {
        "n_estimators": {"min": 10, "max": 500, "default": 100},
        "learning_rate": {"min": 0.01, "max": 0.5, "default": 0.1},
        "max_depth": {"min": 3, "max": 15, "default": 6},
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
    "CreditScore": {"type": "int", "min": 350, "max": 850, "default": 650},
    "Geography": {
        "type": "categorical",
        "choices": ["France", "Spain", "Germany"],
        "default": "France",
    },
    "Gender": {"type": "categorical", "choices": ["Male", "Female"], "default": "Male"},
    "Age": {"type": "int", "min": 18, "max": 95, "default": 40},
    "Tenure": {"type": "int", "min": 0, "max": 10, "default": 5},
    "Balance": {"type": "float", "min": 0, "max": 300000, "default": 50000},
    "NumOfProducts": {"type": "int", "min": 1, "max": 4, "default": 1},
    "HasCrCard": {"type": "binary", "default": 1},  # 1=Yes, 0=No
    "IsActiveMember": {"type": "binary", "default": 1},
    "EstimatedSalary": {"type": "float", "min": 10000, "max": 200000, "default": 80000},
}

NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
]
CATEGORICAL_FEATURES = ["Geography", "Gender"]
BINARY_FEATURES = ["HasCrCard", "IsActiveMember"]
TARGET = "Exited"

# ---- A/B Test sample customers (3 presets) ----
AB_TEST_SAMPLES = {
    "Young high-balance": {
        "CreditScore": 720,
        "Geography": "France",
        "Gender": "Female",
        "Age": 28,
        "Tenure": 3,
        "Balance": 150000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 95000,
    },
    "Senior low-balance": {
        "CreditScore": 580,
        "Geography": "Germany",
        "Gender": "Male",
        "Age": 62,
        "Tenure": 8,
        "Balance": 5000,
        "NumOfProducts": 1,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 45000,
    },
    "Mid-age active": {
        "CreditScore": 680,
        "Geography": "Spain",
        "Gender": "Male",
        "Age": 42,
        "Tenure": 5,
        "Balance": 75000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 110000,
    },
}
