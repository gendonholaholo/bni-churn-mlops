"""Data ingestion + preprocessing + train/val/test split."""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from mlops_churn import config

# Columns to drop (identifiers/leak)
_LEAK_COLUMNS = ["RowNumber", "CustomerId", "Surname"]


def load_raw() -> pd.DataFrame:
    """Load Churn Modelling CSV from config.DATA_RAW_PATH."""
    if not config.DATA_RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.DATA_RAW_PATH}. "
            "Download from https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling "
            "and place at data/raw/Churn_Modelling.csv"
        )
    return pd.read_csv(config.DATA_RAW_PATH)


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
