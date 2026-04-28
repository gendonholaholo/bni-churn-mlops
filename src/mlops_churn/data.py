"""Data ingestion + preprocessing + train/val/test split."""

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_churn import config

# Drop these — they leak identifiers, no model use
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
    """Drop leak/identifier columns. Returns DataFrame with raw features + target.

    NOTE: Numeric scaling + categorical one-hot are done INSIDE the model Pipeline at
    training time (see train.train_one). This means inference can pass raw features
    directly — no separate scaler artifact to manage.
    """
    df = df.copy()
    drop = [c for c in _LEAK_COLUMNS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    return df


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
