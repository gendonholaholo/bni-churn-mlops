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
