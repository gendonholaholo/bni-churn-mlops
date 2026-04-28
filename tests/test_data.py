import pandas as pd

from mlops_churn import config, data


def test_load_raw_returns_dataframe(tmp_path, monkeypatch, synthetic_dataframe):
    """load_raw reads CSV at config.DATA_RAW_PATH and returns DataFrame."""
    csv_path = tmp_path / "Churn_Modelling.csv"
    synthetic_dataframe.to_csv(csv_path, index=False)
    monkeypatch.setattr(config, "DATA_RAW_PATH", csv_path)

    df = data.load_raw()

    assert isinstance(df, pd.DataFrame)
    assert "Exited" in df.columns
    assert len(df) == 100


def test_preprocess_output_schema(synthetic_dataframe):
    """preprocess drops only leak columns; categoricals + numerics stay raw."""
    out = data.preprocess(synthetic_dataframe)

    # Leak columns dropped if present
    assert "RowNumber" not in out.columns
    assert "CustomerId" not in out.columns
    assert "Surname" not in out.columns

    # Categoricals stay as strings (not encoded)
    assert "Geography" in out.columns
    assert "Gender" in out.columns
    assert out["Geography"].dtype == object  # still strings

    # Numerics stay in original units (not scaled)
    assert "CreditScore" in out.columns

    # Target preserved
    assert "Exited" in out.columns

    # No NaN
    assert out.isna().sum().sum() == 0


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
