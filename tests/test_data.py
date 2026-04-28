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
