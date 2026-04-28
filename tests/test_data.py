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
