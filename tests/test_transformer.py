"""Tests for the backward-looking time-series feature transformer."""

import numpy as np
import pandas as pd
import pytest

from quant_frame.analytics.transformer import TimeSeriesTransformer


class TestTimeSeriesTransformer:
    """Test suite for :class:`TimeSeriesTransformer`."""

    def test_instantiation(self) -> None:
        """A ``TimeSeriesTransformer`` should be instantiable."""
        transformer = TimeSeriesTransformer()
        assert isinstance(transformer, TimeSeriesTransformer)

    # --------------------------------------------------------------------- #
    #  add_moving_average
    # --------------------------------------------------------------------- #

    def test_add_moving_average_calculates_rolling_mean(self) -> None:
        """Rolling mean should be correct; early rows without full history NaN."""
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=idx)

        transformer = TimeSeriesTransformer()
        result = transformer.add_moving_average(df, column="price", window=3)

        assert "price_ma_3" in result.columns
        assert pd.isna(result.loc["2024-01-01", "price_ma_3"])
        assert pd.isna(result.loc["2024-01-02", "price_ma_3"])
        assert result.loc["2024-01-03", "price_ma_3"] == pytest.approx(20.0)
        assert result.loc["2024-01-04", "price_ma_3"] == pytest.approx(30.0)
        assert result.loc["2024-01-05", "price_ma_3"] == pytest.approx(40.0)

    def test_add_moving_average_returns_new_dataframe(self) -> None:
        """The original DataFrame must remain unmodified."""
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0]}, index=idx)
        original = df.copy()

        transformer = TimeSeriesTransformer()
        transformer.add_moving_average(df, column="price", window=2)

        pd.testing.assert_frame_equal(df, original)

    def test_add_moving_average_preserves_existing_columns(self) -> None:
        """Existing columns should survive in the returned frame."""
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0], "volume": [100, 200, 300]}, index=idx)

        transformer = TimeSeriesTransformer()
        result = transformer.add_moving_average(df, column="price", window=2)

        assert "price" in result.columns
        assert "volume" in result.columns
        assert "price_ma_2" in result.columns

    # --------------------------------------------------------------------- #
    #  add_lag
    # --------------------------------------------------------------------- #

    def test_add_lag_shifts_column_down(self) -> None:
        """Lag should shift values down, leaving NaN at the top."""
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=idx)

        transformer = TimeSeriesTransformer()
        result = transformer.add_lag(df, column="price", lag=1)

        assert "price_lag_1" in result.columns
        assert pd.isna(result.loc["2024-01-01", "price_lag_1"])
        assert result.loc["2024-01-02", "price_lag_1"] == pytest.approx(10.0)
        assert result.loc["2024-01-03", "price_lag_1"] == pytest.approx(20.0)
        assert result.loc["2024-01-04", "price_lag_1"] == pytest.approx(30.0)
        assert result.loc["2024-01-05", "price_lag_1"] == pytest.approx(40.0)

    def test_add_lag_with_lag_greater_than_one(self) -> None:
        """Lag > 1 should shift by the specified number of periods."""
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=idx)

        transformer = TimeSeriesTransformer()
        result = transformer.add_lag(df, column="price", lag=2)

        assert "price_lag_2" in result.columns
        assert pd.isna(result.loc["2024-01-01", "price_lag_2"])
        assert pd.isna(result.loc["2024-01-02", "price_lag_2"])
        assert result.loc["2024-01-03", "price_lag_2"] == pytest.approx(10.0)
        assert result.loc["2024-01-04", "price_lag_2"] == pytest.approx(20.0)
        assert result.loc["2024-01-05", "price_lag_2"] == pytest.approx(30.0)

    def test_add_lag_returns_new_dataframe(self) -> None:
        """The original DataFrame must not be mutated."""
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0]}, index=idx)
        original = df.copy()

        transformer = TimeSeriesTransformer()
        transformer.add_lag(df, column="price", lag=1)

        pd.testing.assert_frame_equal(df, original)

    def test_add_lag_preserves_existing_columns(self) -> None:
        """Existing columns should remain intact after adding a lag."""
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0], "volume": [100, 200, 300]}, index=idx)

        transformer = TimeSeriesTransformer()
        result = transformer.add_lag(df, column="price", lag=1)

        assert "price" in result.columns
        assert "volume" in result.columns
        assert "price_lag_1" in result.columns
