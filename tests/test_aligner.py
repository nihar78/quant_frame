"""Tests for the time-series aligner utility."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from quant_frame.analytics.aligner import TimeSeriesAligner


class TestTimeSeriesAligner:
    """Test suite for :class:`TimeSeriesAligner`."""

    def test_instantiation(self) -> None:
        """A ``TimeSeriesAligner`` should be instantiable."""
        aligner = TimeSeriesAligner()
        assert isinstance(aligner, TimeSeriesAligner)

    # --------------------------------------------------------------------- #
    #  resample_frequency
    # --------------------------------------------------------------------- #

    def test_resample_frequency_inserts_missing_daily_rows(self) -> None:
        """Missing calendar days (e.g. weekends) should be inserted as NaN."""
        # Friday to Monday — weekend should be back-filled as NaN
        idx = pd.DatetimeIndex(
            [dt.date(2024, 1, 5), dt.date(2024, 1, 8)],  # Fri, Mon
        )
        df = pd.DataFrame({"price": [100.0, 103.0]}, index=idx)
        df.index.name = "date"

        aligner = TimeSeriesAligner()
        result = aligner.resample_frequency(df, freq="D")

        expected_idx = pd.date_range(start="2024-01-05", end="2024-01-08", freq="D")
        assert list(result.index) == list(expected_idx)
        assert result.loc["2024-01-05", "price"] == 100.0
        assert result.loc["2024-01-08", "price"] == 103.0
        assert pd.isna(result.loc["2024-01-06", "price"])
        assert pd.isna(result.loc["2024-01-07", "price"])

    def test_resample_frequency_preserves_multiple_columns(self) -> None:
        """All columns in the original frame should survive resampling."""
        idx = pd.DatetimeIndex([dt.date(2024, 1, 1), dt.date(2024, 1, 3)])
        df = pd.DataFrame(
            {"open": [10.0, 12.0], "close": [11.0, 13.0]},
            index=idx,
        )
        aligner = TimeSeriesAligner()
        result = aligner.resample_frequency(df, freq="D")

        assert "open" in result.columns
        assert "close" in result.columns
        assert pd.isna(result.loc["2024-01-02", "open"])
        assert pd.isna(result.loc["2024-01-02", "close"])

    def test_resample_frequency_returns_new_dataframe(self) -> None:
        """The original DataFrame should remain unmodified."""
        idx = pd.DatetimeIndex([dt.date(2024, 1, 1), dt.date(2024, 1, 3)])
        df = pd.DataFrame({"price": [100.0, 102.0]}, index=idx)
        original = df.copy()

        aligner = TimeSeriesAligner()
        aligner.resample_frequency(df, freq="D")

        pd.testing.assert_frame_equal(df, original)

    # --------------------------------------------------------------------- #
    #  forward_fill
    # --------------------------------------------------------------------- #

    def test_forward_fill_carries_last_known_value(self) -> None:
        """NaN rows should inherit the most recent non-NaN value."""
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({"value": [1.0, np.nan, np.nan, 4.0, np.nan]}, index=idx)

        aligner = TimeSeriesAligner()
        result = aligner.forward_fill(df)

        assert result.loc["2024-01-01", "value"] == 1.0
        assert result.loc["2024-01-02", "value"] == 1.0
        assert result.loc["2024-01-03", "value"] == 1.0
        assert result.loc["2024-01-04", "value"] == 4.0
        assert result.loc["2024-01-05", "value"] == 4.0

    def test_forward_fill_returns_new_dataframe(self) -> None:
        """The original DataFrame should not be mutated."""
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"value": [1.0, np.nan, np.nan]}, index=idx)
        original = df.copy()

        aligner = TimeSeriesAligner()
        aligner.forward_fill(df)

        pd.testing.assert_frame_equal(df, original)

    # --------------------------------------------------------------------- #
    #  interpolate_linear
    # --------------------------------------------------------------------- #

    def test_interpolate_linear_fills_midpoint(self) -> None:
        """A single NaN between two known values should become the midpoint."""
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"value": [10.0, np.nan, 20.0]}, index=idx)

        aligner = TimeSeriesAligner()
        result = aligner.interpolate_linear(df)

        assert result.loc["2024-01-01", "value"] == 10.0
        assert result.loc["2024-01-02", "value"] == 15.0
        assert result.loc["2024-01-03", "value"] == 20.0

    def test_interpolate_linear_multiple_gaps(self) -> None:
        """Multiple consecutive NaNs should receive evenly-spaced values."""
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {"value": [0.0, np.nan, np.nan, np.nan, 4.0]}, index=idx
        )

        aligner = TimeSeriesAligner()
        result = aligner.interpolate_linear(df)

        assert result.loc["2024-01-01", "value"] == 0.0
        assert result.loc["2024-01-02", "value"] == 1.0
        assert result.loc["2024-01-03", "value"] == 2.0
        assert result.loc["2024-01-04", "value"] == 3.0
        assert result.loc["2024-01-05", "value"] == 4.0

    def test_interpolate_linear_returns_new_dataframe(self) -> None:
        """The source DataFrame should remain untouched."""
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"value": [10.0, np.nan, 20.0]}, index=idx)
        original = df.copy()

        aligner = TimeSeriesAligner()
        aligner.interpolate_linear(df)

        pd.testing.assert_frame_equal(df, original)
