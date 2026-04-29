"""Time-Series Aligner for handling missing dates and frequency resampling."""

from __future__ import annotations

import pandas as pd


class TimeSeriesAligner:
    """Utility class for aligning, filling, and resampling time-series DataFrames.

    This class provides a set of pure methods that operate on Pandas
    ``DataFrame`` objects with a ``DatetimeIndex``.  Each method returns a new
    ``DataFrame``, leaving the original unmodified.

    Example:
        >>> import pandas as pd
        >>> idx = pd.date_range("2024-01-01", periods=3, freq="B")  # business days
        >>> df = pd.DataFrame({"price": [100.0, 101.0, 102.0]}, index=idx)
        >>> aligner = TimeSeriesAligner()
        >>> daily = aligner.resample_frequency(df, freq="D")
        >>> filled = aligner.forward_fill(daily)
    """

    def resample_frequency(self, df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
        """Resample a DataFrame to a regular calendar frequency, inserting NaNs.

        The supplied DataFrame must have a ``DatetimeIndex``.  The method
        generates a complete date range covering the existing index and then
        reindexes the frame so that any missing periods are represented as
        ``NaN`` rows.

        Args:
            df: Source DataFrame with a ``DatetimeIndex``.
            freq: Target frequency string compatible with Pandas (e.g. ``"D"``
                for daily, ``"H"`` for hourly).  Defaults to ``"D"``.

        Returns:
            A new DataFrame with a regularly-spaced ``DatetimeIndex`` and
            ``NaN`` values for previously missing dates.
        """
        result = df.copy()
        full_range = pd.date_range(start=result.index.min(), end=result.index.max(), freq=freq)
        return result.reindex(full_range)

    def forward_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Carry the last known observation forward through ``NaN`` rows.

        Args:
            df: DataFrame potentially containing ``NaN`` values.

        Returns:
            A new DataFrame where each ``NaN`` has been replaced by the most
            recent non-NaN value in the same column.
        """
        return df.ffill()

    def interpolate_linear(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill ``NaN`` values using linear interpolation.

        Linear interpolation computes the straight-line value between two
        non-missing observations and assigns the intermediate points their
        mathematical midpoints along that line.

        Args:
            df: DataFrame potentially containing ``NaN`` values.

        Returns:
            A new DataFrame with ``NaN`` values replaced by linearly
            interpolated estimates.
        """
        return df.interpolate(method="linear")
