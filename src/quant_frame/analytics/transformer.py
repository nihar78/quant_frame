"""Backward-looking time-series feature transformer."""

from __future__ import annotations

import pandas as pd


class TimeSeriesTransformer:
    """Utility class for generating safe, backward-looking time-series features.

    This class provides pure methods that operate on Pandas ``DataFrame``
    objects, returning new frames that include moving-average and lagged
    columns.  The original ``DataFrame`` is never modified.

    Example:
        >>> import pandas as pd
        >>> idx = pd.date_range("2024-01-01", periods=5, freq="D")
        >>> df = pd.DataFrame({"price": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=idx)
        >>> tx = TimeSeriesTransformer()
        >>> df_ma = tx.add_moving_average(df, column="price", window=3)
        >>> df_lag = tx.add_lag(df, column="price", lag=1)
    """

    def add_moving_average(
        self,
        df: pd.DataFrame,
        *,
        column: str,
        window: int,
    ) -> pd.DataFrame:
        """Compute a backward-looking rolling mean and append it as a new column.

        The new column is dynamically named ``{column}_ma_{window}``.
        Rows that do not yet have enough historical observations to fill the
        requested window will contain ``NaN``.

        Args:
            df: Source DataFrame containing the time-series data.
            column: Name of the column on which to compute the moving average.
            window: Number of past periods (inclusive of the current row) to
                include in the rolling mean.

        Returns:
            A new DataFrame with an additional column holding the rolling mean.
            The original ``df`` is left untouched.

        Raises:
            KeyError: If ``column`` is not present in ``df``.
        """
        result = df.copy()
        new_col = f"{column}_ma_{window}"
        result[new_col] = result[column].rolling(window=window).mean()
        return result

    def add_lag(
        self,
        df: pd.DataFrame,
        *,
        column: str,
        lag: int,
    ) -> pd.DataFrame:
        """Create a lagged (backward-shifted) copy of a column.

        The new column is dynamically named ``{column}_lag_{lag}``.
        The first ``lag`` rows will contain ``NaN`` because no earlier
        observation exists.

        Args:
            df: Source DataFrame containing the time-series data.
            column: Name of the column to lag.
            lag: Number of periods to shift backward (positive integer).

        Returns:
            A new DataFrame with an additional column holding the lagged values.
            The original ``df`` is left untouched.

        Raises:
            KeyError: If ``column`` is not present in ``df``.
        """
        result = df.copy()
        new_col = f"{column}_lag_{lag}"
        result[new_col] = result[column].shift(periods=lag)
        return result
