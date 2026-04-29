"""Stateful scalers for feature normalization."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


class ZScoreScaler:
    """Stateful Z-Score normalizer for Pandas ``DataFrame`` objects.

    This scaler computes the mean and population standard deviation (``ddof=0``)
    for specified columns during ``fit()`` and stores them in internal state.
    Subsequent calls to ``transform()`` reuse those values, making this class
    suitable for train/test workflows where look-ahead bias must be avoided.

    Division-by-zero is handled gracefully: if a column's standard deviation is
    ``0``, scaled values for that column default to ``0``.

    All methods return new ``DataFrame`` copies; the input frame is never
    modified.

    Example:
        >>> import pandas as pd
        >>> train = pd.DataFrame({"price": [10.0, 20.0, 30.0]})
        >>> test  = pd.DataFrame({"price": [15.0, 25.0]})
        >>> scaler = ZScoreScaler()
        >>> scaler.fit(train, columns=["price"])
        >>> scaled_test = scaler.transform(test, columns=["price"])
    """

    def __init__(self) -> None:
        """Initialize an unfitted scaler."""
        self._means: Optional[Dict[str, float]] = None
        self._stds: Optional[Dict[str, float]] = None

    def fit(self, df: pd.DataFrame, *, columns: List[str]) -> "ZScoreScaler":
        """Calculate and store the mean and standard deviation of ``columns``.

        Args:
            df: Source DataFrame containing the training data.
            columns: List of column names to compute statistics for.

        Returns:
            The scaler instance (for method chaining).

        Raises:
            KeyError: If any column in ``columns`` is not present in ``df``.
        """
        self._means = {}
        self._stds = {}
        for col in columns:
            self._means[col] = float(df[col].mean())
            self._stds[col] = float(df[col].std(ddof=0))
        return self

    def transform(self, df: pd.DataFrame, *, columns: List[str]) -> pd.DataFrame:
        """Scale ``columns`` using the stored mean and standard deviation.

        Args:
            df: DataFrame to transform.
            columns: List of column names to scale.

        Returns:
            A new DataFrame with the specified columns normalized to z-scores.
            The original ``df`` is left untouched.

        Raises:
            ValueError: If the scaler has not been fitted yet.
            KeyError: If any column in ``columns`` is not present in ``df``.
        """
        if self._means is None or self._stds is None:
            raise ValueError("Scaler has not been fitted")

        result = df.copy()
        for col in columns:
            mean = self._means[col]
            std = self._stds[col]
            if std == 0:
                result[col] = 0.0
            else:
                result[col] = (result[col] - mean) / std
        return result

    def fit_transform(self, df: pd.DataFrame, *, columns: List[str]) -> pd.DataFrame:
        """Fit the scaler on ``df`` and immediately transform it.

        This is a convenience method equivalent to calling ``fit()`` followed
        by ``transform()`` on the same frame.

        Args:
            df: Source DataFrame.
            columns: List of column names to fit and scale.

        Returns:
            A new DataFrame with the specified columns normalized to z-scores.
        """
        self.fit(df, columns=columns)
        return self.transform(df, columns=columns)
