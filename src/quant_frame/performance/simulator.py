"""Vectorised back-test simulator implementation."""

from __future__ import annotations

import pandas as pd


class VectorizedSimulator:
    """A vectorised simulator that converts model signals into strategy returns.

    This simulator enforces strict temporal execution by shifting the signal
    forward by one row before multiplying it by the actual returns.  This
    prevents look-ahead bias: a signal generated at time ``t`` is only
    applied to the realised return at time ``t+1``.

    Missing signals are treated as cash (``0.0``).
    """

    def simulate(
        self,
        df: pd.DataFrame,
        *,
        signal_col: str = "predicted",
        return_col: str = "actual",
    ) -> pd.Series:
        """Convert model signals into strategy returns.

        The implementation follows these steps:

        1. Extract ``signal_col`` from ``df``.
        2. Shift the series forward by 1 row so that the signal at index
           ``t`` is aligned with the return at index ``t+1``.
        3. Replace any missing values (including the first row after the
           shift) with ``0.0``, representing a flat cash position.
        4. Multiply element-wise by ``return_col`` to obtain realised
           strategy returns.

        Args:
            df: Input tabular data containing at least ``signal_col`` and
                ``return_col``.
            signal_col: Name of the column holding the model's predicted
                signals (positions).  Defaults to ``"predicted"``.
            return_col: Name of the column holding the ground-truth daily
                returns.  Defaults to ``"actual"``.

        Returns:
            A :class:`pandas.Series` of daily strategy returns with the same
            index as ``df``.
        """
        positions: pd.Series = df[signal_col].shift(1).fillna(0.0)
        strategy_returns: pd.Series = positions * df[return_col]
        return strategy_returns
