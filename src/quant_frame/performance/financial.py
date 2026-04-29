"""Financial performance metrics implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_frame.core.metrics import BaseMetrics


class FinancialMetrics(BaseMetrics):
    """Concrete metrics implementation for financial strategy evaluation.

    Computes standard portfolio-performance indicators from a DataFrame that
    contains predicted daily returns (``pred_col``) and (optionally) actual
    daily returns (``actual_col``).

    All calculations are performed with strictly vectorised pandas and NumPy
    operations.
    """

    def calculate(
        self,
        df: pd.DataFrame,
        actual_col: str,
        pred_col: str,
    ) -> dict[str, float]:
        """Compute financial performance metrics.

        The following metrics are returned:

        * ``cumulative_return`` – total compounded return over the period.
        * ``annualized_sharpe`` – Sharpe ratio annualised with ``sqrt(252)``.
        * ``max_drawdown`` – worst peak-to-trough decline.
        * ``win_rate`` – proportion of days with a positive predicted return.

        Args:
            df: Input tabular data containing at least ``actual_col`` and
                ``pred_col``.
            actual_col: Name of the column holding ground-truth daily returns.
                Currently unused in the financial rubric but kept for API
                consistency with :class:`BaseMetrics`.
            pred_col: Name of the column holding the *strategy's* predicted
                daily returns.

        Returns:
            A dictionary mapping metric names to their numeric values.
            Missing or zero-variance data results in ``0.0`` for each metric.
        """
        returns = df[pred_col]

        if returns.empty:
            return {
                "cumulative_return": 0.0,
                "annualized_sharpe": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }

        # Cumulative return
        cum_returns = (1.0 + returns).cumprod()
        cumulative_return = float(cum_returns.iloc[-1] - 1.0)

        # Annualized Sharpe ratio
        mean_ret = returns.mean()
        std_ret = returns.std(ddof=0)
        if std_ret == 0.0 or pd.isna(std_ret):
            annualized_sharpe = 0.0
        else:
            annualized_sharpe = float(mean_ret / std_ret * np.sqrt(252))

        # Maximum drawdown
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = float(drawdown.min())

        # Win rate
        win_rate = float((returns > 0).mean())

        return {
            "cumulative_return": cumulative_return,
            "annualized_sharpe": annualized_sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
        }
