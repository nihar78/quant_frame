"""Tests for financial tearsheet plotting utilities."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from matplotlib.figure import Figure

from quant_frame.performance.plots import plot_financial_tearsheet


class TestPlotFinancialTearsheet:
    """Test suite for :func:`plot_financial_tearsheet`."""

    @staticmethod
    def _make_returns() -> pd.Series:
        """Generate a synthetic daily return series for testing."""
        rng = np.random.default_rng(seed=42)
        dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
        returns = pd.Series(
            rng.normal(loc=0.001, scale=0.02, size=100),
            index=dates,
        )
        return returns

    @patch("matplotlib.pyplot.show")
    def test_executes_without_errors(self, mock_show: object) -> None:
        """``plot_financial_tearsheet`` must execute without raising exceptions."""
        returns = self._make_returns()
        fig = plot_financial_tearsheet(returns)
        assert fig is not None
        plt.close("all")

    @patch("matplotlib.pyplot.show")
    def test_returns_matplotlib_figure(self, mock_show: object) -> None:
        """``plot_financial_tearsheet`` must return a ``matplotlib.figure.Figure``."""
        returns = self._make_returns()
        fig = plot_financial_tearsheet(returns)
        assert isinstance(fig, Figure)
        plt.close("all")

    @patch("matplotlib.pyplot.show")
    def test_empty_series_executes_without_errors(self, mock_show: object) -> None:
        """An empty return series must be handled gracefully."""
        returns = pd.Series([], dtype=float)
        fig = plot_financial_tearsheet(returns)
        assert isinstance(fig, Figure)
        plt.close("all")
