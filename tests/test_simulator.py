"""Tests for the vectorised back-test simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_frame.performance.simulator import VectorizedSimulator


class TestVectorizedSimulator:
    """Test suite for :class:`VectorizedSimulator`."""

    def test_instantiation(self) -> None:
        """``VectorizedSimulator`` must instantiate without error."""
        sim = VectorizedSimulator()
        assert isinstance(sim, VectorizedSimulator)

    def test_simulate_returns_series(self) -> None:
        """``simulate`` must return a ``pd.Series`` of daily strategy returns."""
        sim = VectorizedSimulator()
        df = pd.DataFrame(
            {
                "predicted": [1.0, -1.0, 0.5, -0.5],
                "actual": [0.01, -0.02, 0.03, -0.01],
            }
        )
        result = sim.simulate(df, signal_col="predicted", return_col="actual")
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_simulate_shifts_signal_forward_by_one(self) -> None:
        """The simulator must shift the signal forward by 1 to avoid look-ahead bias."""
        sim = VectorizedSimulator()
        df = pd.DataFrame(
            {
                "predicted": [1.0, -1.0, 0.5, -0.5],
                "actual": [0.01, -0.02, 0.03, -0.01],
            }
        )
        result = sim.simulate(df, signal_col="predicted", return_col="actual")
        # First day: signal shifted -> NaN -> fillna(0.0) -> 0.0 * actual = 0.0
        # Second day: signal from index 0 (1.0) * actual at index 1 (-0.02) = -0.02
        # Third day: signal from index 1 (-1.0) * actual at index 2 (0.03) = -0.03
        # Fourth day: signal from index 2 (0.5) * actual at index 3 (-0.01) = -0.005
        expected = pd.Series([0.0, -0.02, -0.03, -0.005])
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected)

    def test_simulate_first_day_defaults_to_zero(self) -> None:
        """The first day's strategy return must default to ``0.0``."""
        sim = VectorizedSimulator()
        df = pd.DataFrame(
            {
                "predicted": [10.0, 20.0],
                "actual": [0.05, 0.10],
            }
        )
        result = sim.simulate(df, signal_col="predicted", return_col="actual")
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-12)

    def test_simulate_handles_nan_gracefully(self) -> None:
        """Missing signals must be treated as ``0.0`` (cash)."""
        sim = VectorizedSimulator()
        df = pd.DataFrame(
            {
                "predicted": [np.nan, 1.0, np.nan, -1.0],
                "actual": [0.01, 0.02, 0.03, -0.01],
            }
        )
        result = sim.simulate(df, signal_col="predicted", return_col="actual")
        # Shift behaviour:
        #   index 0: NaN (shifted from nowhere) -> fillna -> 0.0
        #   index 1: NaN (shifted from index 0) -> fillna -> 0.0
        #   index 2: 1.0 (shifted from index 1) -> 1.0 * 0.03 = 0.03
        #   index 3: NaN (shifted from index 2) -> fillna -> 0.0 * -0.01 = 0.0
        expected = pd.Series([0.0, 0.0, 0.03, 0.0])
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected)
