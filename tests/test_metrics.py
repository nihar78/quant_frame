"""Tests for the metrics interface and concrete financial implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_frame.core.metrics import BaseMetrics
from quant_frame.performance.financial import FinancialMetrics


class TestBaseMetrics:
    """Test suite for :class:`BaseMetrics`."""

    def test_direct_instantiation_raises_typeerror(self) -> None:
        """Instantiating ``BaseMetrics`` directly must raise ``TypeError``."""
        with pytest.raises(TypeError):
            BaseMetrics()  # type: ignore[abstract]

    def test_subclass_without_calculate_raises_typeerror(self) -> None:
        """A subclass that omits ``calculate`` must be non-instantiable."""

        class DummyMetrics(BaseMetrics):
            pass

        with pytest.raises(TypeError):
            DummyMetrics()  # type: ignore[abstract]

    def test_valid_subclass_instantiates(self) -> None:
        """A fully implemented subclass must instantiate without error."""

        class ValidMetrics(BaseMetrics):
            def calculate(
                self,
                df: pd.DataFrame,
                actual_col: str,
                pred_col: str,
            ) -> dict[str, float]:
                return {"score": 1.0}

        metrics = ValidMetrics()
        assert isinstance(metrics, BaseMetrics)


class TestFinancialMetrics:
    """Test suite for :class:`FinancialMetrics`."""

    def test_instantiation(self) -> None:
        """``FinancialMetrics`` must instantiate without error."""
        fm = FinancialMetrics()
        assert isinstance(fm, FinancialMetrics)
        assert isinstance(fm, BaseMetrics)

    def test_calculate_returns_expected_keys(self) -> None:
        """``calculate`` must return a dictionary with the required metrics."""
        rng = np.random.default_rng(seed=42)
        returns = rng.normal(loc=0.001, scale=0.02, size=100)
        df = pd.DataFrame(
            {
                "actual": returns,
                "predicted": returns,
            }
        )
        fm = FinancialMetrics()
        result = fm.calculate(df, actual_col="actual", pred_col="predicted")

        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "cumulative_return",
            "annualized_sharpe",
            "max_drawdown",
            "win_rate",
        }
        for key in result:
            assert isinstance(result[key], float)

    def test_calculate_with_zero_returns(self) -> None:
        """All zeros input must yield zero-valued metrics safely."""
        df = pd.DataFrame(
            {
                "actual": np.zeros(50),
                "predicted": np.zeros(50),
            }
        )
        fm = FinancialMetrics()
        result = fm.calculate(df, actual_col="actual", pred_col="predicted")

        assert result["cumulative_return"] == pytest.approx(0.0, abs=1e-12)
        assert result["annualized_sharpe"] == pytest.approx(0.0, abs=1e-12)
        assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
        assert result["win_rate"] == pytest.approx(0.0, abs=1e-12)

    def test_calculate_with_empty_dataframe(self) -> None:
        """An empty DataFrame must yield zero-valued metrics safely."""
        df = pd.DataFrame({"actual": pd.Series([], dtype=float), "predicted": pd.Series([], dtype=float)})
        fm = FinancialMetrics()
        result = fm.calculate(df, actual_col="actual", pred_col="predicted")

        assert result["cumulative_return"] == pytest.approx(0.0, abs=1e-12)
        assert result["annualized_sharpe"] == pytest.approx(0.0, abs=1e-12)
        assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
        assert result["win_rate"] == pytest.approx(0.0, abs=1e-12)
