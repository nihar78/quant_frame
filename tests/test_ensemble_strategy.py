"""Tests for the ensemble meta-learner strategy."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_frame.core.model_strategy import BaseModelStrategy
from quant_frame.strategies.ensemble_strategy import EnsembleStrategy


class DummyStrategy(BaseModelStrategy):
    """Deterministic dummy strategy for unit testing."""

    def __init__(self, constant: float) -> None:
        self.constant = constant
        self._is_fitted = False

    def train(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str | None = None,
    ) -> None:
        self._is_fitted = True

    def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Not fitted")
        return np.full(len(df), self.constant, dtype=np.float64)

    def save(self, filepath: str) -> None:
        pass


class TestEnsembleStrategy:
    """Test suite for :class:`EnsembleStrategy`."""

    @pytest.fixture
    def dummy_df(self) -> pd.DataFrame:
        """Return a small DataFrame for testing."""
        return pd.DataFrame(
            {
                "feat_a": [1.0, 2.0, 3.0],
                "feat_b": [0.5, 1.5, 2.5],
                "target": [1.0, 2.0, 3.0],
            }
        )

    def test_empty_strategy_list_raises(self) -> None:
        """``__init__`` must raise when given an empty strategy list."""
        with pytest.raises(ValueError, match="At least one strategy"):
            EnsembleStrategy(strategies=[])

    def test_train_sets_is_fitted(self, dummy_df: pd.DataFrame) -> None:
        """``train`` fits all children and sets the ensemble fitted flag."""
        s1 = DummyStrategy(constant=1.0)
        s2 = DummyStrategy(constant=2.0)
        ensemble = EnsembleStrategy(strategies=[s1, s2])
        ensemble.train(dummy_df, features=["feat_a", "feat_b"], target="target")
        assert ensemble._is_fitted is True
        assert s1._is_fitted is True
        assert s2._is_fitted is True

    def test_predict_averages_outputs(self, dummy_df: pd.DataFrame) -> None:
        """``predict`` returns the arithmetic mean of child predictions."""
        s1 = DummyStrategy(constant=1.0)
        s2 = DummyStrategy(constant=3.0)
        ensemble = EnsembleStrategy(strategies=[s1, s2])
        ensemble.train(dummy_df, features=["feat_a", "feat_b"], target="target")
        preds = ensemble.predict(dummy_df, features=["feat_a", "feat_b"])
        expected = np.full(len(dummy_df), 2.0, dtype=np.float64)
        np.testing.assert_array_almost_equal(preds, expected)

    def test_predict_before_train_raises(self, dummy_df: pd.DataFrame) -> None:
        """Calling ``predict`` before ``train`` must raise ``ValueError``."""
        ensemble = EnsembleStrategy(strategies=[DummyStrategy(constant=1.0)])
        with pytest.raises(ValueError, match="not been fitted"):
            ensemble.predict(dummy_df, features=["feat_a", "feat_b"])

    def test_save_raises_not_implemented(self, dummy_df: pd.DataFrame) -> None:
        """``save`` must raise ``NotImplementedError``."""
        ensemble = EnsembleStrategy(strategies=[DummyStrategy(constant=1.0)])
        ensemble.train(dummy_df, features=["feat_a", "feat_b"])
        with pytest.raises(NotImplementedError):
            ensemble.save("/tmp/ensemble.json")
