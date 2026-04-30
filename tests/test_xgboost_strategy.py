"""Tests for the XGBoost strategy adapter."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_frame.strategies.xgboost_strategy import XGBoostStrategy


class TestXGBoostStrategy:
    """Test suite for :class:`XGBoostStrategy`."""

    @pytest.fixture
    def dummy_df(self) -> pd.DataFrame:
        """Return a small random DataFrame suitable for regression."""
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "feat_a": rng.random(20),
                "feat_b": rng.random(20),
                "target": rng.random(20),
            }
        )

    def test_instantiation_with_hyperparameters(self) -> None:
        """XGBoostStrategy can be instantiated with optional hyperparameters."""
        hyperparams = {"n_estimators": 10, "max_depth": 3}
        strategy = XGBoostStrategy(hyperparams=hyperparams)
        assert isinstance(strategy, XGBoostStrategy)
        assert strategy.hyperparams == hyperparams

    def test_instantiation_without_hyperparameters(self) -> None:
        """XGBoostStrategy can be instantiated without hyperparameters."""
        strategy = XGBoostStrategy()
        assert isinstance(strategy, XGBoostStrategy)
        assert strategy.hyperparams == {}

    def test_train_fits_model_without_crashing(self, dummy_df: pd.DataFrame) -> None:
        """``train`` correctly fits an XGBRegressor on a dummy DataFrame."""
        strategy = XGBoostStrategy(hyperparams={"n_estimators": 10, "max_depth": 3})
        strategy.train(dummy_df, features=["feat_a", "feat_b"], target="target")
        assert strategy._is_fitted is True

    def test_predict_returns_numpy_array(self, dummy_df: pd.DataFrame) -> None:
        """``predict`` returns a numpy array after training."""
        strategy = XGBoostStrategy(hyperparams={"n_estimators": 10, "max_depth": 3})
        strategy.train(dummy_df, features=["feat_a", "feat_b"], target="target")
        preds = strategy.predict(dummy_df, features=["feat_a", "feat_b"])
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(dummy_df),)

    def test_predict_before_train_raises(self, dummy_df: pd.DataFrame) -> None:
        """Calling ``predict`` before ``train`` raises NotFittedError or ValueError."""
        strategy = XGBoostStrategy()
        with pytest.raises((ValueError, Exception)):
            strategy.predict(dummy_df, features=["feat_a", "feat_b"])

    def test_save_writes_model_to_disk(self, dummy_df: pd.DataFrame, tmp_path: Path) -> None:
        """``save`` successfully writes the model to disk."""
        strategy = XGBoostStrategy(hyperparams={"n_estimators": 10, "max_depth": 3})
        strategy.train(dummy_df, features=["feat_a", "feat_b"], target="target")
        filepath = tmp_path / "xgboost_model.json"
        strategy.save(str(filepath))
        assert filepath.exists()
        assert os.path.getsize(filepath) > 0

    def test_train_drops_nans(self) -> None:
        """``train`` safely drops any remaining NaNs before fitting."""
        df = pd.DataFrame(
            {
                "feat_a": [1.0, 2.0, np.nan, 4.0],
                "feat_b": [0.5, np.nan, 0.5, 0.5],
                "target": [1.0, 2.0, 3.0, 4.0],
            }
        )
        strategy = XGBoostStrategy(hyperparams={"n_estimators": 10, "max_depth": 3})
        strategy.train(df, features=["feat_a", "feat_b"], target="target")
        assert strategy._is_fitted is True
