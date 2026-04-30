"""Tests for the Gaussian HMM strategy adapter."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_frame.strategies.hmm_strategy import GaussianHMMStrategy


class TestGaussianHMMStrategy:
    """Test suite for :class:`GaussianHMMStrategy`."""

    @pytest.fixture
    def dummy_df(self) -> pd.DataFrame:
        """Return a small random DataFrame suitable for HMM fitting."""
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "feat_a": rng.random(20),
                "feat_b": rng.random(20),
            }
        )

    def test_instantiation_with_n_components(self) -> None:
        """GaussianHMMStrategy can be instantiated with n_components."""
        strategy = GaussianHMMStrategy(n_components=3)
        assert isinstance(strategy, GaussianHMMStrategy)
        assert strategy.n_components == 3
        assert strategy.covariance_type is None

    def test_instantiation_with_covariance_type(self) -> None:
        """GaussianHMMStrategy can be instantiated with an optional covariance_type."""
        strategy = GaussianHMMStrategy(n_components=2, covariance_type="full")
        assert isinstance(strategy, GaussianHMMStrategy)
        assert strategy.covariance_type == "full"

    def test_instantiation_defaults(self) -> None:
        """GaussianHMMStrategy uses sensible defaults when no arguments are given."""
        strategy = GaussianHMMStrategy()
        assert isinstance(strategy, GaussianHMMStrategy)
        assert strategy.n_components == 2
        assert strategy.covariance_type is None
        assert strategy.hyperparams == {}

    def test_train_fits_model_without_crashing(self, dummy_df: pd.DataFrame) -> None:
        """``train`` correctly fits a GaussianHMM on a dummy DataFrame."""
        strategy = GaussianHMMStrategy(n_components=2, covariance_type="full")
        strategy.train(dummy_df, features=["feat_a", "feat_b"])
        assert strategy._is_fitted is True
        assert strategy._model is not None

    def test_train_ignores_target_col(self, dummy_df: pd.DataFrame) -> None:
        """``train`` does not use *target* even when it is provided."""
        df = dummy_df.copy()
        df["target"] = np.random.default_rng(42).random(len(df))
        strategy = GaussianHMMStrategy(n_components=2)
        # Should not raise, and should ignore target completely
        strategy.train(df, features=["feat_a", "feat_b"], target="target")
        assert strategy._is_fitted is True

    def test_train_accepts_none_target_col(self, dummy_df: pd.DataFrame) -> None:
        """``train`` accepts ``target=None`` without failing."""
        strategy = GaussianHMMStrategy(n_components=2)
        strategy.train(dummy_df, features=["feat_a", "feat_b"], target=None)
        assert strategy._is_fitted is True

    def test_train_drops_nans(self) -> None:
        """``train`` safely drops any NaNs across feature columns before fitting."""
        df = pd.DataFrame(
            {
                "feat_a": [1.0, 2.0, np.nan, 4.0],
                "feat_b": [0.5, np.nan, 0.5, 0.5],
            }
        )
        strategy = GaussianHMMStrategy(n_components=2)
        strategy.train(df, features=["feat_a", "feat_b"])
        assert strategy._is_fitted is True

    def test_predict_returns_numpy_array_of_ints(self, dummy_df: pd.DataFrame) -> None:
        """``predict`` returns a numpy array of integer hidden-state labels."""
        strategy = GaussianHMMStrategy(n_components=2, covariance_type="full")
        strategy.train(dummy_df, features=["feat_a", "feat_b"])
        preds = strategy.predict(dummy_df, features=["feat_a", "feat_b"])
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(dummy_df),)
        assert np.issubdtype(preds.dtype, np.integer)
        assert set(preds).issubset({0, 1})

    def test_predict_before_train_raises(self, dummy_df: pd.DataFrame) -> None:
        """Calling ``predict`` before ``train`` raises NotFittedError or ValueError."""
        strategy = GaussianHMMStrategy()
        with pytest.raises((ValueError, Exception)):
            strategy.predict(dummy_df, features=["feat_a", "feat_b"])

    def test_save_writes_model_to_disk(self, dummy_df: pd.DataFrame, tmp_path: Path) -> None:
        """``save`` successfully writes the model to disk using joblib."""
        strategy = GaussianHMMStrategy(n_components=2, covariance_type="full")
        strategy.train(dummy_df, features=["feat_a", "feat_b"])
        filepath = tmp_path / "hmm_model.joblib"
        strategy.save(str(filepath))
        assert filepath.exists()
        assert os.path.getsize(filepath) > 0

    def test_save_before_train_raises(self, tmp_path: Path) -> None:
        """Calling ``save`` before ``train`` raises an error."""
        strategy = GaussianHMMStrategy()
        with pytest.raises((ValueError, Exception)):
            strategy.save(str(tmp_path / "hmm_model.joblib"))
