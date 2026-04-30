"""Tests for the PPO continuous strategy adapter."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_frame.strategies.ppo_strategy import PPOStrategy


class TestPPOStrategy:
    """Test suite for :class:`PPOStrategy`."""

    @pytest.fixture
    def dummy_df(self) -> pd.DataFrame:
        """Return a small random DataFrame suitable for RL training."""
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "feat_a": rng.random(20),
                "feat_b": rng.random(20),
                "target": rng.random(20),
            }
        )

    def test_instantiation_with_hyperparameters(self) -> None:
        """PPOStrategy can be instantiated with optional RL hyperparameters."""
        strategy = PPOStrategy(total_timesteps=100)
        assert isinstance(strategy, PPOStrategy)
        assert strategy.total_timesteps == 100

    def test_instantiation_defaults(self) -> None:
        """PPOStrategy uses sensible defaults when no arguments are given."""
        strategy = PPOStrategy()
        assert isinstance(strategy, PPOStrategy)
        assert strategy.total_timesteps == 1000

    def test_train_fits_model_without_crashing(self, dummy_df: pd.DataFrame) -> None:
        """``train`` correctly fits a PPO agent on an AllocationEnv."""
        strategy = PPOStrategy(total_timesteps=50)
        strategy.train(dummy_df, features=["feat_a", "feat_b"], target="target")
        assert strategy._is_fitted is True

    def test_predict_returns_numpy_array_between_minus_one_and_one(
        self, dummy_df: pd.DataFrame
    ) -> None:
        """``predict`` returns a numpy array of continuous allocations in [-1, 1]."""
        strategy = PPOStrategy(total_timesteps=50)
        strategy.train(dummy_df, features=["feat_a", "feat_b"], target="target")
        preds = strategy.predict(dummy_df, features=["feat_a", "feat_b"])
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(dummy_df),)
        assert np.all(preds >= -1.0)
        assert np.all(preds <= 1.0)

    def test_predict_before_train_raises(self, dummy_df: pd.DataFrame) -> None:
        """Calling ``predict`` before ``train`` raises ValueError."""
        strategy = PPOStrategy()
        with pytest.raises(ValueError):
            strategy.predict(dummy_df, features=["feat_a", "feat_b"])

    def test_save_writes_model_to_disk(self, dummy_df: pd.DataFrame, tmp_path: Path) -> None:
        """``save`` successfully writes the PPO model to disk."""
        strategy = PPOStrategy(total_timesteps=50)
        strategy.train(dummy_df, features=["feat_a", "feat_b"], target="target")
        filepath = tmp_path / "ppo_model"
        strategy.save(str(filepath))
        saved_file = filepath.parent / (filepath.name + ".zip")
        assert saved_file.exists()
        assert os.path.getsize(saved_file) > 0

    def test_save_before_train_raises(self, tmp_path: Path) -> None:
        """Calling ``save`` before ``train`` raises an error."""
        strategy = PPOStrategy()
        with pytest.raises(ValueError):
            strategy.save(str(tmp_path / "ppo_model"))
