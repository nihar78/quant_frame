"""PPO continuous allocation strategy adapter.

This module provides a concrete implementation of :class:`BaseModelStrategy`
using ``stable_baselines3.PPO`` trained inside an :class:`AllocationEnv`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from stable_baselines3 import PPO

from quant_frame.core.model_strategy import BaseModelStrategy
from quant_frame.rl.allocation_env import AllocationEnv

LOGGER = logging.getLogger(__name__)


class PPOStrategy(BaseModelStrategy):
    """Concrete strategy wrapping a Proximal Policy Optimisation agent.

    The strategy encapsulates the full model lifecycle: training an RL agent
    inside an :class:`AllocationEnv`, generating continuous allocation weights
    between ``-1`` and ``1``, and persisting the policy network to disk via
    SB3's ``save`` method.

    Args:
        total_timesteps: Number of environment steps used during training.
            Defaults to ``1000``.
        hyperparams: Optional dictionary of keyword arguments passed directly
            to ``stable_baselines3.PPO`` (e.g. *learning_rate*, *n_steps*,
            etc.).

    Example:
        >>> import pandas as pd
        >>> from quant_frame.strategies.ppo_strategy import PPOStrategy
        >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
        >>> strategy = PPOStrategy(total_timesteps=100)
        >>> strategy.train(df, features=["x"], target="y")
        >>> preds = strategy.predict(df, features=["x"])
    """

    def __init__(
        self,
        total_timesteps: int = 1000,
        hyperparams: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the strategy.

        Args:
            total_timesteps: Number of timesteps to train the PPO agent for.
            hyperparams: Optional dictionary of PPO hyperparameters.
        """
        self.total_timesteps: int = total_timesteps
        self.hyperparams: dict[str, Any] = hyperparams if hyperparams is not None else {}
        self._model: PPO | None = None
        self._is_fitted: bool = False

    def train(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str | None = None,
    ) -> None:
        """Fit the PPO agent inside an :class:`AllocationEnv`.

        Any rows containing NaN values across the requested *features* or the
        *target* are silently dropped before training.

        Args:
            df: Input tabular data containing both features and the response.
            features: Ordered list of column names used as explanatory
                variables.
            target: Name of the column in *df* that contains the target
                variable.

        Raises:
            ValueError: If *target* is ``None``.
        """
        if target is None:
            raise ValueError("target must be provided for PPOStrategy.train()")

        subset = df[features + [target]].dropna()
        if len(subset) < len(df):
            LOGGER.info(
                "Dropped %d row(s) with NaN values before training.",
                len(df) - len(subset),
            )

        env = AllocationEnv(
            data=subset,
            feature_cols=features,
            target_col=target,
        )

        self._model = PPO("MlpPolicy", env, verbose=0, **self.hyperparams)
        self._model.learn(total_timesteps=self.total_timesteps)
        self._is_fitted = True

    def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        """Generate continuous allocation weights for each row.

        Because the observation requires the previous action, predictions are
        generated sequentially row-by-row.

        Args:
            df: Input tabular data containing the explanatory variables.
            features: Ordered list of column names the model was trained on.

        Returns:
            A one-dimensional NumPy array of continuous allocation weights
            with length equal to the number of rows in *df*.  Each weight is
            bounded in ``[-1.0, 1.0]``.

        Raises:
            ValueError: If the model has not yet been trained.
        """
        if not self._is_fitted or self._model is None:
            raise ValueError(
                "The model has not been fitted yet. Call train() before predict()."
            )

        previous_action: float = 0.0
        results: list[float] = []

        for _, row in df.iterrows():
            feature_vec = row[features].to_numpy(dtype=np.float32)
            observation = np.concatenate(
                [
                    feature_vec,
                    np.array([previous_action], dtype=np.float32),
                ]
            )
            action, _ = self._model.predict(observation, deterministic=True)
            action_value = float(action[0])
            previous_action = action_value
            results.append(action_value)

        return np.array(results, dtype=np.float32)

    def save(self, filepath: str) -> None:
        """Persist the trained PPO model to *filepath*.

        The model is serialised using Stable-Baselines3's native ``save``
        method, which writes both the policy weights and the replay buffer.

        Args:
            filepath: Destination filesystem path.  Parent directories are
                created automatically if they do not exist.

        Raises:
            ValueError: If the model has not yet been trained.
        """
        if not self._is_fitted or self._model is None:
            raise ValueError(
                "The model has not been fitted yet. Call train() before save()."
            )

        dest = Path(filepath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(dest))
