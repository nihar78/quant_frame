"""Domain-agnostic allocation environment for reinforcement learning."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pandas as pd
from gymnasium import spaces


class AllocationEnv(gym.Env[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    """Gymnasium environment for learning continuous weight allocation.

    The agent observes a vector of features and its own previous action, then
    chooses a continuous weight in ``[-1, 1]``. The reward is the product of
    the chosen weight and the current target value, minus a friction penalty
    for changing the weight from one step to the next.

    Attributes:
        data: The full DataFrame containing features and target values.
        feature_cols: Column names used as the observation features.
        target_col: Column name for the value to optimize against.
        friction_penalty: Cost per unit of absolute weight change.
        current_step: Index of the current row in *data*.
        action_space: Continuous box of shape ``(1,)`` in ``[-1, 1]``.
        observation_space: Continuous box of shape
            ``(len(feature_cols) + 1,)`` with dtype ``float32``.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        friction_penalty: float = 0.0,
    ) -> None:
        """Initialize the allocation environment.

        Args:
            data: A pandas DataFrame holding all time-series rows.
            feature_cols: Ordered list of column names that form the state.
            target_col: Column name whose value is multiplied by the action
                to compute the reward.
            friction_penalty: Penalty coefficient for changing the action
                from the previous step. Defaults to ``0.0``.
        """
        super().__init__()
        self.data: pd.DataFrame = data.reset_index(drop=True)
        self.feature_cols: list[str] = feature_cols
        self.target_col: str = target_col
        self.friction_penalty: float = friction_penalty
        self.current_step: int = 0
        self._previous_action: float = 0.0

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        obs_dim = len(self.feature_cols) + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        """Reset the environment to the beginning of the data.

        Args:
            seed: Optional seed for the environment's internal RNG.
            options: Unused, provided for API compatibility.

        Returns:
            observation: The first row's features concatenated with the
                previous action (initially ``0.0``) as a ``float32`` array.
            info: An empty dictionary.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self._previous_action = 0.0
        observation = self._get_observation()
        return observation, {}

    def step(
        self,
        action: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Execute one time step within the environment.

        Args:
            action: A ``float32`` array of shape ``(1,)`` representing the
                continuous weight to allocate.

        Returns:
            observation: The next row's features plus the previous action.
            reward: Scalar reward for this step.
            terminated: ``True`` if the episode has reached the end of the
                data, otherwise ``False``.
            truncated: Always ``False`` for this environment.
            info: An empty dictionary.
        """
        action_value = float(action[0])
        target_value = float(
            self.data[self.target_col].iloc[self.current_step]
        )
        reward = (
            action_value * target_value
        ) - (
            self.friction_penalty * abs(action_value - self._previous_action)
        )

        self._previous_action = action_value
        self.current_step += 1

        terminated = self.current_step >= len(self.data)
        truncated = False

        if terminated:
            observation = np.zeros(
                (len(self.feature_cols) + 1,), dtype=np.float32
            )
        else:
            observation = self._get_observation()  # type: ignore[assignment]

        return observation, float(reward), terminated, truncated, {}

    def _get_observation(self) -> npt.NDArray[np.float32]:
        """Build the observation vector for the current step.

        Returns:
            A ``float32`` array of shape ``(len(feature_cols) + 1,)`` where
            the last element is the previous action.
        """
        features = self.data[self.feature_cols].iloc[self.current_step].to_numpy()
        observation = np.concatenate(
            [
                features.astype(np.float32, copy=False),
                np.array([self._previous_action], dtype=np.float32),
            ]
        )
        return observation
