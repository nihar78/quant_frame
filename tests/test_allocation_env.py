"""Tests for the domain-agnostic allocation Gymnasium environment."""

import numpy as np
import pandas as pd
import pytest
from gymnasium import Env
from gymnasium.spaces import Box

from quant_frame.rl import AllocationEnv


def _dummy_df(rows: int = 5) -> pd.DataFrame:
    """Return a reproducible small DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "feat_a": rng.normal(size=rows),
            "feat_b": rng.normal(size=rows),
            "target": rng.normal(size=rows),
        }
    )


class TestAllocationEnvInstantiation:
    """Test suite for constructing :class:`AllocationEnv`."""

    def test_instantiation_with_defaults(self) -> None:
        """An env should instantiate with a DataFrame, feature cols and target col."""
        df = _dummy_df()
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a", "feat_b"],
            target_col="target",
        )
        assert isinstance(env, Env)

    def test_instantiation_with_custom_friction(self) -> None:
        """Friction penalty should be accepted and stored."""
        df = _dummy_df()
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a", "feat_b"],
            target_col="target",
            friction_penalty=0.05,
        )
        assert env.friction_penalty == pytest.approx(0.05)

    def test_default_friction_is_zero(self) -> None:
        """When omitted, friction_penalty must default to 0.0."""
        df = _dummy_df()
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a", "feat_b"],
            target_col="target",
        )
        assert env.friction_penalty == pytest.approx(0.0)


class TestAllocationEnvSpaces:
    """Test suite for action and observation spaces."""

    def test_action_space_is_correct_box(self) -> None:
        """action_space must be a Box of shape (1,) bounded [-1, 1]."""
        df = _dummy_df()
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a", "feat_b"],
            target_col="target",
        )
        assert isinstance(env.action_space, Box)
        assert env.action_space.shape == (1,)
        assert np.isclose(env.action_space.low[0], -1.0)
        assert np.isclose(env.action_space.high[0], 1.0)

    def test_observation_space_shape(self) -> None:
        """observation_space must have shape == len(feature_cols) + 1."""
        df = _dummy_df()
        feature_cols = ["feat_a", "feat_b"]
        env = AllocationEnv(
            data=df,
            feature_cols=feature_cols,
            target_col="target",
        )
        assert isinstance(env.observation_space, Box)
        expected_shape = (len(feature_cols) + 1,)
        assert env.observation_space.shape == expected_shape

    def test_observation_space_dtype_is_float32(self) -> None:
        """The observation space dtype must be float32."""
        df = _dummy_df()
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a"],
            target_col="target",
        )
        assert env.observation_space.dtype == np.float32


class TestAllocationEnvReset:
    """Test suite for the reset API."""

    def test_reset_returns_float32_observation(self) -> None:
        """reset must return an np.ndarray whose dtype is float32."""
        df = _dummy_df()
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a", "feat_b"],
            target_col="target",
        )
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

    def test_reset_returns_dict_info(self) -> None:
        """The second return value must be a dict."""
        df = _dummy_df()
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a", "feat_b"],
            target_col="target",
        )
        obs, info = env.reset(seed=42)
        assert isinstance(info, dict)

    def test_reset_observation_includes_previous_action(self) -> None:
        """The last element of the observation should be the previous action."""
        df = _dummy_df(10)
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a", "feat_b"],
            target_col="target",
        )
        obs, _ = env.reset(seed=42)
        assert obs.shape == (3,)
        assert obs[-1] == pytest.approx(0.0)

    def test_reset_sets_step_to_zero(self) -> None:
        """After reset, the internal step counter must be 0."""
        df = _dummy_df()
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a", "feat_b"],
            target_col="target",
        )
        env.reset(seed=42)
        assert env.current_step == 0


class TestAllocationEnvStep:
    """Test suite for the step API."""

    def test_step_returns_tuple_of_five(self) -> None:
        """step must return (obs, reward, terminated, truncated, info)."""
        df = _dummy_df(3)
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a", "feat_b"],
            target_col="target",
        )
        env.reset(seed=42)
        result = env.step(np.array([0.5], dtype=np.float32))
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_step_advances_row(self) -> None:
        """Calling step should increment current_step."""
        df = _dummy_df(5)
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a", "feat_b"],
            target_col="target",
        )
        env.reset(seed=42)
        env.step(np.array([0.5], dtype=np.float32))
        assert env.current_step == 1

    def test_reward_no_friction(self) -> None:
        """Reward equals action * target when friction is zero."""
        df = pd.DataFrame(
            {
                "feat_a": [1.0, 2.0],
                "feat_b": [3.0, 4.0],
                "target": [10.0, 20.0],
            }
        )
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a", "feat_b"],
            target_col="target",
            friction_penalty=0.0,
        )
        env.reset(seed=42)
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        expected_reward = 0.5 * 10.0
        assert reward == pytest.approx(expected_reward)

    def test_reward_with_friction(self) -> None:
        """Reward must subtract friction_penalty * abs(action - previous_action)."""
        df = pd.DataFrame(
            {
                "feat_a": [1.0, 2.0, 3.0],
                "feat_b": [3.0, 4.0, 5.0],
                "target": [10.0, 20.0, 30.0],
            }
        )
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a", "feat_b"],
            target_col="target",
            friction_penalty=0.1,
        )
        env.reset(seed=42)
        action1 = np.array([0.5], dtype=np.float32)
        obs1, reward1, terminated1, truncated1, info1 = env.step(action1)
        expected_reward1 = (0.5 * 10.0) - (0.1 * abs(0.5 - 0.0))
        assert reward1 == pytest.approx(expected_reward1)

        action2 = np.array([0.8], dtype=np.float32)
        obs2, reward2, terminated2, truncated2, info2 = env.step(action2)
        expected_reward2 = (0.8 * 20.0) - (0.1 * abs(0.8 - 0.5))
        assert reward2 == pytest.approx(expected_reward2)

    def test_terminated_true_at_end(self) -> None:
        """When stepping past the last row, terminated must be True."""
        df = pd.DataFrame(
            {
                "feat_a": [1.0],
                "feat_b": [2.0],
                "target": [10.0],
            }
        )
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a", "feat_b"],
            target_col="target",
        )
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(
            np.array([0.5], dtype=np.float32)
        )
        assert terminated is True

    def test_truncated_is_always_false(self) -> None:
        """This environment never truncates an episode early."""
        df = _dummy_df(2)
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a", "feat_b"],
            target_col="target",
        )
        env.reset(seed=42)
        _, _, _, truncated, _ = env.step(np.array([0.5], dtype=np.float32))
        assert truncated is False

    def test_episode_can_be_run_to_completion(self) -> None:
        """A full episode should run without errors."""
        df = _dummy_df(5)
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a", "feat_b"],
            target_col="target",
        )
        obs, info = env.reset(seed=42)
        terminated = False
        total_reward = 0.0
        step_count = 0
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            if step_count > 100:
                pytest.fail("Episode did not terminate within 100 steps")
        assert step_count == len(df)

    def test_previous_action_tracked_in_observation(self) -> None:
        """The last element of the observation should reflect the previous action."""
        df = pd.DataFrame(
            {
                "feat_a": [1.0, 2.0, 3.0],
                "feat_b": [4.0, 5.0, 6.0],
                "target": [10.0, 20.0, 30.0],
            }
        )
        env = AllocationEnv(
            data=df,
            feature_cols=["feat_a", "feat_b"],
            target_col="target",
        )
        obs, _ = env.reset(seed=42)
        assert obs[-1] == pytest.approx(0.0)

        action = np.array([0.75], dtype=np.float32)
        obs2, _, _, _, _ = env.step(action)
        assert obs2[-1] == pytest.approx(0.75)
