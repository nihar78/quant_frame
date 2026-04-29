"""Tests for the model strategy interface and registry."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_frame.core.model_strategy import BaseModelStrategy, ModelRegistry


class TestBaseModelStrategy:
    """Test suite for :class:`BaseModelStrategy`."""

    def test_direct_instantiation_raises_typeerror(self) -> None:
        """Instantiating ``BaseModelStrategy`` directly must raise ``TypeError``."""
        with pytest.raises(TypeError):
            BaseModelStrategy()  # type: ignore[abstract]

    def test_subclass_without_train_raises_typeerror(self) -> None:
        """A subclass that omits ``train`` must be non-instantiable."""

        class DummyStrategy(BaseModelStrategy):
            def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
                return np.array([])

            def save(self, filepath: str) -> None:
                pass

        with pytest.raises(TypeError):
            DummyStrategy()  # type: ignore[abstract]

    def test_subclass_without_predict_raises_typeerror(self) -> None:
        """A subclass that omits ``predict`` must be non-instantiable."""

        class DummyStrategy(BaseModelStrategy):
            def train(
                self,
                df: pd.DataFrame,
                features: list[str],
                target: str | None = None,
            ) -> None:
                pass

            def save(self, filepath: str) -> None:
                pass

        with pytest.raises(TypeError):
            DummyStrategy()  # type: ignore[abstract]

    def test_subclass_without_save_raises_typeerror(self) -> None:
        """A subclass that omits ``save`` must be non-instantiable."""

        class DummyStrategy(BaseModelStrategy):
            def train(
                self,
                df: pd.DataFrame,
                features: list[str],
                target: str | None = None,
            ) -> None:
                pass

            def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
                return np.array([])

        with pytest.raises(TypeError):
            DummyStrategy()  # type: ignore[abstract]

    def test_valid_subclass_instantiates(self) -> None:
        """A fully implemented subclass must instantiate without error."""

        class ValidStrategy(BaseModelStrategy):
            def train(
                self,
                df: pd.DataFrame,
                features: list[str],
                target: str | None = None,
            ) -> None:
                pass

            def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
                return np.array([])

            def save(self, filepath: str) -> None:
                pass

        strategy = ValidStrategy()
        assert isinstance(strategy, BaseModelStrategy)


class TestModelRegistry:
    """Test suite for :class:`ModelRegistry`."""

    def test_register_and_get(self) -> None:
        """Registering a valid strategy and retrieving it by key must work."""

        class ValidStrategy(BaseModelStrategy):
            def train(
                self,
                df: pd.DataFrame,
                features: list[str],
                target: str | None = None,
            ) -> None:
                pass

            def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
                return np.array([])

            def save(self, filepath: str) -> None:
                pass

        ModelRegistry.register("valid", ValidStrategy)
        retrieved = ModelRegistry.get("valid")
        assert retrieved is ValidStrategy

    def test_get_unregistered_key_raises_valueerror(self) -> None:
        """Requesting an unregistered key must raise ``ValueError``."""
        with pytest.raises(ValueError, match="not registered"):
            ModelRegistry.get("non_existent_strategy")
