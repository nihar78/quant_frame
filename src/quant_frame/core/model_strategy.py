"""Abstract base class and registry for machine-learning strategies.

This module defines the :class:`BaseModelStrategy` contract that every
machine-learning model in the framework must adhere to, and provides a
lightweight :class:`ModelRegistry` for dynamic, string-keyed lookup of
strategy implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import pandas as pd


class BaseModelStrategy(ABC):
    """Abstract base class for all machine-learning strategies.

    A *strategy* encapsulates the lifecycle of a machine-learning model:
    training on historical data, generating predictions from new data, and
    persisting the trained artefact to (and later loading from) a file path.
    Every concrete strategy must implement :meth:`train`, :meth:`predict`,
    and :meth:`save`.

    Example:
        >>> class MyRandomForest(BaseModelStrategy):
        ...     def train(self, df: pd.DataFrame, features: list[str], target: str | None = None) -> None:
        ...         pass
        ...     def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        ...         return np.array([])
        ...     def save(self, filepath: str) -> None:
        ...         pass
    """

    @abstractmethod
    def train(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str | None = None,
    ) -> None:
        """Fit the model on the supplied training data.

        Args:
            df: Input tabular data containing both features and (optionally)
                the target variable.
            features: Ordered list of column names in *df* that serve as
                explanatory variables.
            target: Name of the column in *df* that contains the response
                variable.  May be ``None`` for unsupervised strategies.
        """

    @abstractmethod
    def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        """Generate predictions for the given observation(s).

        Args:
            df: Input tabular data containing the explanatory variables.
            features: Ordered list of column names in *df* that the model was
                trained on.

        Returns:
            A NumPy array containing the model's predictions.  The shape and
            dtype are strategy-dependent (e.g. ``(n_samples,)`` for
            regression / classification, or ``(n_samples, n_states)`` for
            probabilistic models).
        """

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Persist the trained model to *filepath*.

        The concrete implementation decides the serialisation format
        (pickle, joblib, JSON, ONNX, etc.).  The path should be treated as
        opaque—callers are responsible for ensuring parent directories exist.

        Args:
            filepath: Destination filesystem path for the model artefact.
        """


class ModelRegistry:
    """Lightweight class-level registry for :class:`BaseModelStrategy` subclasses.

    Strategies are registered under human-readable string keys and can be
    retrieved later for dynamic instantiation (e.g. from a pipeline config).

    Example:
        >>> ModelRegistry.register("rf", MyRandomForest)
        >>> ModelRegistry.get("rf")
        <class 'MyRandomForest'>
    """

    _store: ClassVar[dict[str, type[BaseModelStrategy]]] = {}

    @classmethod
    def register(cls, name: str, strategy: type[BaseModelStrategy]) -> None:
        """Register a strategy class under *name*.

        Args:
            name: Unique identifier used to retrieve the strategy later.
            strategy: A concrete subclass of :class:`BaseModelStrategy`.
        """
        cls._store[name] = strategy

    @classmethod
    def get(cls, name: str) -> type[BaseModelStrategy]:
        """Retrieve a previously registered strategy class by *name*.

        Args:
            name: The key that was passed to :meth:`register`.

        Returns:
            The strategy class associated with *name*.

        Raises:
            ValueError: If *name* has not been registered.
        """
        if name not in cls._store:
            raise ValueError(f"Strategy '{name}' is not registered.")
        return cls._store[name]
