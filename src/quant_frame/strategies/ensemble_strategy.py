"""Mathematical ensemble strategy that averages predictions of sub-strategies.

This module provides a concrete implementation of :class:`BaseModelStrategy`
that acts as a meta-learner by delegating to a list of instantiated strategies
and returning the arithmetic mean of their predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_frame.core.model_strategy import BaseModelStrategy

LOGGER = logging.getLogger(__name__)


class EnsembleStrategy(BaseModelStrategy):
    """Meta-learner that averages predictions from multiple sub-strategies.

    Each constituent strategy must already be instantiated.  :meth:`train` is
    forwarded to every child, and :meth:`predict` returns the element-wise
    mean across all child predictions.

    Args:
        strategies: Ordered list of instantiated :class:`BaseModelStrategy`
            objects to ensemble.

    Example:
        >>> from quant_frame.strategies.xgboost_strategy import XGBoostStrategy
        >>> s1 = XGBoostStrategy(hyperparams={"n_estimators": 10})
        >>> s2 = XGBoostStrategy(hyperparams={"n_estimators": 20})
        >>> ensemble = EnsembleStrategy(strategies=[s1, s2])
        >>> ensemble.train(df, features=["x"], target="y")
        >>> preds = ensemble.predict(df, features=["x"])
    """

    def __init__(self, strategies: list[BaseModelStrategy]) -> None:
        """Initialise the ensemble.

        Args:
            strategies: Non-empty list of strategy instances.
        """
        if not strategies:
            raise ValueError("At least one strategy must be provided to EnsembleStrategy.")
        self.strategies: list[BaseModelStrategy] = strategies
        self._is_fitted: bool = False

    def train(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str | None = None,
    ) -> None:
        """Train every constituent strategy on the supplied data.

        Args:
            df: Input tabular data containing both features and (optionally)
                the target variable.
            features: Ordered list of column names in *df* that serve as
                explanatory variables.
            target: Name of the column in *df* that contains the response
                variable.  May be ``None`` for unsupervised strategies.
        """
        for strategy in self.strategies:
            strategy.train(df, features=features, target=target)
        self._is_fitted = True

    def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        """Generate the averaged prediction of all constituent strategies.

        Args:
            df: Input tabular data containing the explanatory variables.
            features: Ordered list of column names the sub-models were
                trained on.

        Returns:
            A NumPy array containing the mean prediction across all
            strategies, computed along axis ``0``.

        Raises:
            ValueError: If the ensemble has not yet been trained.
        """
        if not self._is_fitted:
            raise ValueError(
                "The ensemble has not been fitted yet. Call train() before predict()."
            )

        predictions: list[np.ndarray] = []
        for strategy in self.strategies:
            preds = strategy.predict(df, features=features)
            predictions.append(preds)

        stacked = np.stack(predictions, axis=0)
        return np.mean(stacked, axis=0)

    def save(self, filepath: str) -> None:
        """Persist the ensemble to *filepath*.

        Because an ensemble is a collection of already-trained models,
        this method currently raises :class:`NotImplementedError`.

        Args:
            filepath: Destination filesystem path.
        """
        raise NotImplementedError("EnsembleStrategy.save() is not yet supported.")
