"""XGBoost regression strategy adapter.

This module provides a concrete implementation of :class:`BaseModelStrategy`
using the scikit-learn API wrapper ``xgboost.XGBRegressor``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from quant_frame.core.model_strategy import BaseModelStrategy

LOGGER = logging.getLogger(__name__)


class XGBoostStrategy(BaseModelStrategy):
    """Concrete strategy wrapping an XGBoost regressor.

    The strategy encapsulates the full model lifecycle: training with optional
    hyperparameter overrides, generating predictions as a NumPy array, and
    persisting the fitted booster to disk via XGBoost's ``save_model``.

    Args:
        hyperparams: Optional dictionary of keyword arguments passed directly
            to ``xgboost.XGBRegressor``.  Common keys include *n_estimators*,
            *max_depth*, *learning_rate*, etc.  If omitted, the XGBoost
            defaults are used.

    Example:
        >>> import pandas as pd
        >>> from quant_frame.strategies.xgboost_strategy import XGBoostStrategy
        >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
        >>> strategy = XGBoostStrategy(hyperparams={"n_estimators": 10})
        >>> strategy.train(df, features=["x"], target="y")
        >>> preds = strategy.predict(df, features=["x"])
    """

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        """Initialise the strategy.

        Args:
            hyperparams: Optional dictionary of XGBRegressor hyperparameters.
        """
        self.hyperparams: dict[str, Any] = hyperparams if hyperparams is not None else {}
        self._model: XGBRegressor | None = None
        self._is_fitted: bool = False

    def train(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str | None = None,
    ) -> None:
        """Fit the underlying XGBRegressor on the supplied data.

        Any rows containing NaN values across the requested *features* or
        the *target* are silently dropped before fitting.

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
            raise ValueError("target must be provided for XGBoostStrategy.train()")

        subset = df[features + [target]].dropna()
        if len(subset) < len(df):
            LOGGER.info(
                "Dropped %d row(s) with NaN values before training.",
                len(df) - len(subset),
            )

        x = subset[features].to_numpy(dtype=np.float64)
        y = subset[target].to_numpy(dtype=np.float64)

        self._model = XGBRegressor(**self.hyperparams)
        self._model.fit(x, y)
        self._is_fitted = True

    def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        """Generate predictions using the trained XGBRegressor.

        Args:
            df: Input tabular data containing the explanatory variables.
            features: Ordered list of column names the model was trained on.

        Returns:
            A one-dimensional NumPy array of predicted values with length equal
            to the number of rows in *df*.

        Raises:
            ValueError: If the model has not yet been trained.
        """
        if not self._is_fitted or self._model is None:
            raise ValueError(
                "The model has not been fitted yet. Call train() before predict()."
            )

        x = df[features].to_numpy(dtype=np.float64)
        return self._model.predict(x)

    def save(self, filepath: str) -> None:
        """Persist the trained model to *filepath*.

        The booster is serialised using XGBoost's native ``save_model`` method
        in JSON format.

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
        self._model.save_model(str(dest))
