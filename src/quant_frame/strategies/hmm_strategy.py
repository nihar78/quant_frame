"""Gaussian Hidden Markov Model strategy adapter.

This module provides a concrete implementation of :class:`BaseModelStrategy`
using ``hmmlearn.hmm.GaussianHMM`` for unsupervised regime detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from quant_frame.core.model_strategy import BaseModelStrategy

LOGGER = logging.getLogger(__name__)


class GaussianHMMStrategy(BaseModelStrategy):
    """Unsupervised regime detection via a Gaussian Hidden Markov Model.

    The strategy wraps ``hmmlearn.hmm.GaussianHMM`` and exposes the standard
    :class:`BaseModelStrategy` lifecycle: ``train``, ``predict``, and ``save``.
    Because the model is unsupervised, the *target_col* argument is ignored.

    Args:
        n_components: Number of hidden states (regimes) to fit.  Defaults to 2.
        covariance_type: Type of covariance parameters to use.  One of
            ``"spherical"``, ``"diag"``, ``"full"``, or ``"tied"``.  If
            omitted, the ``hmmlearn`` default (``"spherical"``) is used.
        hyperparams: Optional dictionary of additional keyword arguments passed
            directly to ``GaussianHMM`` (e.g. *random_state*, *n_iter*, etc.).

    Example:
        >>> import pandas as pd
        >>> from quant_frame.strategies.hmm_strategy import GaussianHMMStrategy
        >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.5, 1.5, 2.5]})
        >>> strategy = GaussianHMMStrategy(n_components=2)
        >>> strategy.train(df, features=["x", "y"])
        >>> states = strategy.predict(df, features=["x", "y"])
    """

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str | None = None,
        hyperparams: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the strategy.

        Args:
            n_components: Number of hidden states.
            covariance_type: Covariance parameterisation type.
            hyperparams: Additional keyword arguments for ``GaussianHMM``.
        """
        self.n_components: int = n_components
        self.covariance_type: str | None = covariance_type
        self.hyperparams: dict[str, Any] = hyperparams if hyperparams is not None else {}
        self._model: GaussianHMM | None = None
        self._is_fitted: bool = False

    def train(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str | None = None,
    ) -> None:
        """Fit the underlying GaussianHMM on the supplied features.

        Any rows containing NaN values across the requested *features* are
        silently dropped before fitting.  The *target* is ignored because
        the model is unsupervised.

        Args:
            df: Input tabular data containing the explanatory variables.
            features: Ordered list of column names used as explanatory
                variables.
            target: Ignored.  Present only for API compatibility with
                :class:`BaseModelStrategy`.
        """
        subset = df[features].dropna()
        if len(subset) < len(df):
            LOGGER.info(
                "Dropped %d row(s) with NaN values before training.",
                len(df) - len(subset),
            )

        x = subset[features].to_numpy(dtype=np.float64)

        model_kwargs = dict(self.hyperparams)
        model_kwargs["n_components"] = self.n_components
        if self.covariance_type is not None:
            model_kwargs["covariance_type"] = self.covariance_type

        self._model = GaussianHMM(**model_kwargs)
        self._model.fit(x)
        self._is_fitted = True

    def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        """Decode the most likely hidden-state sequence for the observations.

        Args:
            df: Input tabular data containing the explanatory variables.
            features: Ordered list of column names the model was trained on.

        Returns:
            A one-dimensional NumPy array of integer state labels with length
            equal to the number of rows in *df*.

        Raises:
            ValueError: If the model has not yet been trained.
        """
        if not self._is_fitted or self._model is None:
            raise ValueError(
                "The model has not been fitted yet. Call train() before predict()."
            )

        x = df[features].to_numpy(dtype=np.float64)
        states: np.ndarray = self._model.predict(x)
        return states

    def save(self, filepath: str) -> None:
        """Persist the trained model to *filepath* using ``joblib``.

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
        joblib.dump(self._model, dest)
