"""Abstract base classes for performance metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseMetrics(ABC):
    """Abstract base class for all performance metrics.

    A *metrics* class is responsible for computing quantitative performance
    indicators from a DataFrame containing actual and predicted values.
    Every concrete metrics implementation must implement the
    :meth:`calculate` method.

    Example:
        >>> class MyMetrics(BaseMetrics):
        ...     def calculate(
        ...         self,
        ...         df: pd.DataFrame,
        ...         actual_col: str,
        ...         pred_col: str,
        ...     ) -> dict[str, float]:
        ...         return {"score": 0.0}
    """

    @abstractmethod
    def calculate(
        self,
        df: pd.DataFrame,
        actual_col: str,
        pred_col: str,
    ) -> dict[str, float]:
        """Compute performance metrics for the supplied data.

        Args:
            df: Input tabular data containing at least the actual and
                predicted value columns.
            actual_col: Name of the column in *df* that contains the
                ground-truth values.
            pred_col: Name of the column in *df* that contains the
                predicted values.

        Returns:
            A dictionary mapping metric names to their numeric values.
        """
