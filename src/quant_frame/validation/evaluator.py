"""Walk-forward evaluation orchestrator."""

from __future__ import annotations

import pandas as pd

from quant_frame.analytics.scalers import ZScoreScaler
from quant_frame.analytics.transformer import TimeSeriesTransformer
from quant_frame.core.model_strategy import BaseModelStrategy
from quant_frame.validation.splitter import WalkForwardSplitter


class WalkForwardEvaluator:
    """Master pipeline runner that orchestrates splitting, scaling, training,
    and predicting across time for walk-forward cross-validation.

    The evaluator guarantees strict chronological separation between training
    and test data (enforced by the supplied :class:`WalkForwardSplitter`).
    Within each fold it:

    1. Drops rows that contain ``NaN`` in any of the feature or target
       columns (commonly introduced by lag / moving-average transforms).
    2. Fits the :class:`ZScoreScaler` on the *clean training* slice and
       transforms both train and test.
    3. Trains the :class:`BaseModelStrategy` on the scaled training data.
    4. Generates predictions on the scaled test data.

    Results from every fold are concatenated into a single DataFrame that
    covers the entire out-of-sample period.

    Attributes:
        strategy: The machine-learning strategy to train and evaluate.
        splitter: Chronological splitter that yields (train, test) folds.
        transformer: Time-series feature transformer (stored for API
            consistency; callers are expected to apply transforms before
            calling :meth:`evaluate`).
        scaler: Stateful Z-score scaler used to normalise features.
    """

    def __init__(
        self,
        *,
        strategy: BaseModelStrategy,
        splitter: WalkForwardSplitter,
        transformer: TimeSeriesTransformer,
        scaler: ZScoreScaler,
    ) -> None:
        """Initialise the :class:`WalkForwardEvaluator`.

        Args:
            strategy: Concrete model strategy implementing ``train``,
                ``predict``, and ``save``.
            splitter: Walk-forward splitter that produces chronologically
                safe folds.
            transformer: Time-series transformer instance.
            scaler: Z-score scaler instance.
        """
        self.strategy: BaseModelStrategy = strategy
        self.splitter: WalkForwardSplitter = splitter
        self.transformer: TimeSeriesTransformer = transformer
        self.scaler: ZScoreScaler = scaler

    def evaluate(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "target",
        feature_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run the walk-forward evaluation pipeline.

        The method iterates over every (train, test) fold produced by
        :attr:`splitter`, drops ``NaN`` rows, scales features using the
        train-set statistics, trains the strategy, and collects predictions.

        Args:
            df: Chronologically sorted input DataFrame containing both
                features and the target variable.
            target_col: Name of the column holding the ground-truth values.
                Defaults to ``"target"``.
            feature_cols: Ordered list of column names that serve as model
                features. Defaults to ``["price"]`` when ``None``.

        Returns:
            A consolidated :class:`pandas.DataFrame` indexed by the original
            timestamps of all out-of-sample observations. It contains two
            columns:

            * ``actual``    – the ground-truth target values.
            * ``predicted`` – the model's predicted values.
        """
        if feature_cols is None:
            feature_cols = ["price"]

        required_cols: list[str] = list(feature_cols) + [target_col]
        fold_results: list[pd.DataFrame] = []

        for train_df, test_df in self.splitter.split(df):
            # Gracefully remove rows that contain NaN in any required column.
            # This is common after applying lag / moving-average transforms.
            train_clean = train_df.dropna(subset=required_cols).copy()
            test_clean = test_df.dropna(subset=required_cols).copy()

            if train_clean.empty or test_clean.empty:
                continue

            # Fit scaler on train data only, then transform both sets.
            scaled_train = self.scaler.fit_transform(train_clean, columns=feature_cols)
            scaled_test = self.scaler.transform(test_clean, columns=feature_cols)

            # Train the strategy on the scaled training fold.
            self.strategy.train(
                scaled_train,
                features=feature_cols,
                target=target_col,
            )

            # Generate predictions on the scaled test fold.
            predictions = self.strategy.predict(
                scaled_test,
                features=feature_cols,
            )

            # Build a result frame for this fold preserving the original index.
            fold_result = pd.DataFrame(
                {
                    "actual": test_clean[target_col].values,
                    "predicted": predictions,
                },
                index=test_clean.index,
            )
            fold_results.append(fold_result)

        if not fold_results:
            return pd.DataFrame(columns=["actual", "predicted"])

        return pd.concat(fold_results)
