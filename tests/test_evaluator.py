"""Tests for the walk-forward evaluation orchestrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_frame.analytics.scalers import ZScoreScaler
from quant_frame.analytics.transformer import TimeSeriesTransformer
from quant_frame.core.model_strategy import BaseModelStrategy
from quant_frame.validation.evaluator import WalkForwardEvaluator
from quant_frame.validation.splitter import WalkForwardSplitter


class DummyStrategy(BaseModelStrategy):
    """A trivial strategy that predicts all ones and does nothing on train."""

    def train(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str | None = None,
    ) -> None:
        """No-op training."""

    def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        """Return an array of 1.0s with length equal to *df*."""
        return np.ones(len(df), dtype=float)

    def save(self, filepath: str) -> None:
        """No-op persistence."""


class TestWalkForwardEvaluatorInstantiation:
    """Test suite for :class:`WalkForwardEvaluator` construction."""

    def test_instantiation_with_all_dependencies(self) -> None:
        """Evaluator should be instantiable with strategy, splitter, transformer, and scaler."""
        strategy = DummyStrategy()
        splitter = WalkForwardSplitter(train_size=5, test_size=2)
        transformer = TimeSeriesTransformer()
        scaler = ZScoreScaler()

        evaluator = WalkForwardEvaluator(
            strategy=strategy,
            splitter=splitter,
            transformer=transformer,
            scaler=scaler,
        )

        assert evaluator.strategy is strategy
        assert evaluator.splitter is splitter
        assert evaluator.transformer is transformer


class TestWalkForwardEvaluatorEvaluate:
    """Test suite for the ``evaluate`` method behaviour."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """A small chronologically ordered DataFrame for evaluator tests."""
        idx = pd.date_range("2024-01-01", periods=12, freq="D")
        return pd.DataFrame(
            {
                "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
                "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            },
            index=idx,
        )

    def test_evaluate_returns_dataframe(self, sample_df: pd.DataFrame) -> None:
        """evaluate() must return a pandas DataFrame."""
        evaluator = WalkForwardEvaluator(
            strategy=DummyStrategy(),
            splitter=WalkForwardSplitter(train_size=5, test_size=2),
            transformer=TimeSeriesTransformer(),
            scaler=ZScoreScaler(),
        )
        result = evaluator.evaluate(sample_df, target_col="target", feature_cols=["price"])
        assert isinstance(result, pd.DataFrame)

    def test_evaluate_returns_correct_out_of_sample_rows(self, sample_df: pd.DataFrame) -> None:
        """The result must contain exactly the out-of-sample test rows."""
        evaluator = WalkForwardEvaluator(
            strategy=DummyStrategy(),
            splitter=WalkForwardSplitter(train_size=5, test_size=2),
            transformer=TimeSeriesTransformer(),
            scaler=ZScoreScaler(),
        )
        result = evaluator.evaluate(sample_df, target_col="target", feature_cols=["price"])

        # With train_size=5, test_size=2, and 12 rows (expanding):
        # split 1: train [0:5], test [5:7] -> 2 rows
        # split 2: train [0:7], test [7:9] -> 2 rows
        # split 3: train [0:9], test [9:11] -> 2 rows
        # => 6 test rows total
        assert len(result) == 6

    def test_evaluate_contains_actual_and_predicted(self, sample_df: pd.DataFrame) -> None:
        """The result DataFrame must contain actual target values and predictions."""
        evaluator = WalkForwardEvaluator(
            strategy=DummyStrategy(),
            splitter=WalkForwardSplitter(train_size=5, test_size=2),
            transformer=TimeSeriesTransformer(),
            scaler=ZScoreScaler(),
        )
        result = evaluator.evaluate(sample_df, target_col="target", feature_cols=["price"])

        assert "actual" in result.columns
        assert "predicted" in result.columns

    def test_evaluate_predictions_are_from_strategy(self, sample_df: pd.DataFrame) -> None:
        """Predictions must match the strategy's output (all 1.0 for DummyStrategy)."""
        evaluator = WalkForwardEvaluator(
            strategy=DummyStrategy(),
            splitter=WalkForwardSplitter(train_size=5, test_size=2),
            transformer=TimeSeriesTransformer(),
            scaler=ZScoreScaler(),
        )
        result = evaluator.evaluate(sample_df, target_col="target", feature_cols=["price"])

        assert np.allclose(result["predicted"].values, 1.0)

    def test_evaluate_actuals_match_target(self, sample_df: pd.DataFrame) -> None:
        """Actual values must match the original target for the test timestamps."""
        evaluator = WalkForwardEvaluator(
            strategy=DummyStrategy(),
            splitter=WalkForwardSplitter(train_size=5, test_size=2),
            transformer=TimeSeriesTransformer(),
            scaler=ZScoreScaler(),
        )
        result = evaluator.evaluate(sample_df, target_col="target", feature_cols=["price"])

        expected_index = pd.DatetimeIndex(
            [
                "2024-01-06",
                "2024-01-07",
                "2024-01-08",
                "2024-01-09",
                "2024-01-10",
                "2024-01-11",
            ]
        )
        pd.testing.assert_index_equal(result.index, expected_index)
        expected_actuals = sample_df.loc[expected_index, "target"]
        assert np.allclose(result["actual"].values, expected_actuals.values)


    def test_evaluate_drops_nan_rows_before_scaling_and_model(self) -> None:
        """Rows with NaN in feature or target columns must be dropped before scaler/model."""
        idx = pd.date_range("2024-01-01", periods=8, freq="D")
        df = pd.DataFrame(
            {
                "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            },
            index=idx,
        )
        # Inject NaN into a test row
        df.loc["2024-01-07", "price"] = np.nan

        evaluator = WalkForwardEvaluator(
            strategy=DummyStrategy(),
            splitter=WalkForwardSplitter(train_size=4, test_size=2),
            transformer=TimeSeriesTransformer(),
            scaler=ZScoreScaler(),
        )
        result = evaluator.evaluate(df, target_col="target", feature_cols=["price"])

        # The NaN row should be excluded from predictions
        assert "2024-01-07" not in result.index
        # But the non-NaN test rows should still be present
        assert "2024-01-06" in result.index
        assert "2024-01-08" in result.index

    def test_evaluate_uses_fit_transform_on_train_and_transform_on_test(self) -> None:
        """The scaler must be fit on train data only, then applied to test data."""

        class RecordingScaler(ZScoreScaler):
            """A scaler that records whether fit and transform were called."""

            def __init__(self) -> None:
                super().__init__()
                self.fit_calls: list[pd.DataFrame] = []
                self.transform_calls: list[pd.DataFrame] = []

            def fit(self, df: pd.DataFrame, *, columns: list[str]) -> "RecordingScaler":
                self.fit_calls.append(df.copy())
                return super().fit(df, columns=columns)

            def transform(self, df: pd.DataFrame, *, columns: list[str]) -> pd.DataFrame:
                self.transform_calls.append(df.copy())
                return super().transform(df, columns=columns)

        scaler = RecordingScaler()
        evaluator = WalkForwardEvaluator(
            strategy=DummyStrategy(),
            splitter=WalkForwardSplitter(train_size=4, test_size=2),
            transformer=TimeSeriesTransformer(),
            scaler=scaler,
        )

        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
                "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            },
            index=idx,
        )

        evaluator.evaluate(df, target_col="target", feature_cols=["price"])

        # With train_size=4, test_size=2, and 10 rows (expanding):
        # split 1: train [0:4], test [4:6]
        # split 2: train [0:6], test [6:8]
        # split 3: train [0:8], test [8:10]
        # => 3 splits
        assert len(scaler.fit_calls) == 3
        # fit_transform calls fit then transform; we also call transform explicitly on test
        assert len(scaler.transform_calls) == 6

        # Verify that fit was called on train data of increasing size
        assert len(scaler.fit_calls[0]) == 4
        assert len(scaler.fit_calls[1]) == 6
        assert len(scaler.fit_calls[2]) == 8

    def test_evaluate_returns_consolidated_df_for_rolling_window(self) -> None:
        """Rolling-window splits should also produce a consolidated DataFrame."""
        idx = pd.date_range("2024-01-01", periods=12, freq="D")
        df = pd.DataFrame(
            {
                "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
                "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            },
            index=idx,
        )

        evaluator = WalkForwardEvaluator(
            strategy=DummyStrategy(),
            splitter=WalkForwardSplitter(train_size=4, test_size=2, window_type="rolling"),
            transformer=TimeSeriesTransformer(),
            scaler=ZScoreScaler(),
        )
        result = evaluator.evaluate(df, target_col="target", feature_cols=["price"])

        # rolling: train_size=4, test_size=2
        # i=4: train [0:4], test [4:6]
        # i=6: train [2:6], test [6:8]
        # i=8: train [4:8], test [8:10]
        # i=10: train [6:10], test [10:12]
        # => 4 splits, 8 test rows total
        assert len(result) == 8
        assert "actual" in result.columns
        assert "predicted" in result.columns
