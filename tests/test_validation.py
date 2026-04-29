"""Tests for the time-series cross-validation splitter."""

import pandas as pd
import pytest

from quant_frame.validation.splitter import WalkForwardSplitter


class TestWalkForwardSplitterInstantiation:
    """Test suite for :class:`WalkForwardSplitter` construction."""

    def test_instantiation_with_rolling_window(self) -> None:
        """A splitter should be instantiable with train_size, test_size, and 'rolling' window_type."""
        splitter = WalkForwardSplitter(train_size=10, test_size=5, window_type="rolling")
        assert splitter.train_size == 10
        assert splitter.test_size == 5
        assert splitter.window_type == "rolling"

    def test_instantiation_with_expanding_window(self) -> None:
        """A splitter should be instantiable with train_size, test_size, and 'expanding' window_type."""
        splitter = WalkForwardSplitter(train_size=10, test_size=5, window_type="expanding")
        assert splitter.train_size == 10
        assert splitter.test_size == 5
        assert splitter.window_type == "expanding"

    def test_default_window_type_is_expanding(self) -> None:
        """If window_type is omitted, it should default to 'expanding'."""
        splitter = WalkForwardSplitter(train_size=10, test_size=5)
        assert splitter.window_type == "expanding"

    def test_invalid_window_type_raises(self) -> None:
        """An unsupported window_type should raise ValueError."""
        with pytest.raises(ValueError, match="window_type"):
            WalkForwardSplitter(train_size=10, test_size=5, window_type="invalid")

    def test_non_positive_train_size_raises(self) -> None:
        """A non-positive train_size should raise ValueError."""
        with pytest.raises(ValueError, match="train_size"):
            WalkForwardSplitter(train_size=0, test_size=5)

    def test_non_positive_test_size_raises(self) -> None:
        """A non-positive test_size should raise ValueError."""
        with pytest.raises(ValueError, match="test_size"):
            WalkForwardSplitter(train_size=10, test_size=-1)


class TestWalkForwardSplitterSplit:
    """Test suite for the ``split`` method behaviour."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """A small chronologically ordered DataFrame for splitter tests."""
        idx = pd.date_range("2024-01-01", periods=20, freq="D")
        return pd.DataFrame({"price": range(20)}, index=idx)

    def test_split_returns_generator(self, sample_df: pd.DataFrame) -> None:
        """``split`` should return a generator of tuples."""
        splitter = WalkForwardSplitter(train_size=5, test_size=3)
        result = splitter.split(sample_df)
        assert hasattr(result, "__iter__")
        first = next(result)
        assert isinstance(first, tuple)
        assert len(first) == 2
        train_df, test_df = first
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

    def test_split_returns_list_of_tuples_when_consumed(self, sample_df: pd.DataFrame) -> None:
        """When consumed, the generator yields (train_df, test_df) tuples."""
        splitter = WalkForwardSplitter(train_size=5, test_size=3)
        splits = list(splitter.split(sample_df))
        assert all(isinstance(t, tuple) and len(t) == 2 for t in splits)
        assert all(
            isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame)
            for train, test in splits
        )

    def test_leakage_check(self, sample_df: pd.DataFrame) -> None:
        """For every split, the latest train timestamp must be strictly earlier
        than the earliest test timestamp."""
        splitter = WalkForwardSplitter(train_size=5, test_size=3)
        for train_df, test_df in splitter.split(sample_df):
            assert train_df.index.max() < test_df.index.min()

    def test_rolling_window_train_size_constant(self, sample_df: pd.DataFrame) -> None:
        """With ``window_type='rolling'``, every training slice should have the
        exact ``train_size`` rows."""
        splitter = WalkForwardSplitter(
            train_size=5, test_size=3, window_type="rolling"
        )
        for train_df, _ in splitter.split(sample_df):
            assert len(train_df) == 5

    def test_rolling_window_train_slides_forward(self, sample_df: pd.DataFrame) -> None:
        """With ``window_type='rolling'``, the train window should slide forward
        by ``test_size`` on each split."""
        splitter = WalkForwardSplitter(
            train_size=5, test_size=3, window_type="rolling"
        )
        splits = list(splitter.split(sample_df))
        assert len(splits) > 1
        # First train: [0:5], second train: [3:8]
        pd.testing.assert_index_equal(splits[0][0].index, sample_df.iloc[0:5].index)
        pd.testing.assert_index_equal(splits[1][0].index, sample_df.iloc[3:8].index)

    def test_expanding_window_train_grows(self, sample_df: pd.DataFrame) -> None:
        """With ``window_type='expanding'``, the training set should grow by
        ``test_size`` rows on each subsequent split."""
        splitter = WalkForwardSplitter(
            train_size=5, test_size=3, window_type="expanding"
        )
        train_lengths = [len(train) for train, _ in splitter.split(sample_df)]
        assert train_lengths == [5, 8, 11, 14, 17]

    def test_expanding_window_test_size_constant(self, sample_df: pd.DataFrame) -> None:
        """With ``window_type='expanding'``, every test slice should have the
        exact ``test_size`` rows."""
        splitter = WalkForwardSplitter(
            train_size=5, test_size=3, window_type="expanding"
        )
        for _, test_df in splitter.split(sample_df):
            assert len(test_df) == 3

    def test_no_splits_when_data_too_short(self, sample_df: pd.DataFrame) -> None:
        """If the DataFrame is shorter than ``train_size + test_size``, no
        splits should be yielded."""
        splitter = WalkForwardSplitter(train_size=50, test_size=10)
        splits = list(splitter.split(sample_df))
        assert splits == []

    def test_non_integer_numeric_input_raises(self) -> None:
        """Float values for train_size / test_size should raise ValueError."""
        with pytest.raises(ValueError, match="train_size"):
            WalkForwardSplitter(train_size=5.5, test_size=2)
