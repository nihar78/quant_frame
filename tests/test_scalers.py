"""Tests for the stateful Z-Score scaler utility."""

import numpy as np
import pandas as pd
import pytest

from quant_frame.analytics.scalers import ZScoreScaler


class TestZScoreScaler:
    """Test suite for :class:`ZScoreScaler`."""

    def test_instantiation(self) -> None:
        """A ``ZScoreScaler`` should be instantiable."""
        scaler = ZScoreScaler()
        assert isinstance(scaler, ZScoreScaler)

    # --------------------------------------------------------------------- #
    #  fit
    # --------------------------------------------------------------------- #

    def test_fit_calculates_mean_and_std(self) -> None:
        """fit() should store the mean and standard deviation of requested columns."""
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0, 40.0]})
        scaler = ZScoreScaler()
        scaler.fit(df, columns=["price"])

        assert hasattr(scaler, "_means")
        assert hasattr(scaler, "_stds")
        assert scaler._means is not None
        assert scaler._stds is not None
        assert scaler._means["price"] == pytest.approx(25.0)
        assert scaler._stds["price"] == pytest.approx(np.std([10.0, 20.0, 30.0, 40.0], ddof=0))

    def test_fit_multiple_columns(self) -> None:
        """fit() should handle multiple columns independently."""
        df = pd.DataFrame({"price": [1.0, 2.0, 3.0], "volume": [100.0, 200.0, 300.0]})
        scaler = ZScoreScaler()
        scaler.fit(df, columns=["price", "volume"])

        assert "price" in scaler._means
        assert "volume" in scaler._means
        assert scaler._means["price"] == pytest.approx(2.0)
        assert scaler._means["volume"] == pytest.approx(200.0)

    # --------------------------------------------------------------------- #
    #  transform
    # --------------------------------------------------------------------- #

    def test_transform_applies_stored_stats(self) -> None:
        """transform() should use previously stored mean/std and return a new DataFrame."""
        fit_df = pd.DataFrame({"price": [10.0, 20.0, 30.0, 40.0, 50.0]})
        scaler = ZScoreScaler()
        scaler.fit(fit_df, columns=["price"])

        transform_df = pd.DataFrame({"price": [15.0, 25.0]})
        result = scaler.transform(transform_df, columns=["price"])

        mean = scaler._means["price"]
        std = scaler._stds["price"]
        assert result.loc[0, "price"] == pytest.approx((15.0 - mean) / std)
        assert result.loc[1, "price"] == pytest.approx((25.0 - mean) / std)

    def test_transform_returns_new_dataframe(self) -> None:
        """The input DataFrame must not be mutated."""
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0]})
        scaler = ZScoreScaler()
        scaler.fit(df, columns=["price"])

        original = df.copy()
        scaler.transform(df, columns=["price"])
        pd.testing.assert_frame_equal(df, original)

    def test_transform_before_fit_raises(self) -> None:
        """Calling transform() before fit() must raise a ValueError."""
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0]})
        scaler = ZScoreScaler()

        with pytest.raises(ValueError, match="Scaler has not been fitted"):
            scaler.transform(df, columns=["price"])

    def test_transform_preserves_unscaled_columns(self) -> None:
        """Columns not requested for scaling should remain untouched."""
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0], "volume": [100, 200, 300]})
        scaler = ZScoreScaler()
        scaler.fit(df, columns=["price"])

        new_df = pd.DataFrame({"price": [15.0, 25.0], "volume": [150, 250]})
        result = scaler.transform(new_df, columns=["price"])

        assert list(result["volume"]) == [150, 250]

    def test_transform_handles_zero_std(self) -> None:
        """If std is 0, scaling should default to 0 to avoid NaN propagation."""
        df = pd.DataFrame({"price": [5.0, 5.0, 5.0]})
        scaler = ZScoreScaler()
        scaler.fit(df, columns=["price"])

        new_df = pd.DataFrame({"price": [5.0, 6.0]})
        result = scaler.transform(new_df, columns=["price"])

        # When std is 0, the scaled value should be 0 (or at least not NaN)
        assert not result["price"].isna().any()
        assert result.loc[0, "price"] == pytest.approx(0.0)
        assert result.loc[1, "price"] == pytest.approx(0.0)

    # --------------------------------------------------------------------- #
    #  fit_transform
    # --------------------------------------------------------------------- #

    def test_fit_transform_combines_both_steps(self) -> None:
        """fit_transform() should fit on the data then transform it."""
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0, 40.0]})
        scaler = ZScoreScaler()
        result = scaler.fit_transform(df, columns=["price"])

        mean = scaler._means["price"]
        std = scaler._stds["price"]
        expected = (df["price"] - mean) / std
        pd.testing.assert_series_equal(result["price"], expected, check_names=False)

    def test_fit_transform_returns_new_dataframe(self) -> None:
        """The original DataFrame must not be mutated by fit_transform."""
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0]})
        scaler = ZScoreScaler()
        original = df.copy()
        scaler.fit_transform(df, columns=["price"])
        pd.testing.assert_frame_equal(df, original)
