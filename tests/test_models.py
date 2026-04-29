"""Tests for the core data models."""

import datetime as dt
from typing import Any

import pytest
from pydantic import ValidationError

from quant_frame import TimeSeriesObservation


class TestTimeSeriesObservation:
    """Test suite for :class:`TimeSeriesObservation`."""

    def test_valid_instantiation(self) -> None:
        """A valid observation should instantiate without errors."""
        ts = dt.datetime(2024, 1, 15, 9, 30, tzinfo=dt.timezone.utc)
        obs = TimeSeriesObservation(
            timestamp=ts,
            asset_id="AAPL",
            features={"open": 150.0, "close": 155.5},
        )
        assert obs.timestamp == ts
        assert obs.asset_id == "AAPL"
        assert obs.features == {"open": 150.0, "close": 155.5}

    @pytest.mark.parametrize(
        "invalid_features",
        [
            {"open": "not_a_float"},
            {"volume": 100},      # int is not a strict float
            {"flag": True},       # bool is not a strict float
            {"mixed": 1.0, "bad": "oops"},
        ],
    )
    def test_features_must_be_strict_floats(self, invalid_features: dict[str, Any]) -> None:
        """Validation must reject non-float values inside *features*."""
        with pytest.raises(ValidationError):
            TimeSeriesObservation(
                timestamp=dt.datetime.now(tz=dt.timezone.utc),
                asset_id="TEST",
                features=invalid_features,
            )

    def test_empty_features_is_allowed(self) -> None:
        """An empty feature map should be accepted."""
        obs = TimeSeriesObservation(
            timestamp=dt.datetime.now(tz=dt.timezone.utc),
            asset_id="EMPTY",
            features={},
        )
        assert obs.features == {}
