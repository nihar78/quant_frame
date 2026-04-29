"""Tests for the data quality validators."""

import datetime as dt
from typing import Any

import pytest

from quant_frame.core.models import TimeSeriesObservation
from quant_frame.core.validators import ThresholdValidator


class TestThresholdValidator:
    """Test suite for :class:`ThresholdValidator`."""

    def test_instantiation_with_thresholds(self) -> None:
        """A validator should be instantiable with a dictionary of thresholds."""
        thresholds: dict[str, dict[str, float]] = {
            "heart_rate": {"min": 30.0, "max": 220.0},
            "price": {"min": 0.0},
        }
        validator = ThresholdValidator(thresholds=thresholds)
        assert validator.thresholds == thresholds

    def test_filter_anomalies_removes_out_of_bounds(self) -> None:
        """Observations with features outside defined bounds should be removed."""
        thresholds = {
            "heart_rate": {"min": 30.0, "max": 220.0},
        }
        validator = ThresholdValidator(thresholds=thresholds)

        ts = dt.datetime(2024, 1, 15, 9, 30, tzinfo=dt.timezone.utc)
        valid_obs = TimeSeriesObservation(
            timestamp=ts,
            asset_id="A",
            features={"heart_rate": 80.0},
        )
        too_low = TimeSeriesObservation(
            timestamp=ts,
            asset_id="B",
            features={"heart_rate": 20.0},
        )
        too_high = TimeSeriesObservation(
            timestamp=ts,
            asset_id="C",
            features={"heart_rate": 250.0},
        )

        observations = [valid_obs, too_low, too_high]
        result = validator.filter_anomalies(observations)

        assert len(result) == 1
        assert result[0] == valid_obs

    def test_filter_anomalies_allows_untracked_features(self) -> None:
        """Features not present in thresholds should pass through safely."""
        thresholds = {
            "heart_rate": {"min": 30.0, "max": 220.0},
        }
        validator = ThresholdValidator(thresholds=thresholds)

        ts = dt.datetime(2024, 1, 15, 9, 30, tzinfo=dt.timezone.utc)
        obs = TimeSeriesObservation(
            timestamp=ts,
            asset_id="A",
            features={
                "heart_rate": 80.0,
                "blood_pressure": 120.0,  # not in thresholds
            },
        )

        result = validator.filter_anomalies([obs])
        assert len(result) == 1
        assert result[0] == obs

    def test_filter_anomalies_with_min_only(self) -> None:
        """A threshold specifying only a ``min`` should still filter correctly."""
        thresholds = {
            "price": {"min": 0.0},
        }
        validator = ThresholdValidator(thresholds=thresholds)

        ts = dt.datetime(2024, 1, 15, 9, 30, tzinfo=dt.timezone.utc)
        valid = TimeSeriesObservation(
            timestamp=ts,
            asset_id="A",
            features={"price": 10.0},
        )
        invalid = TimeSeriesObservation(
            timestamp=ts,
            asset_id="B",
            features={"price": -5.0},
        )

        result = validator.filter_anomalies([valid, invalid])
        assert len(result) == 1
        assert result[0] == valid

    def test_filter_anomalies_with_max_only(self) -> None:
        """A threshold specifying only a ``max`` should still filter correctly."""
        thresholds = {
            "score": {"max": 100.0},
        }
        validator = ThresholdValidator(thresholds=thresholds)

        ts = dt.datetime(2024, 1, 15, 9, 30, tzinfo=dt.timezone.utc)
        valid = TimeSeriesObservation(
            timestamp=ts,
            asset_id="A",
            features={"score": 95.0},
        )
        invalid = TimeSeriesObservation(
            timestamp=ts,
            asset_id="B",
            features={"score": 105.0},
        )

        result = validator.filter_anomalies([valid, invalid])
        assert len(result) == 1
        assert result[0] == valid

    def test_filter_anomalies_empty_list(self) -> None:
        """Passing an empty list should return an empty list."""
        validator = ThresholdValidator(thresholds={"x": {"min": 0.0}})
        assert validator.filter_anomalies([]) == []

    def test_filter_anomalies_no_features(self) -> None:
        """An observation with no features should pass through when thresholds are defined."""
        validator = ThresholdValidator(thresholds={"heart_rate": {"min": 30.0}})

        ts = dt.datetime(2024, 1, 15, 9, 30, tzinfo=dt.timezone.utc)
        obs = TimeSeriesObservation(
            timestamp=ts,
            asset_id="A",
            features={},
        )

        result = validator.filter_anomalies([obs])
        assert len(result) == 1
        assert result[0] == obs
