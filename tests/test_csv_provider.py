"""Tests for the generic CSV ingestion provider."""

from __future__ import annotations

import datetime as dt
import io
from typing import Any

import pytest
import pandas as pd

from quant_frame.core.models import TimeSeriesObservation
from quant_frame.adapters.csv_provider import CSVProvider


class TestCSVProvider:
    """Test suite for :class:`CSVProvider`."""

    @pytest.fixture
    def sample_csv(self) -> str:
        """Return a basic CSV string with a timestamp and two features."""
        return (
            "date,open,close,volume\n"
            "2024-01-15,150.0,155.5,1000.0\n"
            "2024-01-16,151.0,154.0,900.0\n"
        )

    def test_instantiation(self, sample_csv: str) -> None:
        """A CSVProvider must instantiate with a file path, asset_id and timestamp_col."""
        buf = io.StringIO(sample_csv)
        provider = CSVProvider(
            source=buf,
            asset_id="AAPL",
            timestamp_col="date",
        )
        assert provider is not None
        assert isinstance(provider, CSVProvider)

    def test_extract_maps_timestamp_and_features(self, sample_csv: str) -> None:
        """``extract()`` must return a list of TimeSeriesObservation with correct data."""
        buf = io.StringIO(sample_csv)
        provider = CSVProvider(
            source=buf,
            asset_id="AAPL",
            timestamp_col="date",
        )
        observations = provider.extract()

        assert isinstance(observations, list)
        assert len(observations) == 2
        assert all(isinstance(o, TimeSeriesObservation) for o in observations)

        first = observations[0]
        assert first.asset_id == "AAPL"
        assert first.timestamp == pd.Timestamp("2024-01-15").to_pydatetime()
        assert first.features == {"open": 150.0, "close": 155.5, "volume": 1000.0}

        second = observations[1]
        assert second.asset_id == "AAPL"
        assert second.timestamp == pd.Timestamp("2024-01-16").to_pydatetime()
        assert second.features == {"open": 151.0, "close": 154.0, "volume": 900.0}

    def test_extract_multiple_numeric_dtypes(self) -> None:
        """Columns with integer, float and booleans as numbers must all be cast to float."""
        csv_text = (
            "datetime,flag,count,price\n"
            "2024-03-01,1,100,10.5\n"
            "2024-03-02,0,200,11.5\n"
        )
        buf = io.StringIO(csv_text)
        provider = CSVProvider(
            source=buf,
            asset_id="TSLA",
            timestamp_col="datetime",
        )
        observations = provider.extract()

        assert len(observations) == 2
        # Pandas will read 1/0 as ints; provider must explicitly cast to float
        assert observations[0].features == {
            "flag": 1.0,
            "count": 100.0,
            "price": 10.5,
        }
        assert all(type(v) is float for v in observations[0].features.values())

    def test_extract_ignores_non_numeric_string_columns(self) -> None:
        """String columns (other than timestamp) must be safely dropped."""
        csv_text = (
            "date,label,ticker,close\n"
            "2024-04-01,buy,AAPL,150.0\n"
            "2024-04-02,sell,AAPL,151.0\n"
        )
        buf = io.StringIO(csv_text)
        provider = CSVProvider(
            source=buf,
            asset_id="AAPL",
            timestamp_col="date",
        )
        observations = provider.extract()

        assert len(observations) == 2
        assert observations[0].features == {"close": 150.0}
        assert observations[1].features == {"close": 151.0}

    def test_extract_with_null_values(self) -> None:
        """Rows that have nullish numeric values must drop that key; no KeyError."""
        csv_text = (
            "date,open,close\n"
            "2024-05-01,10.0,\n"
            "2024-05-02,,12.0\n"
        )
        buf = io.StringIO(csv_text)
        provider = CSVProvider(
            source=buf,
            asset_id="X",
            timestamp_col="date",
        )
        observations = provider.extract()

        assert len(observations) == 2
        assert observations[0].features == {"open": 10.0}
        assert observations[1].features == {"close": 12.0}

    def test_empty_csv_returns_empty_list(self) -> None:
        """A CSV with no data rows must return an empty list."""
        csv_text = "date,open,close\n"
        buf = io.StringIO(csv_text)
        provider = CSVProvider(
            source=buf,
            asset_id="EMPTY",
            timestamp_col="date",
        )
        observations = provider.extract()
        assert observations == []

    def test_rows_with_only_non_numeric_columns(self) -> None:
        """If no numeric columns remain after dropping strings, features is empty."""
        csv_text = (
            "date,note\n"
            "2024-06-01,hello\n"
        )
        buf = io.StringIO(csv_text)
        provider = CSVProvider(
            source=buf,
            asset_id="NOTE",
            timestamp_col="date",
        )
        observations = provider.extract()

        assert len(observations) == 1
        assert observations[0].features == {}
