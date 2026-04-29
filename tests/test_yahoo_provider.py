"""Tests for the live Yahoo Finance ingestion provider."""

from __future__ import annotations

import datetime as dt
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_frame.core.models import TimeSeriesObservation
from quant_frame.adapters.yahoo_provider import YahooFinanceProvider


class TestYahooFinanceProvider:
    """Test suite for :class:`YahooFinanceProvider`."""

    @pytest.fixture
    def mock_history_df(self) -> pd.DataFrame:
        """Return a dummy DataFrame mimicking yfinance Ticker.history() output."""
        return pd.DataFrame(
            {
                "Open": [150.0, 151.0, 152.0],
                "High": [152.0, 153.0, 154.0],
                "Low": [149.0, 150.0, 151.0],
                "Close": [151.0, 152.0, 153.0],
                "Volume": [1_000_000.0, 1_100_000.0, 1_200_000.0],
            },
            index=pd.to_datetime(
                [
                    "2024-01-15",
                    "2024-01-16",
                    "2024-01-17",
                ]
            ),
        )

    def test_instantiation_with_defaults(self) -> None:
        """A YahooFinanceProvider must instantiate with a ticker and default period."""
        provider = YahooFinanceProvider(ticker="AAPL")
        assert provider is not None
        assert isinstance(provider, YahooFinanceProvider)

    def test_instantiation_with_custom_period(self) -> None:
        """A YahooFinanceProvider must accept an optional period parameter."""
        provider = YahooFinanceProvider(ticker="TSLA", period="6mo")
        assert provider is not None
        assert isinstance(provider, YahooFinanceProvider)

    def test_extract_returns_observations(self, mock_history_df: pd.DataFrame) -> None:
        """``extract()`` must return a list of TimeSeriesObservation with correct data."""
        mock_ticker = MagicMock()
        mock_ticker.history = MagicMock(return_value=mock_history_df)

        with patch("yfinance.Ticker", return_value=mock_ticker):
            provider = YahooFinanceProvider(ticker="AAPL")
            observations = provider.extract()

        assert isinstance(observations, list)
        assert len(observations) == 3
        assert all(isinstance(o, TimeSeriesObservation) for o in observations)

        first = observations[0]
        assert first.asset_id == "AAPL"
        assert first.timestamp == pd.Timestamp("2024-01-15").to_pydatetime()
        assert first.features == {
            "Open": 150.0,
            "High": 152.0,
            "Low": 149.0,
            "Close": 151.0,
            "Volume": 1_000_000.0,
        }

        second = observations[1]
        assert second.asset_id == "AAPL"
        assert second.timestamp == pd.Timestamp("2024-01-16").to_pydatetime()

    def test_extract_strict_floats(self, mock_history_df: pd.DataFrame) -> None:
        """Feature values must be strict floats, not ints."""
        # Force integer-like values in the DataFrame
        int_df = mock_history_df.copy()
        int_df["Volume"] = [1_000_000, 1_100_000, 1_200_000]

        mock_ticker = MagicMock()
        mock_ticker.history = MagicMock(return_value=int_df)

        with patch("yfinance.Ticker", return_value=mock_ticker):
            provider = YahooFinanceProvider(ticker="GOOG")
            observations = provider.extract()

        assert len(observations) == 3
        for obs in observations:
            assert all(type(v) is float for v in obs.features.values())

    def test_empty_dataframe_returns_empty_list(self) -> None:
        """An empty DataFrame from yfinance must result in an empty list."""
        empty_df = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        ).astype(float)
        empty_df.index = pd.DatetimeIndex([])

        mock_ticker = MagicMock()
        mock_ticker.history = MagicMock(return_value=empty_df)

        with patch("yfinance.Ticker", return_value=mock_ticker):
            provider = YahooFinanceProvider(ticker="EMPTY")
            observations = provider.extract()

        assert observations == []
