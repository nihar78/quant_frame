"""Live Yahoo Finance ingestion adapter for the quant_frame library."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import yfinance

from quant_frame.core.interfaces import BaseProvider
from quant_frame.core.models import TimeSeriesObservation


class YahooFinanceProvider(BaseProvider):
    """Extract time-series observations from Yahoo Finance via *yfinance*.

    The provider instantiates a ``yfinance.Ticker`` object for the given
    ticker symbol, calls ``Ticker.history(period=...)`` to retrieve OHLCV
    data, and maps each row to a :class:`TimeSeriesObservation` instance.

    All numeric columns returned by *yfinance* (typically ``Open``,
    ``High``, ``Low``, ``Close``, ``Volume``, and optionally ``Dividends``
    and ``Stock Splits``) are packed into the *features* mapping as strict
    ``float`` values.  The provider skips rows that contain no valid
    numeric values.

    Example:
        >>> provider = YahooFinanceProvider(ticker="AAPL", period="3mo")
        >>> observations = provider.extract()
        >>> len(observations) > 0
        True

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. ``"AAPL"``, ``"^GSPC"``).
        period: Lookback period passed to ``yfinance.Ticker.history``.
            Defaults to ``"1y"``.
    """

    def __init__(self, *, ticker: str, period: str = "1y") -> None:
        """Initialise the Yahoo Finance provider.

        Args:
            ticker: The ticker symbol to fetch data for.
            period: The lookback period for historical data.
        """
        self._ticker: str = ticker
        self._period: str = period

    @property
    def ticker(self) -> str:
        """The ticker symbol this provider queries."""
        return self._ticker

    @property
    def period(self) -> str:
        """The configured lookback period."""
        return self._period

    def extract(self) -> list[TimeSeriesObservation]:
        """Fetch historical data from Yahoo Finance and return observations.

        The method performs the following steps:

        1. Create a ``yfinance.Ticker`` for the configured ticker.
        2. Call ``history(period=self._period)`` to obtain a DataFrame.
        3. Ensure the DataFrame index is timezone-naive and cast to plain
           ``datetime`` objects.
        4. Iterate over rows and build :class:`TimeSeriesObservation`
           instances, converting every feature value to a strict ``float``.

        Returns:
            A list of :class:`TimeSeriesObservation` objects, one per row in
            the returned DataFrame.  If the DataFrame is empty, an empty list
            is returned.
        """
        ticker = yfinance.Ticker(self._ticker)
        df = ticker.history(period=self._period)

        if df.empty:
            return []

        # Ensure the index is DatetimeIndex and convert to timezone-naive
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index

        observations: list[TimeSeriesObservation] = []
        for idx, row in df.iterrows():
            raw_ts: Any = idx
            timestamp: dt.datetime
            if isinstance(raw_ts, pd.Timestamp):
                timestamp = raw_ts.to_pydatetime()
            elif isinstance(raw_ts, dt.datetime):
                timestamp = raw_ts
            else:
                timestamp = pd.to_datetime(raw_ts).to_pydatetime()

            features: dict[str, float] = {}
            for col in df.columns:
                val = row[col]
                if pd.notna(val):
                    features[str(col)] = float(val)

            observations.append(
                TimeSeriesObservation(
                    timestamp=timestamp,
                    asset_id=self._ticker,
                    features=features,
                )
            )

        return observations
