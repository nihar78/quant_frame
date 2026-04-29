"""Generic CSV ingestion adapter for the quant_frame library."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from quant_frame.core.interfaces import BaseProvider
from quant_frame.core.models import TimeSeriesObservation


class CSVProvider(BaseProvider):
    """Extract time-series observations from a CSV data source.

    The provider reads a CSV file (or any file-like object) using *pandas*,
    converts the designated timestamp column to ``datetime`` objects, and
    packs every remaining **numeric** column into the *features* mapping of
    each :class:`TimeSeriesObservation`.

    Non-numeric columns (other than the timestamp) are silently dropped so
    that string annotations or categorical labels do not cause validation
    errors downstream.

    Example:
        >>> import io
        >>> csv = io.StringIO("date,open,close\\n2024-01-15,150.0,155.5\\n")
        >>> provider = CSVProvider(source=csv, asset_id="AAPL", timestamp_col="date")
        >>> provider.extract()
        [TimeSeriesObservation(timestamp=..., asset_id='AAPL', features={'open': 150.0, 'close': 155.5})]

    Args:
        source: Path to a CSV file **or** any file-like object that
            ``pd.read_csv`` can consume (e.g. ``io.StringIO``).
        asset_id: The asset identifier that will be assigned to every
            extracted observation.
        timestamp_col: Name of the column that contains the observation
            timestamps.
    """

    def __init__(
        self,
        *,
        source: str | Any,
        asset_id: str,
        timestamp_col: str,
    ) -> None:
        """Initialise the CSV provider.

        Args:
            source: Path or file-like object pointing to the CSV data.
            asset_id: Unique identifier for the observed asset.
            timestamp_col: Column header that holds the timestamps.
        """
        self._source: str | Any = source
        self._asset_id: str = asset_id
        self._timestamp_col: str = timestamp_col

    def extract(self) -> list[TimeSeriesObservation]:
        """Read the CSV source and return a list of observations.

        The method performs the following steps:

        1. Read the CSV via ``pandas.read_csv``.
        2. Convert *timestamp_col* to ``datetime`` using ``pd.to_datetime``.
        3. Drop every column whose dtype is not numeric (except *timestamp_col*).
        4. Iterate over rows and build :class:`TimeSeriesObservation` instances,
           returning only non-null feature values as strict ``float`` values.

        Returns:
            A list of :class:`TimeSeriesObservation` objects, one per row in the
            CSV.  If the CSV contains no data rows, an empty list is returned.
        """
        df = pd.read_csv(self._source)

        # No data rows → empty DataFrame after read_csv still has columns
        if df.empty:
            return []

        # Ensure the timestamp column exists
        if self._timestamp_col not in df.columns:
            raise KeyError(
                f"Timestamp column '{self._timestamp_col}' not found in CSV columns: {list(df.columns)}"
            )

        # Coerce to datetime
        df[self._timestamp_col] = pd.to_datetime(df[self._timestamp_col])

        # Identify numeric columns (excluding the timestamp)
        numeric_cols = [
            col
            for col in df.columns
            if col != self._timestamp_col and pd.api.types.is_numeric_dtype(df[col])
        ]

        observations: list[TimeSeriesObservation] = []
        for _, row in df.iterrows():
            raw_ts = row[self._timestamp_col]
            # Ensure we have a plain datetime (not Timestamp / NaT)
            timestamp: dt.datetime
            if pd.isna(raw_ts):
                continue  # Skip rows with unparseable timestamp
            if isinstance(raw_ts, pd.Timestamp):
                timestamp = raw_ts.to_pydatetime()
            else:
                timestamp = raw_ts  # type: ignore[assignment]

            features: dict[str, float] = {}
            for col in numeric_cols:
                val = row[col]
                if pd.notna(val):
                    features[col] = float(val)

            observations.append(
                TimeSeriesObservation(
                    timestamp=timestamp,
                    asset_id=self._asset_id,
                    features=features,
                )
            )

        return observations
