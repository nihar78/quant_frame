"""Core data models for the quant_frame library."""

from __future__ import annotations

import datetime as dt
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TimeSeriesObservation(BaseModel):
    """Universal Data Transfer Object for time-series data.

    This model represents a single observation in a time series, capturing a
    timestamp, the associated asset identifier, and an arbitrary collection of
    numerical features.  It is intended to be the primary interchange format
    across ingestion, transformation, and modelling layers.

    Attributes:
        timestamp: Exact point in time for the observation (timezone-aware
            ``datetime``).
        asset_id: Unique identifier for the asset (e.g. ticker, ISIN, or
            internal ID).
        features: Mapping of feature names to their **strict** ``float`` values.
            Pydantic will reject integers, booleans, strings, ``None``, etc.

    Example:
        >>> from datetime import datetime, timezone
        >>> obs = TimeSeriesObservation(
        ...     timestamp=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
        ...     asset_id="AAPL",
        ...     features={"open": 150.0, "close": 155.5, "volume": 1_000_000.0},
        ... )
        >>> obs.asset_id
        'AAPL'
    """

    timestamp: dt.datetime = Field(
        ..., description="Point-in-time for the observation."
    )
    asset_id: str = Field(
        ..., description="Unique identifier for the observed asset."
    )
    features: dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of feature names to strict float values.",
    )

    model_config: ClassVar[ConfigDict] = {
        "frozen": True,
        "str_strip_whitespace": True,
        "extra": "forbid",
    }

    @field_validator("features", mode="before")
    @classmethod
    def _validate_strict_floats(
        cls, value: dict[str, Any]
    ) -> dict[str, Any]:
        """Ensure every feature value is an uncoerced ``float``.

        Pydantic's default ``float`` will happily accept ``int`` and ``bool``.
        We guard against that by checking ``type(val) is float`` before
        handing the data off to Pydantic.

        Args:
            value: Raw *features* dictionary supplied by the caller.

        Returns:
            The same dictionary if every value is a strict ``float``.

        Raises:
            ValueError: If any value is not an instance of ``float``.
        """
        for key, val in value.items():
            if type(val) is not float:
                raise ValueError(
                    f"Feature '{key}' must be a strict float, got {type(val).__name__}"
                )
        return value
