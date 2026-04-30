"""DuckDB-based repository for persisting time-series observations."""

from __future__ import annotations

import datetime as dt
from typing import Any

import duckdb
import pandas as pd

from quant_frame.core.interfaces import BaseRepository
from quant_frame.core.models import TimeSeriesObservation


class DuckDBRepository(BaseRepository):
    """Concrete repository backed by DuckDB with UPSERT semantics.

    The repository initialises a DuckDB connection.  On :meth:`save`, observations
    are bulk-inserted from a :class:`pandas.DataFrame`.  When a row already exists
    for the same *(timestamp, asset_id)* pair the *features* column is updated in
    place rather than creating a duplicate or raising an integrity error.

    Example:
        >>> repo = DuckDBRepository()
        >>> repo.save([
        ...     TimeSeriesObservation(
        ...         timestamp=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
        ...         asset_id="AAPL",
        ...         features={"close": 150.0},
        ...     ),
        ... ])

    Args:
        database: Path to the DuckDB database file, or ``":memory:"`` for an
            in-memory transient database (the default).
    """

    def __init__(self, database: str = ":memory:") -> None:
        """Initialise the repository and create tables if they do not exist."""
        self._conn: duckdb.DuckDBPyConnection = duckdb.connect(database)
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the *observations* table with a composite primary key."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                timestamp TIMESTAMP,
                asset_id VARCHAR,
                features JSON,
                PRIMARY KEY (timestamp, asset_id)
            )
            """
        )

    def save(self, observations: list[TimeSeriesObservation]) -> None:
        """Persist the given observations using an upsert strategy.

        Observations are converted into a :class:`pandas.DataFrame` and inserted
        via ``INSERT INTO ... SELECT ... ON CONFLICT (timestamp, asset_id)
        DO UPDATE SET features = excluded.features``.

        Args:
            observations: A list of :class:`TimeSeriesObservation` objects to
                be persisted.

        Returns:
            ``None``.  The method operates via side-effects on the database.
        """
        if not observations:
            return

        # DuckDB TIMESTAMP does not preserve timezone info; normalise to naive
        # (consistent with SQLAlchemy/SQLite behaviour in this project).
        values: list[dict[str, Any]] = [
            {
                "timestamp": (
                    obs.timestamp.replace(tzinfo=None)
                    if obs.timestamp.tzinfo
                    else obs.timestamp
                ),
                "asset_id": obs.asset_id,
                "features": obs.features,
            }
            for obs in observations
        ]

        df = pd.DataFrame(values)
        view_name = "_quant_frame_observations_temp"
        self._conn.register(view_name, df)

        try:
            self._conn.execute(
                f"""
                INSERT INTO observations
                SELECT * FROM {view_name}
                ON CONFLICT (timestamp, asset_id)
                DO UPDATE SET features = excluded.features
                """
            )
        finally:
            try:
                self._conn.execute(f"DROP VIEW IF EXISTS {view_name}")
            except Exception:  # pragma: no cover
                pass
