"""SQLAlchemy-based repository for persisting time-series observations."""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Index,
    String,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from quant_frame.core.interfaces import BaseRepository
from quant_frame.core.models import TimeSeriesObservation


class _Base(DeclarativeBase):
    """SQLAlchemy declarative base for ORM models."""


class ObservationModel(_Base):
    """ORM model representing a single time-series observation.

    The combination of *timestamp* and *asset_id* is guaranteed unique via a
    composite unique constraint.  Features are stored as a JSON blob (JSONB on
    PostgreSQL) so that arbitrary feature schemas can evolve without DDL
    migrations.

    Attributes:
        id: Surrogate primary key.
        timestamp: Observation point-in-time (timezone-aware datetime).
        asset_id: Identifier for the observed asset (e.g. ticker, ISIN).
        features: Arbitrary mapping of feature names to float values.
    """

    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(
        primary_key=True, autoincrement=True
    )
    timestamp: Mapped[dt.datetime] = mapped_column(
        DateTime, nullable=False
    )
    asset_id: Mapped[str] = mapped_column(String(255), nullable=False)
    features: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)

    __table_args__ = (
        UniqueConstraint("timestamp", "asset_id", name="uq_observation"),
        Index("ix_observations_asset_id", "asset_id"),
    )


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_conn: Any, _record: Any) -> None:
    """Enable foreign-key support on SQLite connections."""
    import sqlite3

    if isinstance(dbapi_conn, sqlite3.Connection):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")


class SQLAlchemyRepository(BaseRepository):
    """Concrete repository backed by SQLAlchemy with UPSERT semantics.

    The repository is initialised with an SQLAlchemy :class:`Engine`.  On
    :meth:`save`, observations are bulk-inserted.  When a row already exists for
    the same *(timestamp, asset_id)* pair the *features* column is updated in
    place rather than creating a duplicate or raising an integrity error.  This
    mirrors Postgres JSONB UPSERT behaviour and also works with SQLite.

    Example:
        >>> from sqlalchemy import create_engine
        >>> engine = create_engine("sqlite:///:memory:")
        >>> repo = SQLAlchemyRepository(engine)
        >>> repo.save([
        ...     TimeSeriesObservation(
        ...         timestamp=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
        ...         asset_id="AAPL",
        ...         features={"close": 150.0},
        ...     ),
        ... ])

    Args:
        engine: A configured SQLAlchemy :class:`Engine` instance.
    """

    def __init__(self, engine: Engine) -> None:
        """Initialise the repository and create tables if they do not exist."""
        self._engine: Engine = engine
        _Base.metadata.create_all(self._engine)
        self._Session: sessionmaker[Session] = sessionmaker(bind=self._engine)

    def save(self, observations: list[TimeSeriesObservation]) -> None:
        """Persist the given observations using an upsert strategy.

        Each observation is converted into a dictionary suitable for the
        :class:`ObservationModel`.  The batch is inserted with ``ON CONFLICT
        DO UPDATE``, meaning that if an observation with the same *timestamp*
        and *asset_id* already exists, its *features* field will be overwritten
        with the new values.

        Args:
            observations: A list of :class:`TimeSeriesObservation` objects to
                be persisted.

        Returns:
            ``None``.  The method operates via side-effects on the database.
        """
        if not observations:
            return

        values = [
            {
                "timestamp": obs.timestamp,
                "asset_id": obs.asset_id,
                "features": obs.features,
            }
            for obs in observations
        ]

        stmt = sqlite_insert(ObservationModel).values(values)
        update_stmt = stmt.on_conflict_do_update(
            index_elements=["timestamp", "asset_id"],
            set_={"features": stmt.excluded.features},  # type: ignore[attr-defined]
        )

        with self._Session() as session:
            session.execute(update_stmt)
            session.commit()
