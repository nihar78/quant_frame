"""Tests for the SQLAlchemy-based time-series repository."""

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from quant_frame.core.models import TimeSeriesObservation
from quant_frame.repository.postgres_repo import (
    ObservationModel,
    SQLAlchemyRepository,
)


class TestSQLAlchemyRepository:
    """Test suite for :class:`SQLAlchemyRepository`."""

    @pytest.fixture
    def engine(self):
        """Yield an in-memory SQLite engine and dispose of it after each test."""
        engine = create_engine("sqlite:///:memory:")
        yield engine
        engine.dispose()

    @pytest.fixture
    def repo(self, engine):
        """Yield a fresh :class:`SQLAlchemyRepository` backed by the engine."""
        return SQLAlchemyRepository(engine)

    @pytest.fixture
    def session(self, engine):
        """Yield a database session for direct assertions."""
        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as session:
            yield session

    def test_instantiation_with_in_memory_engine(self, repo):
        """A repository must instantiate cleanly with an in-memory SQLite engine."""
        assert repo is not None
        assert isinstance(repo, SQLAlchemyRepository)

    def test_save_writes_observations(self, repo, session):
        """Calling ``save()`` must persist the provided observations."""
        ts = dt.datetime(2024, 1, 15, 9, 30, tzinfo=dt.timezone.utc)
        observations = [
            TimeSeriesObservation(
                timestamp=ts,
                asset_id="AAPL",
                features={"open": 150.0, "close": 155.5},
            ),
            TimeSeriesObservation(
                timestamp=ts,
                asset_id="TSLA",
                features={"open": 200.0, "high": 205.0},
            ),
        ]

        repo.save(observations)

        rows = session.query(ObservationModel).order_by(ObservationModel.asset_id).all()
        assert len(rows) == 2

        # SQLite DateTime does not store timezone info; compare naive
        assert rows[0].timestamp == ts.replace(tzinfo=None)
        assert rows[0].asset_id == "AAPL"
        assert rows[0].features == {"open": 150.0, "close": 155.5}

        assert rows[1].timestamp == ts.replace(tzinfo=None)
        assert rows[1].asset_id == "TSLA"
        assert rows[1].features == {"open": 200.0, "high": 205.0}

    def test_save_upserts_existing_row(self, repo, session):
        """Calling ``save()`` with the same key but different features must upsert."""
        ts = dt.datetime(2024, 1, 15, 9, 30, tzinfo=dt.timezone.utc)
        first = [
            TimeSeriesObservation(
                timestamp=ts,
                asset_id="AAPL",
                features={"close": 150.0},
            ),
        ]
        second = [
            TimeSeriesObservation(
                timestamp=ts,
                asset_id="AAPL",
                features={"close": 155.5, "volume": 1_000_000.0},
            ),
        ]

        repo.save(first)
        repo.save(second)

        rows = session.query(ObservationModel).all()
        assert len(rows) == 1
        assert rows[0].features == {"close": 155.5, "volume": 1_000_000.0}
