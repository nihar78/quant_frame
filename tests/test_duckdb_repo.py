"""Tests for the DuckDB-based time-series repository."""

import datetime as dt

import pytest

from quant_frame.core.models import TimeSeriesObservation
from quant_frame.repository.duckdb_repo import DuckDBRepository


class TestDuckDBRepository:
    """Test suite for :class:`DuckDBRepository`."""

    @pytest.fixture
    def repo(self):
        """Yield a fresh in-memory :class:`DuckDBRepository`."""
        repository = DuckDBRepository(":memory:")
        yield repository
        repository._conn.close()

    def test_instantiation(self, repo):
        """A repository must instantiate cleanly with an in-memory database."""
        assert repo is not None
        assert isinstance(repo, DuckDBRepository)

    def test_save_writes_observations(self, repo):
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

        rows = repo._conn.execute(
            "SELECT timestamp, asset_id, features FROM observations ORDER BY asset_id"
        ).fetchall()
        assert len(rows) == 2

        # DuckDB TIMESTAMP does not store timezone info; compare naive
        assert rows[0][0] == ts.replace(tzinfo=None)
        assert rows[0][1] == "AAPL"
        assert rows[0][2] == '{"open":150.0,"close":155.5}'

        assert rows[1][0] == ts.replace(tzinfo=None)
        assert rows[1][1] == "TSLA"
        assert rows[1][2] == '{"open":200.0,"high":205.0}'

    def test_save_upserts_existing_row(self, repo):
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

        rows = repo._conn.execute(
            "SELECT features FROM observations"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == '{"close":155.5,"volume":1000000.0}'

    def test_save_empty_list_is_no_op(self, repo):
        """Passing an empty list must not raise or modify the database."""
        repo.save([])
        count = repo._conn.execute(
            "SELECT COUNT(*) FROM observations"
        ).fetchone()[0]
        assert count == 0
