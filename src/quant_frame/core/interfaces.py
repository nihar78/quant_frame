"""Abstract base classes defining the core framework contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import TimeSeriesObservation


class BaseProvider(ABC):
    """Abstract base class for all data providers.

    A *provider* is responsible for extracting raw observations from an external
    source (API, file, database, stream, etc.) and returning them as a list of
    :class:`TimeSeriesObservation` instances.  Every concrete provider must
    implement the :meth:`extract` method.

    Example:
        >>> class YahooFinanceProvider(BaseProvider):
        ...     def extract(self) -> list[TimeSeriesObservation]:
        ...         # ... fetch data ...
        ...         return []
    """

    @abstractmethod
    def extract(self) -> list[TimeSeriesObservation]:
        """Extract observations from the external data source.

        Returns:
            A list of :class:`TimeSeriesObservation` objects representing the
            data retrieved from the source.
        """


class BaseRepository(ABC):
    """Abstract base class for all data repositories.

    A *repository* is responsible for persisting a collection of
    :class:`TimeSeriesObservation` objects to a storage backend (database,
    file system, cloud bucket, etc.).  Every concrete repository must
    implement the :meth:`save` method.

    Example:
        >>> class PostgresRepository(BaseRepository):
        ...     def save(self, observations: list[TimeSeriesObservation]) -> None:
        ...         # ... insert into DB ...
        ...         pass
    """

    @abstractmethod
    def save(self, observations: list[TimeSeriesObservation]) -> None:
        """Persist the given observations to the storage backend.

        Args:
            observations: A list of :class:`TimeSeriesObservation` objects to
                be saved.
        """
