"""Repository implementations for the quant_frame library."""

from .duckdb_repo import DuckDBRepository
from .postgres_repo import SQLAlchemyRepository

__all__ = ["DuckDBRepository", "SQLAlchemyRepository"]
