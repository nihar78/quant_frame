"""Data ingestion adapters for external sources."""

from .csv_provider import CSVProvider
from .yahoo_provider import YahooFinanceProvider

__all__ = ["CSVProvider", "YahooFinanceProvider"]
