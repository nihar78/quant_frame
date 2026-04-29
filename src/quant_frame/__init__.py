"""quant_frame - Core quantitative analysis framework."""

from .core.interfaces import BaseProvider, BaseRepository
from .core.models import TimeSeriesObservation

__all__ = [
    "BaseProvider",
    "BaseRepository",
    "hello_frame",
    "TimeSeriesObservation",
]


def hello_frame() -> bool:
    return True
