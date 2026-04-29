"""quant_frame - Core quantitative analysis framework."""

from .core.models import TimeSeriesObservation

__all__ = [
    "hello_frame",
    "TimeSeriesObservation",
]


def hello_frame() -> bool:
    return True
