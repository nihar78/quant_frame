"""quant_frame - Core quantitative analysis framework."""

from .core.config import PipelineConfig
from .core.interfaces import BaseProvider, BaseRepository
from .core.model_strategy import BaseModelStrategy, ModelRegistry
from .core.models import TimeSeriesObservation

__all__ = [
    "BaseModelStrategy",
    "BaseProvider",
    "BaseRepository",
    "hello_frame",
    "ModelRegistry",
    "PipelineConfig",
    "TimeSeriesObservation",
]


def hello_frame() -> bool:
    return True
