"""quant_frame - Core quantitative analysis framework."""

from .adapters.csv_provider import CSVProvider
from .adapters.yahoo_provider import YahooFinanceProvider
from .analytics.aligner import TimeSeriesAligner
from .analytics.scalers import ZScoreScaler
from .analytics.transformer import TimeSeriesTransformer
from .core.config import PipelineConfig
from .core.interfaces import BaseProvider, BaseRepository
from .core.model_strategy import BaseModelStrategy, ModelRegistry
from .core.models import TimeSeriesObservation
from .core.validators import ThresholdValidator
from .repository.postgres_repo import SQLAlchemyRepository

__all__ = [
    "BaseModelStrategy",
    "BaseProvider",
    "BaseRepository",
    "CSVProvider",
    "YahooFinanceProvider",
    "hello_frame",
    "ModelRegistry",
    "PipelineConfig",
    "SQLAlchemyRepository",
    "ThresholdValidator",
    "TimeSeriesAligner",
    "TimeSeriesTransformer",
    "TimeSeriesObservation",
    "ZScoreScaler",
]


def hello_frame() -> bool:
    return True
