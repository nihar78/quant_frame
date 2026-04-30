"""quant_frame - Core quantitative analysis framework."""

from .adapters.csv_provider import CSVProvider
from .adapters.yahoo_provider import YahooFinanceProvider
from .analytics.aligner import TimeSeriesAligner
from .analytics.scalers import ZScoreScaler
from .analytics.transformer import TimeSeriesTransformer
from .core.config import PipelineConfig
from .core.interfaces import BaseProvider, BaseRepository
from .core.metrics import BaseMetrics
from .core.model_strategy import BaseModelStrategy, ModelRegistry
from .core.models import TimeSeriesObservation
from .core.validators import ThresholdValidator
from .performance import FinancialMetrics, plot_financial_tearsheet, VectorizedSimulator
from .repository.postgres_repo import SQLAlchemyRepository
from .strategies.xgboost_strategy import XGBoostStrategy
from .validation.evaluator import WalkForwardEvaluator
from .validation.splitter import WalkForwardSplitter

__all__ = [
    "BaseMetrics",
    "BaseModelStrategy",
    "BaseProvider",
    "BaseRepository",
    "CSVProvider",
    "FinancialMetrics",
    "hello_frame",
    "ModelRegistry",
    "PipelineConfig",
    "plot_financial_tearsheet",
    "SQLAlchemyRepository",
    "ThresholdValidator",
    "TimeSeriesAligner",
    "TimeSeriesTransformer",
    "TimeSeriesObservation",
    "WalkForwardEvaluator",
    "VectorizedSimulator",
    "WalkForwardSplitter",
    "XGBoostStrategy",
    "ZScoreScaler",
]


def hello_frame() -> bool:
    return True