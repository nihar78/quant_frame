"""Machine-learning strategy implementations."""

from quant_frame.core.model_strategy import ModelRegistry

from .xgboost_strategy import XGBoostStrategy

__all__ = ["XGBoostStrategy"]

ModelRegistry.register("xgboost", XGBoostStrategy)
