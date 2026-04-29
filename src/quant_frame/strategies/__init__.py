"""Machine-learning strategy implementations."""

from quant_frame.core.model_strategy import ModelRegistry

from .hmm_strategy import GaussianHMMStrategy
from .xgboost_strategy import XGBoostStrategy

__all__ = ["GaussianHMMStrategy", "XGBoostStrategy"]

ModelRegistry.register("gaussian_hmm", GaussianHMMStrategy)
ModelRegistry.register("xgboost", XGBoostStrategy)
