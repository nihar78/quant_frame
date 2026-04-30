"""Machine-learning strategy implementations."""

from quant_frame.core.model_strategy import ModelRegistry

from .hmm_strategy import GaussianHMMStrategy
from .ppo_strategy import PPOStrategy
from .xgboost_strategy import XGBoostStrategy

__all__ = ["GaussianHMMStrategy", "PPOStrategy", "XGBoostStrategy"]

ModelRegistry.register("gaussian_hmm", GaussianHMMStrategy)
ModelRegistry.register("ppo", PPOStrategy)
ModelRegistry.register("xgboost", XGBoostStrategy)
