"""Machine-learning strategy implementations."""

from quant_frame.core.model_strategy import ModelRegistry

from .ensemble_strategy import EnsembleStrategy
from .hmm_strategy import GaussianHMMStrategy
from .llm_strategy import LLMStrategy
from .ppo_strategy import PPOStrategy
from .xgboost_strategy import XGBoostStrategy

__all__ = ["EnsembleStrategy", "GaussianHMMStrategy", "LLMStrategy", "PPOStrategy", "XGBoostStrategy"]

ModelRegistry.register("ensemble", EnsembleStrategy)
ModelRegistry.register("gaussian_hmm", GaussianHMMStrategy)
ModelRegistry.register("llm", LLMStrategy)
ModelRegistry.register("ppo", PPOStrategy)
ModelRegistry.register("xgboost", XGBoostStrategy)
