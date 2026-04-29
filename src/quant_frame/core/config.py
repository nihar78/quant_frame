"""Strongly-typed configuration manager for strategy parameters.

This module provides nested Pydantic models that enforce correct types and
structures for pipeline configuration data, whether loaded from dictionaries,
YAML strings, or YAML files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class FeatureConfig(BaseModel):
    """Configuration for feature-engineering parameters.

    Attributes:
        ma_windows: List of window lengths (in periods) to use when
            computing moving-average features.
        target_shift: Number of periods to shift the target variable
            forward for prediction/labeling (must be positive).
    """

    ma_windows: list[int] = Field(
        ..., description="Ordered list of moving-average window lengths."
    )
    target_shift: int = Field(
        ..., description="Forward shift applied to the target variable."
    )


class ModelConfig(BaseModel):
    """Configuration for model-related hyperparameters.

    Attributes:
        hmm_components: Number of hidden states/components for an HMM.
        model_params: Free-form dictionary of additional model parameters
            (e.g. ``covariance_type``, ``n_iter``, XGBoost kwargs, etc.).
    """

    hmm_components: int = Field(
        ..., description="Number of hidden Markov model components."
    )
    model_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments passed to the estimator.",
    )


class PipelineConfig(BaseModel):
    """Master configuration object for a modelling pipeline.

    This model aggregates :class:`FeatureConfig` and :class:`ModelConfig`
    into a single validated document and provides a convenience method for
    loading from YAML sources.

    Attributes:
        features: Feature-engineering parameters.
        model: Model hyperparameters and extra kwargs.

    Example:
        >>> config = PipelineConfig(
        ...     features=FeatureConfig(ma_windows=[5, 10], target_shift=1),
        ...     model=ModelConfig(hmm_components=3, model_params={}),
        ... )
        >>> config.features.ma_windows
        [5, 10]
    """

    features: FeatureConfig = Field(
        ..., description="Feature-engineering configuration block."
    )
    model: ModelConfig = Field(
        ..., description="Model configuration block."
    )

    @classmethod
    def from_yaml(cls, yaml_path_or_str: str) -> "PipelineConfig":
        """Load a :class:`PipelineConfig` from a YAML file or raw YAML string.

        This method first checks whether *yaml_path_or_str* corresponds to
        an existing filesystem path.  If so, the file contents are read and
        parsed; otherwise the input is treated as an inline YAML document.

        Args:
            yaml_path_or_str: Either an absolute/relative path to a ``.yaml``
                file **or** a raw YAML payload (e.g. a multi-line string).

        Returns:
            A fully validated :class:`PipelineConfig` instance.

        Raises:
            FileNotFoundError: If *yaml_path_or_str* is interpreted as a
                path and the file does not exist.
            yaml.YAMLError: If the supplied YAML cannot be parsed.
            pydantic.ValidationError: If the parsed data does not conform to
                the expected schema.
        """
        path = Path(yaml_path_or_str)
        if path.exists():
            raw_data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            raw_data = yaml.safe_load(yaml_path_or_str)
        return cls.model_validate(raw_data)
