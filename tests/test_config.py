"""Tests for the configuration manager."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from quant_frame import PipelineConfig
from quant_frame.core.config import FeatureConfig, ModelConfig


class TestFeatureConfig:
    """Test suite for :class:`FeatureConfig`."""

    def test_valid_instantiation(self) -> None:
        """A valid FeatureConfig should instantiate without errors."""
        config = FeatureConfig(ma_windows=[5, 10, 20], target_shift=1)
        assert config.ma_windows == [5, 10, 20]
        assert config.target_shift == 1

    def test_invalid_ma_windows_type(self) -> None:
        """Passing a string for ma_windows should raise ValidationError."""
        with pytest.raises(ValidationError):
            FeatureConfig(ma_windows="not_a_list", target_shift=1)

    def test_invalid_target_shift_type(self) -> None:
        """Passing a string for target_shift should raise ValidationError."""
        with pytest.raises(ValidationError):
            FeatureConfig(ma_windows=[5, 10], target_shift="not_an_int")


class TestModelConfig:
    """Test suite for :class:`ModelConfig`."""

    def test_valid_instantiation(self) -> None:
        """A valid ModelConfig should instantiate without errors."""
        config = ModelConfig(hmm_components=3, model_params={"covariance_type": "full"})
        assert config.hmm_components == 3
        assert config.model_params == {"covariance_type": "full"}

    def test_invalid_hmm_components_type(self) -> None:
        """Passing a string for hmm_components should raise ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(hmm_components="three", model_params={})


class TestPipelineConfig:
    """Test suite for :class:`PipelineConfig`."""

    def test_valid_instantiation_from_dict(self) -> None:
        """A PipelineConfig should instantiate from a nested dictionary."""
        data = {
            "features": {"ma_windows": [5, 10, 20], "target_shift": 1},
            "model": {"hmm_components": 3, "model_params": {"covariance_type": "full"}},
        }
        config = PipelineConfig.model_validate(data)
        assert config.features.ma_windows == [5, 10, 20]
        assert config.features.target_shift == 1
        assert config.model.hmm_components == 3
        assert config.model.model_params == {"covariance_type": "full"}

    def test_invalid_window_type_raises_validation_error(self) -> None:
        """Passing a string instead of a list of ints should raise ValidationError."""
        data = {
            "features": {"ma_windows": "not_a_list", "target_shift": 1},
            "model": {"hmm_components": 3, "model_params": {}},
        }
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(data)


class TestPipelineConfigFromYaml:
    """Test suite for :meth:`PipelineConfig.from_yaml`."""

    def test_from_yaml_string(self) -> None:
        """A PipelineConfig should instantiate from a valid YAML string."""
        yaml_str = """
features:
  ma_windows:
    - 5
    - 10
    - 20
  target_shift: 1
model:
  hmm_components: 3
  model_params:
    covariance_type: full
"""
        config = PipelineConfig.from_yaml(yaml_str)
        assert config.features.ma_windows == [5, 10, 20]
        assert config.features.target_shift == 1
        assert config.model.hmm_components == 3
        assert config.model.model_params == {"covariance_type": "full"}

    def test_from_yaml_file(self, tmp_path: Path) -> None:
        """A PipelineConfig should instantiate from a valid YAML file path."""
        yaml_file = tmp_path / "config.yaml"
        data = {
            "features": {"ma_windows": [10, 30], "target_shift": 5},
            "model": {"hmm_components": 4, "model_params": {"n_iter": 100}},
        }
        yaml_file.write_text(yaml.dump(data))

        config = PipelineConfig.from_yaml(str(yaml_file))
        assert config.features.ma_windows == [10, 30]
        assert config.features.target_shift == 5
        assert config.model.hmm_components == 4
        assert config.model.model_params == {"n_iter": 100}

    def test_from_yaml_invalid_type_raises_validation_error(self) -> None:
        """Passing invalid types via YAML string should raise ValidationError."""
        yaml_str = """
features:
  ma_windows: "not_a_list"
  target_shift: 1
model:
  hmm_components: 3
  model_params: {}
"""
        with pytest.raises(ValidationError):
            PipelineConfig.from_yaml(yaml_str)
