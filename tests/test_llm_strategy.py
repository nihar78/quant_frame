"""Tests for the LLM-driven agentic debate strategy."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_frame.strategies.llm_strategy import LLMStrategy


class TestLLMStrategy:
    """Test suite for LLMStrategy."""

    @pytest.fixture
    def dummy_df(self) -> pd.DataFrame:
        return pd.DataFrame({"feat_a": [1.0, 2.0, 3.0], "feat_b": [0.5, 1.5, 2.5]})

    def _mock_client(self, return_content: str | None) -> MagicMock:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=return_content))]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_instantiation_defaults(self) -> None:
        with patch("quant_frame.strategies.llm_strategy.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            strategy = LLMStrategy(api_key="test-key")
            assert strategy.model_name == "gpt-4o-mini"
            assert strategy.personas == []

    def test_instantiation_with_personas(self) -> None:
        with patch("quant_frame.strategies.llm_strategy.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            strategy = LLMStrategy(api_key="test-key", model_name="gpt-4", personas=["bullish", "bearish"])
            assert strategy.model_name == "gpt-4"
            assert strategy.personas == ["bullish", "bearish"]

    def test_train_sets_is_fitted(self, dummy_df: pd.DataFrame) -> None:
        with patch("quant_frame.strategies.llm_strategy.OpenAI"):
            strategy = LLMStrategy(api_key="test-key")
        strategy.train(dummy_df, features=["feat_a", "feat_b"])
        assert strategy._is_fitted is True

    def test_predict_before_train_raises(self, dummy_df: pd.DataFrame) -> None:
        with patch("quant_frame.strategies.llm_strategy.OpenAI"):
            strategy = LLMStrategy(api_key="test-key")
        with pytest.raises(ValueError, match="not been fitted"):
            strategy.predict(dummy_df, features=["feat_a", "feat_b"])

    @patch("quant_frame.strategies.llm_strategy.OpenAI")
    def test_predict_returns_numpy_array(self, mock_openai: MagicMock, dummy_df: pd.DataFrame) -> None:
        mock_openai.return_value = self._mock_client(return_content='{"consensus_weight": 0.75}')
        strategy = LLMStrategy(api_key="test-key")
        strategy.train(dummy_df, features=["feat_a", "feat_b"])
        preds = strategy.predict(dummy_df, features=["feat_a", "feat_b"])
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(dummy_df),)
        np.testing.assert_array_almost_equal(preds, np.full(len(dummy_df), 0.75))

    @patch("quant_frame.strategies.llm_strategy.OpenAI")
    def test_predict_defaults_on_malformed_json(self, mock_openai: MagicMock, dummy_df: pd.DataFrame) -> None:
        mock_openai.return_value = self._mock_client(return_content="not valid json")
        strategy = LLMStrategy(api_key="test-key")
        strategy.train(dummy_df, features=["feat_a", "feat_b"])
        preds = strategy.predict(dummy_df, features=["feat_a", "feat_b"])
        np.testing.assert_array_almost_equal(preds, np.zeros(len(dummy_df)))

    @patch("quant_frame.strategies.llm_strategy.OpenAI")
    def test_predict_defaults_on_missing_key(self, mock_openai: MagicMock, dummy_df: pd.DataFrame) -> None:
        mock_openai.return_value = self._mock_client(return_content='{"other_key": 42}')
        strategy = LLMStrategy(api_key="test-key")
        strategy.train(dummy_df, features=["feat_a", "feat_b"])
        preds = strategy.predict(dummy_df, features=["feat_a", "feat_b"])
        np.testing.assert_array_almost_equal(preds, np.zeros(len(dummy_df)))

    @patch("quant_frame.strategies.llm_strategy.OpenAI")
    def test_predict_defaults_on_none_content(self, mock_openai: MagicMock, dummy_df: pd.DataFrame) -> None:
        mock_openai.return_value = self._mock_client(return_content=None)
        strategy = LLMStrategy(api_key="test-key")
        strategy.train(dummy_df, features=["feat_a", "feat_b"])
        preds = strategy.predict(dummy_df, features=["feat_a", "feat_b"])
        np.testing.assert_array_almost_equal(preds, np.zeros(len(dummy_df)))

    @patch("quant_frame.strategies.llm_strategy.OpenAI")
    def test_save_does_not_raise(self, mock_openai: MagicMock, tmp_path: Path) -> None:
        mock_openai.return_value = MagicMock()
        strategy = LLMStrategy(api_key="test-key")
        strategy.save(str(tmp_path / "llm.json"))
