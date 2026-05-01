"""LLM-driven agentic debate strategy.

This module provides a concrete implementation of :class:`BaseModelStrategy`
that leverages a large language model (OpenAI) to simulate a panel of personas
debating state features and returning a consensus weight.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI

from quant_frame.core.model_strategy import BaseModelStrategy

LOGGER = logging.getLogger(__name__)


class LLMStrategy(BaseModelStrategy):
    """Strategy that queries an LLM for a consensus weight per observation.

    The model is treated as pre-trained, so :meth:`train` is a no-op.
    :meth:`predict` sends each row (as JSON) to the LLM together with a
    system prompt that enumerates the supplied *personas* and instructs the
    model to output a JSON object containing a ``consensus_weight`` float in
    ``[-1.0, 1.0]``.

    Args:
        api_key: OpenAI API key used to initialise the client.
        model_name: OpenAI model identifier.  Defaults to ``"gpt-4o-mini"``.
        personas: Optional list of persona descriptions that are injected
            into the system prompt.

    Example:
        >>> strategy = LLMStrategy(api_key="sk-...", personas=["bullish analyst", "bearish analyst"])
        >>> strategy.train(df, features=["x"])
        >>> preds = strategy.predict(df, features=["x"])
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        personas: list[str] | None = None,
    ) -> None:
        """Initialise the strategy.

        Args:
            api_key: OpenAI API key.
            model_name: Model identifier passed to ``chat.completions.create``.
            personas: Descriptive strings for the debating agents.
        """
        self.api_key: str = api_key
        self.model_name: str = model_name
        self.personas: list[str] = personas if personas is not None else []
        self._client: OpenAI = OpenAI(api_key=api_key)
        self._is_fitted: bool = False

    def train(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str | None = None,
    ) -> None:
        """No-op: the LLM is pre-trained.

        Args:
            df: Ignored.
            features: Ignored.
            target: Ignored.
        """
        self._is_fitted = True

    def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        """Generate consensus weights for every row via the LLM.

        Args:
            df: Input tabular data containing the explanatory variables.
            features: Ordered list of column names the model was (logically)
                trained on.

        Returns:
            A one-dimensional NumPy array of consensus weights with length
            equal to the number of rows in *df*.

        Raises:
            ValueError: If :meth:`train` has not been called.
        """
        if not self._is_fitted:
            raise ValueError(
                "The model has not been fitted yet. Call train() before predict()."
            )

        system_prompt = self._build_system_prompt()
        results: list[float] = []

        for _, row in df.iterrows():
            row_dict = row[features].to_dict()
            row_json = json.dumps(row_dict)
            user_prompt = f"Debate the following state features.\n{row_json}"

            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if content is None:
                    weight = 0.0
                else:
                    parsed: dict[str, Any] = json.loads(content)
                    weight = float(parsed.get("consensus_weight", 0.0))
            except Exception as exc:
                LOGGER.warning("LLM response parsing failed: %s. Defaulting to 0.0.", exc)
                weight = 0.0

            results.append(weight)

        return np.array(results, dtype=np.float64)

    def _build_system_prompt(self) -> str:
        """Assemble the system prompt from personas and instructions."""
        if self.personas:
            personas_text = "\n".join(f"- {p}" for p in self.personas)
            personas_section = (
                f"You are simulating the following personas:\n{personas_text}\n\n"
            )
        else:
            personas_section = ""

        return (
            f"{personas_section}"
            "Debate the following state features. Output ONLY a single JSON object with a "
            "'consensus_weight' key containing a float between -1.0 and 1.0."
        )

    def save(self, filepath: str) -> None:
        """No-op: there is no local model state to persist.

        Args:
            filepath: Ignored.
        """
        LOGGER.info("LLMStrategy.save() is a no-op; nothing to persist.")
