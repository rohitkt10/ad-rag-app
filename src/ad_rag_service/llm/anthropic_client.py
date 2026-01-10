from __future__ import annotations

import logging
import os

from anthropic import Anthropic, APIStatusError

from ad_rag_service import config
from ad_rag_service.llm.interface import LLMClient

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """
    Adapter for Anthropic's Messages API (Claude).
    """

    def __init__(self) -> None:
        api_key_var = config.ANTHROPIC_API_KEY_ENV
        api_key = os.getenv(api_key_var)
        if not api_key:
            logger.error(f"Anthropic API key not found in environment variable: {api_key_var}")
            raise ValueError(f"Anthropic API key not found. Please set {api_key_var}.")

        self.client = Anthropic(api_key=api_key)
        self.model = config.LLM_MODEL_NAME
        logger.info(f"Initialized AnthropicClient with model: {self.model}")

    def complete(
        self,
        prompt: str,
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS,
    ) -> str:
        """
        Generate a completion using Anthropic Claude.
        """
        try:
            logger.debug(
                "Sending request to Anthropic (model=%s, temp=%s, max_tokens=%s)",
                self.model,
                temperature,
                max_tokens,
            )
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Anthropic response content is a list of blocks.
            # We typically want the text from the first text block.
            if not response.content:
                logger.warning("Anthropic returned empty content.")
                return ""

            text_content = []
            for block in response.content:
                if block.type == "text":
                    text_content.append(block.text)

            full_text = "".join(text_content)
            return full_text

        except APIStatusError as e:
            logger.error(f"Anthropic API error: {e}")
            raise RuntimeError(f"Anthropic API error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic generation: {e}")
            raise RuntimeError(f"Unexpected error during Anthropic generation: {e}") from e
