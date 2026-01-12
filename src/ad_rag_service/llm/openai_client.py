from __future__ import annotations

import logging
import os

from openai import OpenAI, OpenAIError

from ad_rag_service import config
from ad_rag_service.llm.interface import LLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """
    Adapter for OpenAI's Chat Completions API.
    """

    def __init__(self) -> None:
        api_key_var = config.OPENAI_API_KEY_ENV
        api_key = os.getenv(api_key_var)
        if not api_key:
            logger.error(f"OpenAI API key not found in environment variable: {api_key_var}")
            raise ValueError(f"OpenAI API key not found. Please set {api_key_var}.")

        self.client = OpenAI(api_key=api_key)
        self.model = config.LLM_MODEL_NAME
        logger.info(f"Initialized OpenAIClient with model: {self.model}")

    def complete(
        self,
        prompt: str,
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS,
    ) -> str:
        """
        Generate a completion using OpenAI.
        """
        try:
            # Estimate a safe word limit (approx 0.5 words per token)
            safe_word_limit = int(max_tokens * 0.5)
            system_instruction = (
                f"\n\nNote: This request has an upper limit on number of output tokens. "
                f"Please keep your answer to within approximately {safe_word_limit} words."
            )
            final_prompt = prompt + system_instruction

            logger.debug(
                "Sending request to OpenAI (model=%s, temp=%s, max_tokens=%s)",
                self.model,
                temperature,
                max_tokens,
            )
            logger.debug(f"Prompt (first 500 chars): {final_prompt[:500]}...")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                reasoning_effort=config.REASONING_EFFORT,
            )
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            if content and content.strip():
                logger.debug(
                    f"OpenAI returned content (len={len(content)}, "
                    f"finish_reason={finish_reason}): {content[:100]}..."
                )
                return content

            if finish_reason == "length":
                logger.warning("OpenAI returned empty content with finish_reason='length'.")

            logger.warning(f"OpenAI returned null/empty content. Full response: {response}")
            return ""

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI generation: {e}")
            raise RuntimeError(f"Unexpected error during OpenAI generation: {e}") from e
