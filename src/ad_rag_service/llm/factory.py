from __future__ import annotations

import logging

from ad_rag_service import config
from ad_rag_service.llm.dummy_client import LLMClientImpl
from ad_rag_service.llm.interface import LLMClient

logger = logging.getLogger(__name__)


def get_llm_client() -> LLMClient:
    """
    Factory function to get the appropriate LLM client based on configuration.
    """
    if config.LLM_PROVIDER == "dummy":
        logger.info("Using Dummy LLM client.")
        return LLMClientImpl()
    elif config.LLM_PROVIDER == "openai":
        from ad_rag_service.llm.openai_client import OpenAIClient

        logger.info("Using OpenAI LLM client.")
        return OpenAIClient()
    elif config.LLM_PROVIDER == "anthropic":
        # TODO: Implement Anthropic client in Phase 3
        raise NotImplementedError("Anthropic client not yet implemented.")
    else:
        # This case should ideally be caught by config validation, but as a safeguard
        raise ValueError(f"Unknown LLM provider: {config.LLM_PROVIDER}")
