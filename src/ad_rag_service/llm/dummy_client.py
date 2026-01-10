import logging

from ad_rag_service.llm.interface import LLMClient

logger = logging.getLogger(__name__)


class LLMClientImpl(LLMClient):
    """
    Placeholder/example LLM client. In a real app, this would integrate with a
    specific LLM provider (e.g., OpenAI, Gemini, HuggingFace Inference API).
    """

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        logger.warning("Using a dummy LLMClientImpl. Replace with a real LLM integration.")
        return "This is a dummy answer from LLMClientImpl [1]."
