from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Abstract interface for an LLM provider."""

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str: ...
