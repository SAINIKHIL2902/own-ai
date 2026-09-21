from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMClientError(Exception):
    """Base exception for LLM provider errors."""
    pass


class BaseLLMClient(ABC):
    """Abstract interface for all LLM providers (Ollama, Gemini, etc.)."""

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate completion for chat messages.
        Must return dict with keys:
            - model: str
            - response: str
            - provider: str ('local' or 'gemini')
        """
        raise NotImplementedError
