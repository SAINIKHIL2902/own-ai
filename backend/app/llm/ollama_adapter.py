import logging
from typing import Any, Dict, List, Optional

from app.config.settings import settings
from app.llm.base import BaseLLMClient
from app.ollama.client import (
    OllamaClient,
    OllamaClientError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
)

logger = logging.getLogger(__name__)


class OllamaAdapter(BaseLLMClient):
    """Adapts the existing OllamaClient to the common BaseLLMClient interface."""

    def __init__(self, client: Optional[Any] = None):
        self._client = client

    @property
    def client(self) -> Any:
        return self._client if self._client is not None else OllamaClient()

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        target_client = self.client
        if hasattr(target_client, "generate_chat"):
            result = await target_client.generate_chat(
                messages=messages,
                model=model or settings.OLLAMA_MODEL,
            )
        elif hasattr(target_client, "generate"):
            result = await target_client.generate(
                messages=messages,
                model=model or settings.OLLAMA_MODEL,
            )
        else:
            raise AttributeError("Client has neither 'generate_chat' nor 'generate' method.")

        return {
            "model": result.get("model", model or settings.OLLAMA_MODEL),
            "response": result.get("response", ""),
            "provider": "local",
        }
