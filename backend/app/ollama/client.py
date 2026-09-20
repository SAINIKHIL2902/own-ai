import logging
from typing import Any, Dict, List, Optional
import httpx

from app.config.settings import settings

logger = logging.getLogger(__name__)


class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaConnectionError(OllamaClientError):
    """Raised when Ollama daemon cannot be reached."""
    pass


class OllamaModelNotFoundError(OllamaClientError):
    """Raised when the requested model is not found in Ollama."""
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.OLLAMA_TIMEOUT

    async def check_health(self) -> bool:
        """Check if Ollama daemon is reachable and responding."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/")
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, Exception) as exc:
            logger.warning(f"Ollama health check failed: {exc}")
            return False

    async def list_models(self) -> List[str]:
        """Fetch list of models available in local Ollama daemon."""
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                if response.status_code != 200:
                    raise OllamaClientError(
                        f"Failed to fetch models from Ollama (status {response.status_code}): {response.text}"
                    )
                data = response.json()
                models = [item.get("name", "") for item in data.get("models", []) if item.get("name")]
                return models
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Ollama is not running at {self.base_url}. Please ensure 'ollama serve' is active."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaConnectionError(
                f"Timed out connecting to Ollama at {self.base_url}."
            ) from exc

    async def generate_chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> Dict[str, Any]:
        """Send chat messages to Ollama /api/chat with stream: false."""
        target_model = model or settings.OLLAMA_MODEL
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=float(self.timeout)) as client:
                response = await client.post(url, json=payload)

                if response.status_code == 404:
                    raise OllamaModelNotFoundError(
                        f"Model '{target_model}' is not installed in Ollama. Please run 'ollama pull {target_model}'."
                    )
                if response.status_code != 200:
                    raise OllamaClientError(
                        f"Ollama error (status {response.status_code}): {response.text}"
                    )

                data = response.json()
                message_content = data.get("message", {}).get("content", "")
                return {
                    "model": data.get("model", target_model),
                    "response": message_content,
                }
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Ollama is not running at {self.base_url}."
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaConnectionError(
                f"Ollama request timed out after {self.timeout} seconds for model '{target_model}'."
            ) from exc
        except (OllamaModelNotFoundError, OllamaConnectionError):
            raise
        except Exception as exc:
            raise OllamaClientError(f"Unexpected error communicating with Ollama: {exc}") from exc
