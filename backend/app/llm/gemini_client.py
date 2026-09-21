import logging
from typing import Any, Dict, List, Optional
import httpx

from app.config.settings import settings
from app.llm.base import BaseLLMClient, LLMClientError

logger = logging.getLogger(__name__)


class GeminiError(LLMClientError):
    """Base exception for Gemini client errors."""
    pass


class GeminiAuthError(GeminiError):
    """Raised when Gemini API key is missing or invalid."""
    pass


class GeminiRateLimitError(GeminiError):
    """Raised when Gemini API rate limit is exceeded."""
    pass


class GeminiConnectionError(GeminiError):
    """Raised when connecting to Gemini API fails or times out."""
    pass


class GeminiAPIError(GeminiError):
    """Raised when Gemini returns a non-200 API error."""
    pass


class GeminiClient(BaseLLMClient):
    """
    Client for Google Gemini REST API.
    Guarantees that credentials are never logged or exposed in exceptions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self._api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        self.base_url = (base_url or settings.GEMINI_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.GEMINI_TIMEOUT

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key and self._api_key.strip())

    def _convert_messages_to_contents(self, messages: List[Dict[str, str]]) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Convert standard role/content messages into Gemini contents format."""
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:  # user
                contents.append({"role": "user", "parts": [{"text": content}]})

        return system_instruction, contents

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send chat messages to Gemini generateContent endpoint.
        NOTE: Messages MUST be sanitized by PrivacyScanner before calling this method.
        """
        if not self.is_configured:
            raise GeminiAuthError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in environment or .env file."
            )

        target_model = model or self.model
        url = f"{self.base_url}/models/{target_model}:generateContent"

        system_instruction, contents = self._convert_messages_to_contents(messages)
        payload: Dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=float(self.timeout)) as client:
                response = await client.post(url, json=payload, headers=headers)

                if response.status_code == 401 or response.status_code == 403:
                    raise GeminiAuthError(
                        f"Gemini API authentication failed (HTTP {response.status_code}). Verify GEMINI_API_KEY."
                    )
                if response.status_code == 429:
                    raise GeminiRateLimitError(
                        "Gemini API rate limit exceeded (HTTP 429). Please retry later."
                    )
                if response.status_code != 200:
                    raise GeminiAPIError(
                        f"Gemini API returned error HTTP {response.status_code}: {response.text[:200]}"
                    )

                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise GeminiAPIError("Gemini response contained no candidates.")

                candidate = candidates[0]
                content_obj = candidate.get("content", {})
                parts = content_obj.get("parts", [])
                text_response = "".join(p.get("text", "") for p in parts)

                return {
                    "model": target_model,
                    "response": text_response,
                    "provider": "gemini",
                }

        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise GeminiConnectionError(
                f"Failed to connect to Gemini API endpoint. Check internet connection: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise GeminiConnectionError(
                f"Gemini API request timed out after {self.timeout} seconds."
            ) from exc
        except (GeminiError, LLMClientError):
            raise
        except Exception as exc:
            raise GeminiAPIError(f"Unexpected error communicating with Gemini API: {exc}") from exc
