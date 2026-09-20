import pytest
from unittest.mock import AsyncMock, patch
import httpx

from app.llm.gemini_client import (
    GeminiAPIError,
    GeminiAuthError,
    GeminiClient,
    GeminiConnectionError,
    GeminiRateLimitError,
)


@pytest.mark.asyncio
async def test_gemini_client_missing_api_key_raises_auth_error():
    client = GeminiClient(api_key=None)
    with pytest.raises(GeminiAuthError):
        await client.generate(messages=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_gemini_client_successful_generation():
    client = GeminiClient(api_key="fake-test-key-12345")

    mock_resp = httpx.Response(
        status_code=200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello! I am Gemini."}]
                    }
                }
            ]
        },
        request=httpx.Request("POST", "http://test"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await client.generate(messages=[{"role": "user", "content": "Hi"}])
        assert res["provider"] == "gemini"
        assert res["response"] == "Hello! I am Gemini."


@pytest.mark.asyncio
async def test_gemini_client_rate_limit_error():
    client = GeminiClient(api_key="fake-test-key-12345")
    mock_resp = httpx.Response(
        status_code=429,
        text="Rate limit exceeded",
        request=httpx.Request("POST", "http://test"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        with pytest.raises(GeminiRateLimitError):
            await client.generate(messages=[{"role": "user", "content": "Hi"}])


@pytest.mark.asyncio
async def test_gemini_client_connection_error():
    client = GeminiClient(api_key="fake-test-key-12345")

    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Network dropped")):
        with pytest.raises(GeminiConnectionError):
            await client.generate(messages=[{"role": "user", "content": "Hi"}])
