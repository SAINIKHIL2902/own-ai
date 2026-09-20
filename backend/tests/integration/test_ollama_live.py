import pytest
from app.config.settings import settings
from app.ollama.client import OllamaClient


@pytest.mark.asyncio
async def test_live_ollama_health_and_tags():
    """Verify live Ollama daemon responds to health and tags."""
    client = OllamaClient()
    is_healthy = await client.check_health()
    assert is_healthy is True, f"Ollama daemon at {settings.OLLAMA_BASE_URL} is not responding."

    models = await client.list_models()
    assert isinstance(models, list), "Models should be a list"
    assert any(settings.OLLAMA_MODEL in m for m in models), (
        f"Configured model '{settings.OLLAMA_MODEL}' was not found in live Ollama models: {models}"
    )


@pytest.mark.asyncio
async def test_live_ollama_inference():
    """Verify real inference execution against live local Ollama model."""
    client = OllamaClient()
    messages = [{"role": "user", "content": "Respond with the single word: PONG"}]
    
    result = await client.generate_chat(messages=messages, model=settings.OLLAMA_MODEL)
    
    assert "response" in result
    assert len(result["response"].strip()) > 0
    assert result["model"].startswith("llama3.2:1b")
    # Verify we got a non-empty string response from the real model
    print(f"\n[LIVE TEST] Model '{result['model']}' response: {result['response']}")
