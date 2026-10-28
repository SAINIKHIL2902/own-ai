from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient

from app.chat.service import ChatService
from app.main import app
from app.ollama.models import ChatMessage, ChatRequest
from app.router.router import RoutingDecision


@pytest.fixture
def mock_chat_service():
    mock_ollama = MagicMock()
    mock_ollama.generate = AsyncMock(return_value={"model": "llama3.2:1b", "response": "Local response", "provider": "local"})

    mock_gemini = MagicMock()
    mock_gemini.generate = AsyncMock(return_value={"model": "gemini-3.5-flash-lite", "response": "Cloud response", "provider": "gemini"})

    mock_router = MagicMock()
    mock_router.route.return_value = RoutingDecision(
        selected_provider="local",
        selected_model="llama3.2:1b",
        local_suitability=0.85,
        routing_confidence=0.90,
        task_type="general",
        difficulty="low",
        required_capabilities={},
        reason="Router chosen local",
    )

    mock_mem = MagicMock()
    mock_mem.retrieve_relevant_memories.return_value = []
    mock_mem.process_prompt_for_memories = MagicMock()

    service = ChatService(
        ollama_client=mock_ollama,
        gemini_client=mock_gemini,
        router=mock_router,
        memory_svc=mock_mem,
    )
    return service, mock_ollama, mock_gemini, mock_router


@pytest.mark.asyncio
async def test_offline_mode_directly_dispatches_to_local(mock_chat_service):
    service, mock_ollama, mock_gemini, mock_router = mock_chat_service

    req = ChatRequest(
        message="What is the weather?",
        mode="offline",
        model="llama3.2:1b",
    )
    resp = await service.chat(req)

    assert resp.provider == "local"
    assert resp.model == "llama3.2:1b"
    assert resp.response == "Local response"
    mock_ollama.generate.assert_called_once()
    mock_gemini.generate.assert_not_called()
    mock_router.route.assert_not_called()


@pytest.mark.asyncio
async def test_online_mode_directly_dispatches_to_gemini(mock_chat_service):
    service, mock_ollama, mock_gemini, mock_router = mock_chat_service

    req = ChatRequest(
        message="Write a complex Rust macro",
        mode="online",
        model="gemini-3.5-flash-lite",
    )
    resp = await service.chat(req)

    assert resp.provider == "gemini"
    assert resp.model == "gemini-3.5-flash-lite"
    assert resp.response == "Cloud response"
    mock_gemini.generate.assert_called_once()
    mock_ollama.generate.assert_not_called()
    mock_router.route.assert_not_called()


@pytest.mark.asyncio
async def test_auto_mode_calls_router(mock_chat_service):
    service, mock_ollama, mock_gemini, mock_router = mock_chat_service

    req = ChatRequest(
        message="Hello how are you?",
        mode="auto",
    )
    resp = await service.chat(req)

    mock_router.route.assert_called_once()
    assert resp.provider == "local"
    assert resp.model == "llama3.2:1b"
    mock_ollama.generate.assert_called_once()


@pytest.mark.asyncio
async def test_direct_model_selection_override(mock_chat_service):
    service, mock_ollama, mock_gemini, mock_router = mock_chat_service

    # Select specific gemini model in auto mode
    req = ChatRequest(
        message="Help me with math",
        mode="auto",
        model="gemini-3.5-flash-lite",
    )
    resp = await service.chat(req)

    assert resp.provider == "gemini"
    assert resp.model == "gemini-3.5-flash-lite"
    mock_gemini.generate.assert_called_once()
    mock_router.route.assert_not_called()


def test_models_api_endpoint_returns_modes_and_models():
    client = TestClient(app)
    res = client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert "local_models" in data
    assert "online_models" in data
    assert data["default_mode"] == "auto"
    assert any("gemini" in m for m in data["online_models"])
