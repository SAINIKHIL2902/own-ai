import pytest
from pydantic import ValidationError

from app.chat.service import ChatService
from app.ollama.client import OllamaClient, OllamaConnectionError, OllamaModelNotFoundError
from app.ollama.models import ChatMessage, ChatRequest


def test_chat_request_valid_single_message():
    req = ChatRequest(message="Hello world")
    assert req.message == "Hello world"
    assert req.messages is None


def test_chat_request_valid_messages_list():
    req = ChatRequest(messages=[ChatMessage(role="user", content="Hello")])
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"


def test_chat_request_rejects_empty():
    with pytest.raises(ValidationError):
        ChatRequest()

    with pytest.raises(ValidationError):
        ChatRequest(message="")

    with pytest.raises(ValidationError):
        ChatRequest(message="   ")


def test_chat_request_rejects_empty_messages():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])

    with pytest.raises(ValidationError):
        ChatRequest(messages=[ChatMessage(role="user", content="   ")])


class MockOllamaClient:
    def __init__(self, should_fail_connection=False, should_fail_not_found=False):
        self.should_fail_connection = should_fail_connection
        self.should_fail_not_found = should_fail_not_found
        self.last_messages = None

    async def generate_chat(self, messages, model=None):
        self.last_messages = messages
        if self.should_fail_connection:
            raise OllamaConnectionError("Ollama is not running at http://127.0.0.1:11434")
        if self.should_fail_not_found:
            raise OllamaModelNotFoundError("Model 'llama3.2:1b' is not installed.")
        return {
            "model": model or "test-model",
            "response": "Hello from mock",
        }


@pytest.mark.asyncio
async def test_chat_service_success():
    client = MockOllamaClient()
    service = ChatService(ollama_client=client)

    req = ChatRequest(message="What is Python?")
    resp = await service.chat(req)

    assert resp.response == "Hello from mock"
    assert client.last_messages == [{"role": "user", "content": "What is Python?"}]


@pytest.mark.asyncio
async def test_chat_service_history_orchestration():
    client = MockOllamaClient()
    service = ChatService(ollama_client=client)

    req = ChatRequest(
        messages=[
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello!"),
        ],
        message="How are you?",
    )
    resp = await service.chat(req)

    assert resp.response == "Hello from mock"
    assert len(client.last_messages) == 3
    assert client.last_messages[0] == {"role": "user", "content": "Hi"}
    assert client.last_messages[1] == {"role": "assistant", "content": "Hello!"}
    assert client.last_messages[2] == {"role": "user", "content": "How are you?"}


@pytest.mark.asyncio
async def test_chat_service_propagates_connection_error():
    client = MockOllamaClient(should_fail_connection=True)
    service = ChatService(ollama_client=client)

    with pytest.raises(OllamaConnectionError):
        await service.chat(ChatRequest(message="test"))


@pytest.mark.asyncio
async def test_chat_service_propagates_model_not_found_error():
    client = MockOllamaClient(should_fail_not_found=True)
    service = ChatService(ollama_client=client)

    with pytest.raises(OllamaModelNotFoundError):
        await service.chat(ChatRequest(message="test"))
