import asyncio
from unittest.mock import AsyncMock
import pytest

from app.chat.service import ChatService
from app.llm.base import BaseLLMClient
from app.memory.models import MemoryRecord, MemoryType
from app.memory.repository import MemoryRepository
from app.memory.service import MemoryService
from app.ollama.models import ChatRequest
from app.storage.database import DatabaseManager
from app.storage.repositories import RawRepository


class MockOllamaClient:
    def __init__(self, response: str = "Here is the response."):
        self.response = response
        self.last_messages = []

    async def generate(self, messages, model=None):
        self.last_messages = list(messages)
        return {"model": model or "llama3.2:1b", "response": self.response}


@pytest.mark.asyncio
async def test_chat_service_retrieves_memory_and_personalizes_prompt(tmp_path):
    db_file = tmp_path / "test_chat_mem.db"
    mgr = DatabaseManager(db_path=db_file)
    raw = RawRepository(manager=mgr)
    mem_repo = MemoryRepository(manager=mgr)
    mem_svc = MemoryService(repository=mem_repo)

    # Pre-populate a preference memory
    pref = MemoryRecord(
        memory_id="mem_pref_1",
        user_id="local_user",
        memory_type=MemoryType.PREFERENCE.value,
        content="User prefers bullet points and concise technical summaries",
        source_type="explicit_user",
        confidence=0.95,
        importance=0.9,
    )
    mem_repo.save_memory(pref)

    mock_ollama = MockOllamaClient(response="• Bullet 1\n• Bullet 2")
    mock_gemini = AsyncMock(spec=BaseLLMClient)
    mock_producer = AsyncMock()

    service = ChatService(
        ollama_client=mock_ollama,
        gemini_client=mock_gemini,
        repository=raw,
        producer=mock_producer,
        memory_svc=mem_svc,
    )

    req = ChatRequest(message="Can you give me bullet points on SQLite features?")
    resp = await service.chat(req)

    assert resp.provider == "local"
    assert resp.response == "• Bullet 1\n• Bullet 2"

    # Verify that mock_ollama received the personalized system prompt containing the memory
    system_messages = [m for m in mock_ollama.last_messages if m["role"] == "system"]
    assert len(system_messages) > 0
    assert "User prefers bullet points and concise technical summaries" in system_messages[0]["content"]


@pytest.mark.asyncio
async def test_chat_service_async_memory_detection(tmp_path):
    db_file = tmp_path / "test_chat_async_mem.db"
    mgr = DatabaseManager(db_path=db_file)
    raw = RawRepository(manager=mgr)
    mem_repo = MemoryRepository(manager=mgr)
    mem_svc = MemoryService(repository=mem_repo)

    mock_ollama = MockOllamaClient(response="Understood, I will remember that.")
    mock_producer = AsyncMock()

    service = ChatService(
        ollama_client=mock_ollama,
        repository=raw,
        producer=mock_producer,
        memory_svc=mem_svc,
    )

    # Prompt with explicit memory statement
    req = ChatRequest(message="Remember that I usually prefer TypeScript over JavaScript.")
    resp = await service.chat(req)

    assert resp.response == "Understood, I will remember that."

    # Give the background task a moment to complete execution
    await asyncio.sleep(0.1)

    # Verify memory was detected and saved to SQLite
    active_mems = mem_repo.get_active_memories()
    assert len(active_mems) > 0
    detected_content = [m.content for m in active_mems]
    assert any("prefer typescript" in c.lower() for c in detected_content)
