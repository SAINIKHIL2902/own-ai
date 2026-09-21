from unittest.mock import AsyncMock
import pytest

from app.chat.service import ChatService
from app.llm.base import BaseLLMClient
from app.memory.models import MemoryRecord, MemoryType
from app.memory.repository import MemoryRepository
from app.memory.service import MemoryService
from app.ollama.models import ChatRequest
from app.privacy.scanner import PrivacyScanner
from app.storage.database import DatabaseManager
from app.storage.repositories import RawRepository


class MockGeminiClient(BaseLLMClient):
    def __init__(self):
        self.received_messages = []

    async def generate(self, messages, model=None):
        self.received_messages = list(messages)
        return {"model": model or "gemini-2.5-flash", "response": "Response with sensitive info redacted."}


class MockOllamaClient:
    def __init__(self):
        self.received_messages = []

    async def generate(self, messages, model=None):
        self.received_messages = list(messages)
        return {"model": model or "llama3.2:1b", "response": "Local response with full fidelity."}


@pytest.mark.asyncio
async def test_gemini_receives_only_sanitized_memory_context(tmp_path):
    db_file = tmp_path / "test_privacy.db"
    mgr = DatabaseManager(db_path=db_file)
    raw_repo = RawRepository(manager=mgr)
    mem_repo = MemoryRepository(manager=mgr)
    mem_service = MemoryService(repository=mem_repo)

    # 1. Store a memory containing a secret API key locally
    secret_key = "sk-live_99887766554433221100aabbccddeeff"
    secret_mem = MemoryRecord(
        memory_id="mem_secret",
        user_id="local_user",
        memory_type=MemoryType.FACT.value,
        content=f"The project OpenAI production key is {secret_key}",
        source_type="explicit_user",
        confidence=0.95,
        importance=0.95,
    )
    mem_repo.save_memory(secret_mem)

    # Verify original memory is stored in SQLite
    stored_locally = mem_repo.get_memory("mem_secret")
    assert stored_locally is not None
    assert secret_key in stored_locally.content

    # 2. Setup ChatService with mock clients
    mock_gemini = MockGeminiClient()
    mock_ollama = MockOllamaClient()
    mock_producer = AsyncMock()

    service = ChatService(
        ollama_client=mock_ollama,
        gemini_client=mock_gemini,
        repository=raw_repo,
        producer=mock_producer,
        memory_svc=mem_service,
    )

    # 3. Prompt that requires high reasoning / Gemini routing and references the secret key
    complex_prompt = (
        "Analyze the formal consistency proofs and failure modes between Multi-Raft active-active "
        "replication vs Paxos under Byzantine faults and network partitions, referencing the production deployment key."
    )

    req = ChatRequest(message=complex_prompt)
    resp = await service.chat(req)

    # 4. Check routing and Gemini payload
    assert resp.provider == "gemini"
    assert len(mock_gemini.received_messages) > 0

    # Ensure EVERY message sent to GeminiClient does NOT contain the raw secret key!
    for msg in mock_gemini.received_messages:
        content = msg["content"]
        assert secret_key not in content, (
            f"PRIVACY VIOLATION: Raw sensitive key '{secret_key}' was sent to Gemini in message: {content}"
        )
        if msg["role"] == "system" and "The project OpenAI production key is" in content:
            assert "********" in content or "[CREDENTIAL_REDACTED]" in content


    # 5. CRITICAL PRINCIPLE: Original memory MUST remain stored intact locally!
    reloaded_local = mem_repo.get_memory("mem_secret")
    assert reloaded_local is not None
    assert secret_key in reloaded_local.content, "Original local memory was corrupted or redacted in SQLite!"
