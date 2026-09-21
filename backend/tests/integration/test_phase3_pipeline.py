import json
import pytest
from unittest.mock import AsyncMock, patch

from app.chat.service import ChatService
from app.llm.base import BaseLLMClient
from app.llm.gemini_client import GeminiConnectionError
from app.ollama.models import ChatRequest
from app.router.router import ModelRouter
from app.storage.database import DatabaseManager
from app.storage.repositories import RawRepository


class MockLLMClient(BaseLLMClient):
    def __init__(self, provider: str, response: str = "Mocked LLM completion"):
        self.provider = provider
        self.response = response
        self.last_received_messages = []

    async def generate(self, messages, model=None):
        self.last_received_messages = messages
        return {
            "model": model or f"mock-{self.provider}-model",
            "response": self.response,
            "provider": self.provider,
        }


class FailingLLMClient(BaseLLMClient):
    async def generate(self, messages, model=None):
        raise GeminiConnectionError("Simulated Gemini API connection failure")


@pytest.mark.asyncio
async def test_complete_local_flow(tmp_path):
    db_file = tmp_path / "test_p3_local.db"
    manager = DatabaseManager(db_path=db_file)
    raw = RawRepository(manager=manager)

    mock_ollama = MockLLMClient(provider="local", response="Python is a dynamic programming language.")
    mock_gemini = MockLLMClient(provider="gemini")
    mock_producer = AsyncMock()

    service = ChatService(
        ollama_client=mock_ollama,
        gemini_client=mock_gemini,
        repository=raw,
        producer=mock_producer,
    )

    req = ChatRequest(message="What is Python?")
    resp = await service.chat(req)

    assert resp.provider == "local"
    assert resp.response == "Python is a dynamic programming language."
    assert resp.routing is not None
    assert resp.routing["selected_provider"] == "local"

    # Verify SQLite recorded provider = "local" and preserved message
    asst_msg = raw.get_message_by_id(resp.message_id)
    assert asst_msg is not None
    assert asst_msg["provider"] == "local"
    assert asst_msg["content"] == "Python is a dynamic programming language."


@pytest.mark.asyncio
async def test_complete_gemini_flow_with_strict_privacy_boundary(tmp_path):
    db_file = tmp_path / "test_p3_gemini.db"
    manager = DatabaseManager(db_path=db_file)
    raw = RawRepository(manager=manager)

    mock_ollama = MockLLMClient(provider="local")
    mock_gemini = MockLLMClient(provider="gemini", response="Here is the comprehensive distributed architecture analysis.")
    mock_producer = AsyncMock()

    service = ChatService(
        ollama_client=mock_ollama,
        gemini_client=mock_gemini,
        repository=raw,
        producer=mock_producer,
    )

    # Complex prompt containing secret API key and email
    secret_key = "sk-99887766554433221100aabbccddeeff"
    secret_email = "architect@enterprise.org"
    prompt = (
        f"My internal API key is {secret_key} and email is {secret_email}. "
        "Analyze the formal consistency proofs and failure modes between Multi-Raft active-active "
        "replication vs Paxos under Byzantine faults and network partitions."
    )

    req = ChatRequest(message=prompt)
    resp = await service.chat(req)

    # 1. Routing selected Gemini
    assert resp.provider == "gemini"
    assert resp.routing["selected_provider"] == "gemini"

    # 2. PRIVACY BOUNDARY VERIFICATION:
    # GeminiClient received messages must NOT have raw secrets
    transmitted_messages = mock_gemini.last_received_messages
    transmitted_text = " ".join(m["content"] for m in transmitted_messages)
    assert secret_key not in transmitted_text
    assert secret_email not in transmitted_text
    assert "********" in transmitted_text
    assert "[EMAIL_REDACTED]" in transmitted_text

    # 3. LOCAL STORAGE VERIFICATION:
    # Original user prompt is preserved locally for internal reproducibility
    with manager.get_connection() as conn:
        user_msg = conn.execute("SELECT * FROM messages WHERE role = 'user'").fetchone()
        assert user_msg is not None
        assert secret_key in user_msg["content"]

    # 4. Assistant message recorded with provider = "gemini"
    asst_msg = raw.get_message_by_id(resp.message_id)
    assert asst_msg is not None
    assert asst_msg["provider"] == "gemini"


@pytest.mark.asyncio
async def test_gemini_failure_and_capable_local_fallback(tmp_path):
    db_file = tmp_path / "test_p3_fallback.db"
    manager = DatabaseManager(db_path=db_file)
    raw = RawRepository(manager=manager)

    mock_ollama = MockLLMClient(provider="local", response="Local fallback response.")
    failing_gemini = FailingLLMClient()
    mock_producer = AsyncMock()

    # Router configured with forced gemini selection on a moderately suitable task
    router = ModelRouter()
    service = ChatService(
        ollama_client=mock_ollama,
        gemini_client=failing_gemini,
        router=router,
        repository=raw,
        producer=mock_producer,
    )

    # A prompt where Gemini might be selected or forced, but local suitability is still moderate
    with patch.object(router, "route") as mock_route:
        from app.router.router import RoutingDecision
        mock_route.return_value = RoutingDecision(
            selected_provider="gemini",
            selected_model="gemini-2.5-flash",
            local_suitability=0.65,  # Capable enough for fallback
            routing_confidence=0.50,
            task_type="coding",
            difficulty="medium",
            required_capabilities={"coding": 0.8},
            reason="Uncertain confidence",
            hard_check_passed=True,
        )

        resp = await service.chat(ChatRequest(message="Write a quick Python sort function."))

        # Must record actual responding provider as "local"
        assert resp.provider == "local"
        assert resp.response == "Local fallback response."

        asst_msg = raw.get_message_by_id(resp.message_id)
        assert asst_msg["provider"] == "local"
        assert "fallback" in asst_msg["model"].lower() or asst_msg["model"] == "llama3.2:1b"
