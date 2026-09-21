from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import pytest

from app.behavior.analyzer import behavior_analyzer
from app.main import app
from app.storage.repositories import profile_repo, raw_repo


@pytest.fixture
def client():
    return TestClient(app)


def test_end_to_end_behavior_pipeline(client):
    # 1. Send chat request with mock Ollama client
    with patch("app.main.ollama_client.generate_chat", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {
            "model": "llama3.2:1b",
            "response": "Here is a step-by-step Python Kafka consumer implementation:\n```python\nprint('hello')\n```",
        }
        res = client.post(
            "/chat",
            json={"message": "Can you give me a step-by-step Python script for a Kafka consumer?"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "response" in data

    # 2. Verify raw events were stored in SQLite
    events_count = raw_repo.count_raw_events()
    assert events_count >= 1

    # 3. Simulate processing any spooled events through the behavior analyzer
    behavior_analyzer.drain_local_spool()

    # 4. Verify profile was derived
    profile_res = client.get("/profile")
    assert profile_res.status_code == 200
    profile_data = profile_res.json()

    # Check that Python or Kafka was detected in interests
    topics = [item["topic"] for item in profile_data.get("interests", [])]
    assert any(t in ["Python", "Kafka"] for t in topics)

    # 5. Submit user feedback
    fb_res = client.post(
        "/feedback",
        json={
            "message_id": "msg_test_fb",
            "feedback_type": "positive",
            "rating": 5,
            "comment": "Very helpful code sample!",
        },
    )
    assert fb_res.status_code == 200

    # 6. Test dataset export
    export_res = client.post("/analytics/export")
    assert export_res.status_code == 200
    export_data = export_res.json()
    assert export_data["status"] == "ok"
    assert export_data["records_exported"] >= 1
