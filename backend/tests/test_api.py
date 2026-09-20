from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app
from app.ollama.client import OllamaConnectionError, OllamaModelNotFoundError

client = TestClient(app)


def test_get_health_healthy():
    with patch("app.main.ollama_client.check_health", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = True
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "ollama_connected": True}


def test_get_health_degraded():
    with patch("app.main.ollama_client.check_health", new_callable=AsyncMock) as mock_health:
        mock_health.return_value = False
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "degraded", "ollama_connected": False}


def test_get_models_success():
    with patch("app.main.ollama_client.list_models", new_callable=AsyncMock) as mock_models:
        mock_models.return_value = ["llama3.2:1b"]
        response = client.get("/models")
        assert response.status_code == 200
        assert response.json() == {"models": ["llama3.2:1b"]}


def test_post_chat_success():
    with patch("app.main.ollama_client.generate_chat", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {"model": "llama3.2:1b", "response": "Python is a programming language."}
        response = client.post("/chat", json={"message": "What is Python?"})
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "llama3.2:1b"
        assert data["response"] == "Python is a programming language."


def test_post_chat_empty_message_validation_failure():
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_post_chat_ollama_not_running():
    with patch("app.main.ollama_client.generate_chat", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = OllamaConnectionError("Ollama is not running at http://127.0.0.1:11434")
        response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 503
        assert "Ollama is not running" in response.json()["detail"]


def test_post_chat_model_missing():
    with patch("app.main.ollama_client.generate_chat", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = OllamaModelNotFoundError("Model 'llama3.2:1b' is not installed.")
        response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 404
        assert "not installed" in response.json()["detail"]
