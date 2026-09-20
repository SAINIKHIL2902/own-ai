import os
from app.config.settings import Settings


def test_default_settings():
    settings = Settings()
    assert settings.OLLAMA_BASE_URL.startswith("http://")
    assert settings.OLLAMA_MODEL == "llama3.2:1b"
    assert settings.OLLAMA_TIMEOUT == 120


def test_custom_settings_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11435")
    monkeypatch.setenv("OLLAMA_MODEL", "custom-model:latest")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "60")

    settings = Settings()
    assert settings.OLLAMA_BASE_URL == "http://127.0.0.1:11435"
    assert settings.OLLAMA_MODEL == "custom-model:latest"
    assert settings.OLLAMA_TIMEOUT == 60
