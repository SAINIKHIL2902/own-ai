import json
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.storage.repositories import raw_repo
from app.training.downloader import ModelDownloader
from app.training.gdrive_sync import GoogleDriveSyncManager
from app.training.repository import TrainingRepository


client = TestClient(app)


@pytest.mark.asyncio
async def test_downloader_fetch_and_deploy_model(tmp_path):
    mock_drive = GoogleDriveSyncManager(sync_dir=tmp_path / "gdrive")
    repo = TrainingRepository()
    local_models = tmp_path / "local_models"

    version_id = "v_test_model_001"
    # Create trained model in mock Google Drive
    model_dir = mock_drive.models_dir / version_id
    model_dir.mkdir(parents=True)
    (model_dir / "adapter_config.json").write_text('{"r": 16}')
    (model_dir / "adapter_model.safetensors").write_text("weights")
    (model_dir / "training_complete.json").write_text(f'{{"version_id": "{version_id}", "status": "DONE"}}')

    downloader = ModelDownloader(
        drive_sync=mock_drive,
        repository=repo,
        local_models_dir=local_models,
    )

    # Mock OllamaClient show & generate calls
    with patch("app.training.deployment.OllamaClient.list_models", new_callable=AsyncMock) as mock_list, \
         patch("app.training.deployment.OllamaClient.generate_chat", new_callable=AsyncMock) as mock_chat:
        mock_list.return_value = [f"llama3.2:1b-{version_id}"]
        mock_chat.return_value = {"response": "Smoke test OK"}

        res = await downloader.fetch_and_deploy_model(
            version_id=version_id,
            auto_evaluate=True,
            auto_activate=True,
            mock_eval=True,
        )

        assert res["success"] is True
        assert res["version_id"] == version_id
        assert res["is_active"] is True
        assert (local_models / version_id / "adapter_config.json").exists()


def test_api_colab_export_and_notebook_download():
    user_id = "api_colab_user"
    # Consent
    client.post("/api/training/consent", json={"user_id": user_id, "opted_in": True})

    # Seed conversations
    for i in range(6):
        conv_id = f"conv_colab_{i}"
        raw_repo.save_conversation(conv_id, f"Conv {i}")
        raw_repo.save_message(f"m_u_c_{i}", conv_id, "user", f"Explain caching strategy {i}", "llama3.2:1b", 20.0, user_id=user_id)
        raw_repo.save_message(f"m_a_c_{i}", conv_id, "assistant", f"Strategy {i} improves read latency.", "llama3.2:1b", 50.0, user_id=user_id)

    # 1. Export Colab Unit
    exp_resp = client.post(
        "/api/training/colab/export",
        json={"user_id": user_id, "num_epochs": 2, "sync_to_gdrive": True},
    )
    assert exp_resp.status_code == 200
    data = exp_resp.json()
    assert data["success"] is True
    version_id = data["version_id"]
    assert "colab_notebook_path" in data

    # 2. Download Colab Notebook
    nb_resp = client.get(f"/api/training/colab/notebook/{version_id}")
    assert nb_resp.status_code == 200
    nb_json = nb_resp.json()
    assert "cells" in nb_json

    # 3. List models
    list_resp = client.get("/api/training/colab/models")
    assert list_resp.status_code == 200
    models_data = list_resp.json()
    assert "active_model" in models_data
    assert "baseline_model" in models_data
