import json
import pytest
from unittest.mock import AsyncMock, patch
from app.config.settings import settings
from app.training.deployment import ModelDeploymentManager
from app.training.models import RunStatus, TrainingConfig, TrainingRunRecord
from app.training.repository import TrainingRepository


@pytest.mark.asyncio
async def test_deployment_activation_and_rollback():
    repo = TrainingRepository()
    manager = ModelDeploymentManager(repository=repo)

    # 1. Ensure active model is reset to baseline
    manager.rollback()
    assert manager.get_current_model() == settings.OLLAMA_MODEL

    # 2. Register run and set ready for approval
    cfg = TrainingConfig(candidate_tag="llama3.2:1b-test-active")
    record = TrainingRunRecord(
        run_id="run_deploy_test",
        base_model="llama3.2:1b",
        dataset_version="ds_v1",
        status=RunStatus.READY_FOR_APPROVAL.value,
        config_json=json.dumps(cfg.model_dump()),
        candidate_tag="llama3.2:1b-test-active",
    )
    repo.save_run(record)

    # Mock Ollama health & show calls
    with patch("app.training.deployment.OllamaClient.list_models", new_callable=AsyncMock) as mock_list, \
         patch("app.training.deployment.OllamaClient.generate_chat", new_callable=AsyncMock) as mock_chat:
        mock_list.return_value = ["llama3.2:1b-test-active"]
        mock_chat.return_value = {"response": "OK"}

        success, msg, details = await manager.activate_candidate("run_deploy_test")
        assert success is True
        assert manager.get_current_model() == "llama3.2:1b-test-active"

    # 3. Test Rollback
    rollback_res = manager.rollback()
    assert rollback_res["success"] is True
    assert manager.get_current_model() == settings.OLLAMA_MODEL
