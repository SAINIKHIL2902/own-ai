import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from app.main import app
from app.storage.repositories import raw_repo


client = TestClient(app)


def test_api_consent_flow():
    import uuid
    user_id = f"api_user_{uuid.uuid4().hex[:8]}"

    # 1. Check initial consent (default false)
    resp = client.get(f"/api/training/consent?user_id={user_id}")
    assert resp.status_code == 200
    assert resp.json()["opted_in"] is False

    # 2. Grant consent
    resp = client.post(
        "/api/training/consent",
        json={"user_id": user_id, "opted_in": True},
    )
    assert resp.status_code == 200
    assert resp.json()["opted_in"] is True

    # 3. Verify consent persisted
    resp = client.get(f"/api/training/consent?user_id={user_id}")
    assert resp.status_code == 200
    assert resp.json()["opted_in"] is True


def test_api_dataset_and_training_run():
    user_id = "api_train_user"
    # Grant consent
    client.post("/api/training/consent", json={"user_id": user_id, "opted_in": True})

    # Seed at least 6 conversations
    for i in range(6):
        conv_id = f"conv_api_seed_{i}"
        raw_repo.save_conversation(conv_id, f"Conv {i}")
        raw_repo.save_message(f"m_u_{i}", conv_id, "user", f"How do I use feature {i} effectively?", "llama3.2:1b", 20.0, user_id=user_id)
        raw_repo.save_message(f"m_a_{i}", conv_id, "assistant", f"Feature {i} can be configured through settings.", "llama3.2:1b", 50.0, user_id=user_id)

    # Build dataset
    resp = client.post(
        "/api/training/dataset",
        json={"user_id": user_id, "train_val_split": 0.8},
    )
    assert resp.status_code == 200
    manifest = resp.json()
    assert "version" in manifest
    assert manifest["train_examples"] > 0
    ds_version = manifest["version"]

    # Create training run
    resp = client.post(
        "/api/training/runs",
        json={
            "dataset_version": ds_version,
            "user_id": user_id,
            "num_epochs": 2,
            "batch_size": 2,
            "candidate_tag": "llama3.2:1b-api-test",
        },
    )
    assert resp.status_code == 200
    run_data = resp.json()
    assert run_data["success"] is True
    assert "cloud_bundle_path" in run_data
    run_id = run_data["run_id"]

    # List runs
    resp = client.get("/api/training/runs")
    assert resp.status_code == 200
    runs = resp.json()
    assert any(r["run_id"] == run_id for r in runs)

    # Get single run
    resp = client.get(f"/api/training/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["run_id"] == run_id


def test_api_evaluate_activate_rollback():
    user_id = "eval_api_user"
    client.post("/api/training/consent", json={"user_id": user_id, "opted_in": True})

    # Seed data & create run
    for i in range(6):
        conv_id = f"conv_eval_seed_{i}"
        raw_repo.save_conversation(conv_id, f"Conv {i}")
        raw_repo.save_message(f"m_u_e_{i}", conv_id, "user", f"Explain query optimization technique {i}", "llama3.2:1b", 20.0, user_id=user_id)
        raw_repo.save_message(f"m_a_e_{i}", conv_id, "assistant", f"Technique {i} indexes proper columns.", "llama3.2:1b", 50.0, user_id=user_id)

    ds_resp = client.post("/api/training/dataset", json={"user_id": user_id})
    ds_version = ds_resp.json()["version"]

    run_resp = client.post(
        "/api/training/runs",
        json={"dataset_version": ds_version, "user_id": user_id, "candidate_tag": "llama3.2:1b-eval-tag"},
    )
    run_id = run_resp.json()["run_id"]

    # Evaluate run
    eval_resp = client.post(
        "/api/training/evaluate",
        json={"run_id": run_id, "mock_eval": True},
    )
    assert eval_resp.status_code == 200
    assert eval_resp.json()["status"] == "READY_FOR_APPROVAL"

    # Activate run with mocked Ollama calls
    with patch("app.training.deployment.OllamaClient.list_models", new_callable=AsyncMock) as mock_list, \
         patch("app.training.deployment.OllamaClient.generate_chat", new_callable=AsyncMock) as mock_chat:
        mock_list.return_value = {"models": [{"name": "llama3.2:1b-eval-tag"}]}
        mock_chat.return_value = {"response": "All systems operational"}

        act_resp = client.post("/api/training/activate", json={"run_id": run_id})
        assert act_resp.status_code == 200
        assert act_resp.json()["success"] is True

        # Check status endpoint reflects active model
        status_resp = client.get("/api/training/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["active_model"] == "llama3.2:1b-eval-tag"

    # Rollback
    rb_resp = client.post("/api/training/rollback", json={"reason": "Testing rollback endpoint"})
    assert rb_resp.status_code == 200
    assert rb_resp.json()["success"] is True
    assert rb_resp.json()["active_model"] == "llama3.2:1b"
