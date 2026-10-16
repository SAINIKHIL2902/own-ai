import json
from pathlib import Path
import pytest
import zipfile

from app.training.gdrive_sync import GoogleDriveSyncManager


def test_gdrive_sync_custom_root(tmp_path):
    sync_root = tmp_path / "mock_drive"
    manager = GoogleDriveSyncManager(sync_dir=sync_root)

    assert manager.resolve_drive_root() == sync_root
    assert manager.datasets_dir == sync_root / "own-ai" / "datasets"
    assert manager.models_dir == sync_root / "own-ai" / "models"


def test_sync_dataset_unit_and_retrieve_model(tmp_path):
    sync_root = tmp_path / "mock_drive"
    manager = GoogleDriveSyncManager(sync_dir=sync_root)

    # 1. Prepare dummy unit
    local_unit = tmp_path / "local_unit"
    local_unit.mkdir()
    (local_unit / "train.jsonl").write_text('{"messages": []}\n')
    (local_unit / "manifest.json").write_text('{"version": "v_test_123"}')

    zip_path = tmp_path / "unit_v_test_123.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(local_unit / "train.jsonl", arcname="train.jsonl")

    res = manager.sync_dataset_unit(
        version_id="v_test_123",
        unit_dir=local_unit,
        zip_path=zip_path,
    )
    assert res["success"] is True
    assert (manager.datasets_dir / "v_test_123" / "train.jsonl").exists()
    assert (manager.datasets_dir / "v_test_123" / "unit_v_test_123.zip").exists()

    # 2. Simulate trained model output in Google Drive
    model_dir = manager.models_dir / "v_test_123"
    model_dir.mkdir(parents=True)
    (model_dir / "adapter_config.json").write_text('{"r": 16}')
    (model_dir / "adapter_model.safetensors").write_text("mock weights")
    (model_dir / "training_complete.json").write_text('{"status": "DONE", "version_id": "v_test_123"}')

    models = manager.list_available_models()
    assert len(models) == 1
    assert models[0]["version_id"] == "v_test_123"
    assert models[0]["has_weights"] is True

    latest = manager.get_latest_model()
    assert latest is not None
    assert latest["version_id"] == "v_test_123"

    # 3. Retrieve model into destination
    import_dest = tmp_path / "imported_model"
    success, path, msg = manager.retrieve_model("v_test_123", import_dest)
    assert success is True
    assert (import_dest / "adapter_config.json").exists()
    assert (import_dest / "training_complete.json").exists()
