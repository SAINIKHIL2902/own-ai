import json
from pathlib import Path
import pytest
from app.memory.models import MemoryCreateRequest
from app.memory.service import memory_service
from app.storage.repositories import profile_repo, raw_repo
from app.training.gdrive_sync import GoogleDriveSyncManager
from app.training.models import TrainingConfig
from app.training.repository import TrainingRepository
from app.training.unit_builder import UnifiedTrainingUnitBuilder


def test_unit_builder_consent_enforcement():
    repo = TrainingRepository()
    user_id = "unit_unconsented_user"
    repo.set_consent(user_id, opted_in=False)

    builder = UnifiedTrainingUnitBuilder(train_repository=repo)
    with pytest.raises(PermissionError) as exc_info:
        builder.gather_and_build_unit(user_id=user_id)

    assert "has not granted consent" in str(exc_info.value)


def test_unit_builder_end_to_end(tmp_path):
    repo = TrainingRepository()
    user_id = "unit_builder_user"
    repo.set_consent(user_id, opted_in=True)

    # 1. Seed conversations
    for i in range(6):
        conv_id = f"conv_unit_{i}"
        raw_repo.save_conversation(conv_id, f"Unit Conv {i}")
        raw_repo.save_message(f"mu_{i}", conv_id, "user", f"Explain distributed consensus concept {i}", "llama3.2:1b", 20.0, user_id=user_id)
        raw_repo.save_message(f"ma_{i}", conv_id, "assistant", f"Consensus algorithm {i} ensures fault-tolerant agreement.", "llama3.2:1b", 50.0, user_id=user_id)

    # 2. Seed memory
    memory_service.create_memory_manual(
        MemoryCreateRequest(
            memory_type="preference",
            content="Always provide concise code samples in Python.",
            importance=0.9,
        ),
        user_id=user_id,
    )

    # 3. Setup mock Google Drive sync
    mock_drive = GoogleDriveSyncManager(sync_dir=tmp_path / "gdrive")
    unit_storage = tmp_path / "units"

    builder = UnifiedTrainingUnitBuilder(
        train_repository=repo,
        drive_sync=mock_drive,
        units_dir=unit_storage,
    )

    config = TrainingConfig(num_epochs=2, batch_size=2)
    result = builder.gather_and_build_unit(config=config, user_id=user_id, sync_to_gdrive=True)

    assert result["success"] is True
    version_id = result["version_id"]
    unit_dir = Path(result["unit_dir"])
    assert unit_dir.exists()

    # Verify unit contents
    assert (unit_dir / "train.jsonl").exists()
    assert (unit_dir / "val.jsonl").exists()
    assert (unit_dir / "behavioral_context.json").exists()
    assert (unit_dir / "colab_train.ipynb").exists()
    assert (unit_dir / "manifest.json").exists()
    assert Path(result["zip_path"]).exists()

    # Validate Colab Notebook format
    with open(unit_dir / "colab_train.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
        assert "cells" in nb
        assert len(nb["cells"]) >= 6
        sources = "".join("".join(c["source"]) for c in nb["cells"])
        assert "drive.mount" in sources
        assert "train_gpu.py" in sources
        assert version_id in sources

    # Verify Google Drive sync
    assert (mock_drive.datasets_dir / version_id / "colab_train.ipynb").exists()
    assert (mock_drive.datasets_dir / version_id / f"unit_{version_id}.zip").exists()

    # Verify run record in repository
    run = repo.get_run(version_id)
    assert run is not None
    assert run.dataset_version == version_id
