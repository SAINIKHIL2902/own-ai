import json
from pathlib import Path
import pytest
from app.storage.repositories import raw_repo
from app.training.dataset import DatasetBuilder
from app.training.extractor import ConsentAwareExtractor
from app.training.models import TrainingConfig
from app.training.repository import TrainingRepository


def test_dataset_builder_pipeline():
    repo = TrainingRepository()
    user_id = "test_dataset_builder_user"
    repo.set_consent(user_id, opted_in=True)

    # Seed 6 conversations to exceed min_examples (5)
    for i in range(6):
        conv_id = f"conv_ds_builder_{i}"
        raw_repo.save_conversation(conv_id, f"Dataset Builder Conv {i}")
        raw_repo.save_message(
            f"m_u_{i}", conv_id, "user",
            f"Explain architectural pattern number {i} in software engineering.",
            "llama3.2:1b", 25.0, user_id=user_id
        )
        raw_repo.save_message(
            f"m_a_{i}", conv_id, "assistant",
            f"Pattern number {i} is a reliable architectural structure that promotes decoupling.",
            "llama3.2:1b", 60.0, user_id=user_id
        )

    extractor = ConsentAwareExtractor(training_repository=repo)
    builder = DatasetBuilder(extractor=extractor)
    config = TrainingConfig(
        base_model="llama3.2:1b",
        train_val_split=0.8,
    )

    manifest = builder.build_dataset(config=config, user_id=user_id)

    assert manifest["version"].startswith("v_")
    assert manifest["total_examples"] >= 6
    assert manifest["train_examples"] > 0
    assert manifest["val_examples"] > 0
    assert Path(manifest["train_path"]).exists()
    assert Path(manifest["val_path"]).exists()
    assert Path(manifest["manifest_path"]).exists()

    # Validate JSONL formatting
    with open(manifest["train_path"], "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
        assert len(lines) == manifest["train_examples"]
        for row in lines:
            assert "messages" in row
            assert len(row["messages"]) == 2
            assert row["messages"][0]["role"] == "user"
            assert row["messages"][1]["role"] == "assistant"
