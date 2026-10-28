from pathlib import Path
import zipfile
import pytest
from app.training.models import RunStatus, TrainingConfig
from app.training.packaging import OllamaPackager
from app.training.repository import TrainingRepository
from app.training.runner import TrainingRunner


def test_training_runner_and_cloud_bundle():
    repo = TrainingRepository()
    user_id = "test_bundle_user"
    repo.set_consent(user_id, opted_in=True)

    runner = TrainingRunner(repository=repo)
    # Prepare dummy dataset in runner's builder base directory
    ds_dir = runner.builder.output_base_dir / "dataset_dummy_1"
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "train.jsonl").write_text('{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}\n')
    (ds_dir / "val.jsonl").write_text('{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}\n')
    (ds_dir / "manifest.json").write_text('{"version": "dataset_dummy_1", "train_examples": 1, "val_examples": 1}')

    config = TrainingConfig(
        base_model="llama3.2:1b",
        candidate_tag="llama3.2:1b-test-bundle",
    )

    run = runner.create_run(config=config, dataset_version="dataset_dummy_1", user_id=user_id)
    assert run.run_id.startswith("run_")
    assert run.status == RunStatus.DATA_VALIDATED.value

    # Generate cloud bundle
    bundle_path = runner.prepare_cloud_bundle(run.run_id)
    assert bundle_path.exists()
    assert bundle_path.suffix == ".zip"

    # Inspect zip contents
    with zipfile.ZipFile(bundle_path, "r") as zf:
        namelist = zf.namelist()
        assert "train.jsonl" in namelist
        assert "val.jsonl" in namelist
        assert "train_gpu.py" in namelist
        assert "README_COLAB.md" in namelist


def test_ollama_packager_modelfile():
    packager = OllamaPackager()
    modelfile = packager.generate_modelfile(
        base_model="llama3.2:1b",
        system_prompt="You are a personal AI assistant tailored to user preferences.",
        adapter_path="adapters/run_123",
    )

    assert "FROM llama3.2:1b" in modelfile
    assert "ADAPTER adapters/run_123" in modelfile
    assert "SYSTEM" in modelfile
    assert "PARAMETER temperature" in modelfile
