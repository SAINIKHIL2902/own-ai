from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
import zipfile

from app.config.settings import PROJECT_ROOT
from app.memory.service import MemoryService, memory_service
from app.storage.repositories import ProfileRepository, RawRepository, profile_repo, raw_repo
from app.training.colab_notebook import generate_colab_notebook_content
from app.training.dataset import DatasetBuilder, dataset_builder
from app.training.gdrive_sync import GoogleDriveSyncManager, gdrive_sync
from app.training.models import RunStatus, TrainingConfig, TrainingExample, TrainingRunRecord
from app.training.repository import TrainingRepository, training_repo
from app.training.sanitizer import TrainingSanitizer, training_sanitizer

logger = logging.getLogger(__name__)

UNITS_BASE_DIR = PROJECT_ROOT / "data" / "training" / "units"


class UnifiedTrainingUnitBuilder:
    """
    Gathers all user training data (conversations, behavioral preferences, memories)
    into a single cohesive unit, generates the Colab training notebook and scripts,
    and synchronizes directly to Google Drive for cloud GPU training.
    """

    def __init__(
        self,
        raw_repository: Optional[RawRepository] = None,
        profile_repository: Optional[ProfileRepository] = None,
        mem_service: Optional[MemoryService] = None,
        train_repository: Optional[TrainingRepository] = None,
        builder: Optional[DatasetBuilder] = None,
        sanitizer: Optional[TrainingSanitizer] = None,
        drive_sync: Optional[GoogleDriveSyncManager] = None,
        units_dir: Optional[Path] = None,
    ):
        self.raw_repo = raw_repository or raw_repo
        self.profile_repo = profile_repository or profile_repo
        self.mem_service = mem_service or memory_service
        self.repo = train_repository or training_repo
        self.dataset_builder = builder or dataset_builder
        self.sanitizer = sanitizer or training_sanitizer
        self.drive_sync = drive_sync or gdrive_sync
        self.units_dir = units_dir or UNITS_BASE_DIR

    def gather_and_build_unit(
        self,
        config: Optional[TrainingConfig] = None,
        user_id: str = "local_user",
        sync_to_gdrive: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes complete gathering workflow:
        1. Checks explicit training consent
        2. Gathers chat history, behavior profile, and memory preferences
        3. Builds clean, sanitized train/val JSONL splits
        4. Packages scripts, Colab notebook, and requirements
        5. Archives into unit_<version_id>.zip
        6. Syncs directly to Google Drive
        7. Records run in model registry
        """
        cfg = config or TrainingConfig()

        # 1. Enforce user consent
        consent = self.repo.get_consent(user_id)
        if not consent or not consent.opted_in:
            raise PermissionError(f"User '{user_id}' has not granted consent for training data collection.")

        # 2. Build dataset files (train.jsonl, val.jsonl, manifest.json)
        ds_manifest = self.dataset_builder.build_dataset(config=cfg, user_id=user_id)
        version_id = ds_manifest["version"]

        # 3. Setup unit directory
        unit_dir = self.units_dir / version_id
        unit_dir.mkdir(parents=True, exist_ok=True)

        # Copy dataset files into unit directory
        shutil.copy2(ds_manifest["train_path"], unit_dir / "train.jsonl")
        shutil.copy2(ds_manifest["val_path"], unit_dir / "val.jsonl")

        # Gather additional behavioral and memory context into unit
        behavior_ctx = self._gather_behavior_and_memories(user_id)
        with open(unit_dir / "behavioral_context.json", "w", encoding="utf-8") as f:
            json.dump(behavior_ctx, f, indent=2)

        # Copy standalone training script and requirements
        script_src = PROJECT_ROOT / "scripts" / "train_gpu.py"
        if script_src.exists():
            shutil.copy2(script_src, unit_dir / "train_gpu.py")

        reqs_content = "torch\ntransformers\ndatasets\npeft\naccelerate\nbitsandbytes\n"
        with open(unit_dir / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(reqs_content)

        # Generate Colab Notebook (.ipynb)
        nb_data = generate_colab_notebook_content(
            version_id=version_id,
            base_model=cfg.base_model,
            candidate_tag=cfg.candidate_tag,
            epochs=cfg.num_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
        )
        nb_path = unit_dir / "colab_train.ipynb"
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=2)

        # Unit Manifest
        unit_manifest = {
            "version_id": version_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "config": cfg.model_dump(),
            "train_examples": ds_manifest.get("train_examples", 0),
            "val_examples": ds_manifest.get("val_examples", 0),
            "total_examples": ds_manifest.get("total_examples", 0),
            "behavioral_categories": list(behavior_ctx.keys()),
            "files": [
                "train.jsonl",
                "val.jsonl",
                "behavioral_context.json",
                "train_gpu.py",
                "requirements.txt",
                "colab_train.ipynb",
            ],
        }
        with open(unit_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(unit_manifest, f, indent=2)

        # Create zip archive of the unit
        zip_path = self.units_dir / f"unit_{version_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in unit_dir.iterdir():
                if item.is_file():
                    zf.write(item, arcname=item.name)

        # 4. Sync directly to Google Drive
        gdrive_result = {}
        if sync_to_gdrive:
            gdrive_result = self.drive_sync.sync_dataset_unit(
                version_id=version_id,
                unit_dir=unit_dir,
                zip_path=zip_path,
            )

        # 5. Register in model registry
        run_record = TrainingRunRecord(
            run_id=version_id,
            base_model=cfg.base_model,
            dataset_version=version_id,
            status=RunStatus.DATA_VALIDATED.value,
            config_json=json.dumps(cfg.model_dump()),
            candidate_tag=cfg.candidate_tag,
            metrics_json=json.dumps(unit_manifest),
            artifact_path=str(zip_path),
        )
        self.repo.save_run(run_record)

        logger.info(f"Unified training unit '{version_id}' created and ready for Colab.")
        return {
            "success": True,
            "version_id": version_id,
            "candidate_tag": cfg.candidate_tag,
            "unit_dir": str(unit_dir),
            "zip_path": str(zip_path),
            "colab_notebook_path": str(nb_path),
            "gdrive_sync": gdrive_result,
            "manifest": unit_manifest,
        }

    def _gather_behavior_and_memories(self, user_id: str) -> Dict[str, Any]:
        """Compiles clean preferences and memories into behavioral context."""
        context: Dict[str, Any] = {
            "preferences": {},
            "interests": {},
            "memories": [],
        }

        # 1. Behavioral Preferences
        try:
            prefs = self.profile_repo.get_all_preferences()
            for rec in prefs:
                key = rec.get("preference")
                if key:
                    context["preferences"][key] = {
                        "value": rec.get("value"),
                        "confidence": rec.get("confidence", 0.0),
                    }
        except Exception as exc:
            logger.warning(f"Error fetching preferences for unit: {exc}")

        # 2. Interests
        try:
            interests = self.profile_repo.get_all_interests()
            for rec in interests:
                topic = rec.get("topic")
                if topic:
                    context["interests"][topic] = {
                        "score": rec.get("score", 0.0),
                        "frequency": rec.get("frequency", 0),
                    }
        except Exception as exc:
            logger.warning(f"Error fetching interests for unit: {exc}")

        # 3. Active Memories
        try:
            mems = self.mem_service.list_memories(user_id=user_id, status="active")
            for m in mems:
                clean_content = self.sanitizer.redact_pii(m.content)
                context["memories"].append({
                    "type": m.memory_type,
                    "content": clean_content,
                    "importance": m.importance,
                })
        except Exception as exc:
            logger.warning(f"Error fetching memories for unit: {exc}")

        return context


unified_unit_builder = UnifiedTrainingUnitBuilder()
