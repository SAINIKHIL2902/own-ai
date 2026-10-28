import glob
import json
import logging
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple
import zipfile

from app.config.settings import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)


class GoogleDriveSyncManager:
    """
    Manages direct synchronization between own-ai local storage and Google Drive.
    Enables unified dataset delivery to Google Colab and automated retrieval of
    fine-tuned model checkpoints labeled by version ID.
    """

    def __init__(self, sync_dir: Optional[Path] = None):
        self._custom_sync_dir = sync_dir

    def resolve_drive_root(self) -> Path:
        """
        Locates Google Drive root folder. Checks:
        1. Custom explicit sync directory or settings.resolved_google_drive_sync_dir
        2. Google Drive Desktop macOS CloudStorage paths (~/Library/CloudStorage/GoogleDrive-*)
        3. Traditional ~/Google Drive/My Drive
        """
        if self._custom_sync_dir:
            self._custom_sync_dir.mkdir(parents=True, exist_ok=True)
            return self._custom_sync_dir

        # Check for Google Drive for Desktop on macOS
        cloud_storage = Path.home() / "Library" / "CloudStorage"
        if cloud_storage.exists():
            matches = list(cloud_storage.glob("GoogleDrive-*"))
            for match in matches:
                my_drive = match / "My Drive"
                if my_drive.exists():
                    logger.info(f"Detected Google Drive for Desktop at: {my_drive}")
                    return my_drive
                elif match.is_dir():
                    return match

        # Traditional Google Drive directory
        trad_path = Path.home() / "Google Drive" / "My Drive"
        if trad_path.exists():
            return trad_path

        # Staging fallback in project directory
        default_sync = settings.resolved_google_drive_sync_dir
        default_sync.mkdir(parents=True, exist_ok=True)
        return default_sync

    @property
    def base_app_dir(self) -> Path:
        base = self.resolve_drive_root() / "own-ai"
        base.mkdir(parents=True, exist_ok=True)
        return base

    @property
    def datasets_dir(self) -> Path:
        d = self.base_app_dir / "datasets"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def models_dir(self) -> Path:
        m = self.base_app_dir / "models"
        m.mkdir(parents=True, exist_ok=True)
        return m

    def sync_dataset_unit(
        self,
        version_id: str,
        unit_dir: Path,
        zip_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Syncs a unified training unit (dataset files, scripts, notebook) directly to Google Drive.
        """
        target_version_dir = self.datasets_dir / version_id
        target_version_dir.mkdir(parents=True, exist_ok=True)

        copied_files = []
        # Copy directory files
        if unit_dir.exists():
            for item in unit_dir.iterdir():
                if item.is_file():
                    dest = target_version_dir / item.name
                    shutil.copy2(item, dest)
                    copied_files.append(item.name)

        # Copy archive bundle if provided
        target_zip = None
        if zip_path and zip_path.exists():
            target_zip = target_version_dir / f"unit_{version_id}.zip"
            shutil.copy2(zip_path, target_zip)
            copied_files.append(target_zip.name)

        logger.info(f"Successfully synced dataset unit '{version_id}' to Google Drive at {target_version_dir}")
        return {
            "success": True,
            "version_id": version_id,
            "drive_datasets_path": str(target_version_dir),
            "files_synced": copied_files,
            "zip_path": str(target_zip) if target_zip else None,
        }

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Scans Google Drive models folder for trained model artifacts.
        Returns list of models sorted by modification time (most recent first).
        """
        models = []
        if not self.models_dir.exists():
            return []

        for item in self.models_dir.iterdir():
            # Check directory-based model
            if item.is_dir():
                version_id = item.name
                metadata_file = item / "training_complete.json"
                metadata = {}
                if metadata_file.exists():
                    try:
                        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                
                # Check for adapter files
                has_weights = (item / "adapter_model.safetensors").exists() or (item / "adapter_model.bin").exists() or (item / "adapter_config.json").exists()

                models.append({
                    "version_id": version_id,
                    "path": str(item),
                    "is_archive": False,
                    "has_weights": has_weights,
                    "updated_at": item.stat().st_mtime,
                    "metadata": metadata,
                })
            # Check zip-based model (e.g. model_v_xxx.zip)
            elif item.is_file() and item.suffix == ".zip":
                version_id = item.stem.replace("model_", "")
                models.append({
                    "version_id": version_id,
                    "path": str(item),
                    "is_archive": True,
                    "has_weights": True,
                    "updated_at": item.stat().st_mtime,
                    "metadata": {},
                })

        # Also check ~/Downloads and data/training/models for downloaded zip archives
        extra_search_dirs = [
            Path.home() / "Downloads",
            PROJECT_ROOT / "data" / "training" / "models",
        ]
        for extra_dir in extra_search_dirs:
            if extra_dir.exists():
                for item in extra_dir.glob("model_*.zip"):
                    raw_version = item.stem.replace("model_", "")
                    version_id = re.sub(r"\s*\(\d+\)$", "", raw_version).strip()
                    version_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", version_id)
                    if not any(m["version_id"] == version_id for m in models):
                        models.append({
                            "version_id": version_id,
                            "path": str(item),
                            "is_archive": True,
                            "has_weights": True,
                            "updated_at": item.stat().st_mtime,
                            "metadata": {},
                        })

        # Sort newest first
        models.sort(key=lambda x: x["updated_at"], reverse=True)
        return models

    def get_latest_model(self) -> Optional[Dict[str, Any]]:
        models = self.list_available_models()
        return models[0] if models else None

    def retrieve_model(self, version_id: str, dest_dir: Path) -> Tuple[bool, Path, str]:
        """
        Retrieves a trained model artifact from Google Drive, ~/Downloads, or data/training/models
        and unpacks it into dest_dir.
        """
        # 1. Search directory model in models_dir
        target_dir = self.models_dir / version_id
        if target_dir.exists() and target_dir.is_dir():
            dest_dir.mkdir(parents=True, exist_ok=True)
            for file_path in target_dir.iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, dest_dir / file_path.name)
                elif file_path.is_dir():
                    shutil.copytree(file_path, dest_dir / file_path.name, dirs_exist_ok=True)
            return True, dest_dir, "Successfully imported directory from Google Drive."

        # 2. Search zip archive model across models_dir, ~/Downloads, and data/training/models
        possible_search_dirs = [
            self.models_dir,
            Path.home() / "Downloads",
            PROJECT_ROOT / "data" / "training" / "models",
        ]
        possible_zip_names = [
            f"{version_id}.zip",
            f"model_{version_id}.zip",
        ]
        for search_dir in possible_search_dirs:
            if not search_dir.exists():
                continue
            for zname in possible_zip_names:
                pz = search_dir / zname
                if pz.exists() and pz.is_file():
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(pz, "r") as zf:
                        zf.extractall(dest_dir)
                    return True, dest_dir, f"Successfully extracted archive {pz.name} from {search_dir}."

        return False, dest_dir, f"Model version '{version_id}' not found in Google Drive ({self.models_dir}), ~/Downloads, or data/training/models."


gdrive_sync = GoogleDriveSyncManager()
