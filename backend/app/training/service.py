import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config.settings import settings
from app.training.dataset import DatasetBuilder, dataset_builder
from app.training.deployment import ModelDeploymentManager, model_deployment_manager
from app.training.evaluation import ModelEvaluator, model_evaluator
from app.training.downloader import ModelDownloader, model_downloader
from app.training.gdrive_sync import GoogleDriveSyncManager, gdrive_sync
from app.training.models import RunStatus, TrainingConfig, TrainingConsentRecord, TrainingRunRecord
from app.training.packaging import OllamaPackager, ollama_packager
from app.training.postgres_sync import PostgresSyncManager, postgres_sync
from app.training.repository import TrainingRepository, training_repo
from app.training.runner import TrainingRunner, training_runner
from app.training.unit_builder import UnifiedTrainingUnitBuilder, unified_unit_builder

logger = logging.getLogger(__name__)


class TrainingService:
    """
    Unified application service for Phase 5 behavior-based local model training,
    dataset curation, cloud GPU bundling, and controlled model deployment.
    """

    def __init__(
        self,
        repository: Optional[TrainingRepository] = None,
        builder: Optional[DatasetBuilder] = None,
        runner: Optional[TrainingRunner] = None,
        packager: Optional[OllamaPackager] = None,
        deployment: Optional[ModelDeploymentManager] = None,
        unit_builder: Optional[UnifiedTrainingUnitBuilder] = None,
        downloader: Optional[ModelDownloader] = None,
        pg_sync: Optional[PostgresSyncManager] = None,
    ):
        self.repo = repository or training_repo
        self.builder = builder or dataset_builder
        self.runner = runner or training_runner
        self.packager = packager or ollama_packager
        self.deployment = deployment or model_deployment_manager
        self.unit_builder = unit_builder or unified_unit_builder
        self.downloader = downloader or model_downloader
        self.pg_sync = pg_sync or postgres_sync

    # 1. Consent
    def get_consent(self, user_id: str = "local_user") -> TrainingConsentRecord:
        consent = self.repo.get_consent(user_id)
        if not consent:
            return TrainingConsentRecord(user_id=user_id, opted_in=False, scope="behavior_only")
        return consent

    def set_consent(self, user_id: str = "local_user", opted_in: bool = False) -> TrainingConsentRecord:
        return self.repo.set_consent(user_id=user_id, opted_in=opted_in)

    # 2. Dataset Preparation
    def prepare_dataset(
        self,
        config: Optional[TrainingConfig] = None,
        user_id: str = "local_user",
    ) -> Dict[str, Any]:
        cfg = config or TrainingConfig()
        return self.builder.build_dataset(config=cfg, user_id=user_id)

    # 3. Training Run & Cloud Bundle
    def initiate_training_run(
        self,
        dataset_version: str,
        config: Optional[TrainingConfig] = None,
        user_id: str = "local_user",
    ) -> Dict[str, Any]:
        cfg = config or TrainingConfig()
        run = self.runner.create_run(config=cfg, dataset_version=dataset_version, user_id=user_id)

        if run.status == RunStatus.FAILED.value:
            return {
                "success": False,
                "run_id": run.run_id,
                "error": run.error_message,
            }

        bundle_path = self.runner.prepare_cloud_bundle(run.run_id)
        return {
            "success": True,
            "run_id": run.run_id,
            "status": RunStatus.TRAINING.value,
            "candidate_tag": run.candidate_tag,
            "cloud_bundle_path": str(bundle_path),
            "instructions": "Upload cloud_bundle.zip to Google Colab with free T4 GPU to train.",
        }

    # 4. Evaluation & Approval
    async def evaluate_candidate(
        self,
        run_id: str,
        adapter_path: Optional[str] = None,
        mock_eval: bool = True,
    ) -> Dict[str, Any]:
        run = await self.runner.complete_and_evaluate_run(
            run_id=run_id,
            adapter_path=adapter_path or "trained_adapter",
            mock_eval=mock_eval,
        )
        return {
            "run_id": run.run_id,
            "status": run.status,
            "eval_results": run.to_dict().get("eval_results", {}),
            "rejection_reason": run.rejection_reason,
        }

    # 5. Packaging
    async def package_candidate(self, run_id: str) -> Dict[str, Any]:
        success, msg, details = await self.packager.package_and_register(run_id=run_id)
        return {"success": success, "message": msg, "details": details}

    # 6. Deployment & Rollback
    async def activate_model(self, run_id: str) -> Dict[str, Any]:
        success, msg, details = await self.deployment.activate_candidate(run_id=run_id)
        return {"success": success, "message": msg, "details": details}

    def rollback_model(self) -> Dict[str, Any]:
        return self.deployment.rollback()

    # 7. Status & Readiness
    def get_status(self) -> Dict[str, Any]:
        active_model = self.deployment.get_current_model()
        runs = self.repo.list_runs(limit=10)
        consent = self.get_consent()
        return {
            "active_model": active_model,
            "baseline_model": settings.OLLAMA_MODEL,
            "training_consent": consent.to_dict(),
            "total_registered_runs": len(runs),
            "recent_runs": [r.to_dict() for r in runs[:5]],
        }

    # 8. Unified Google Colab & Google Drive Workflow
    def export_colab_unit(
        self,
        config: Optional[TrainingConfig] = None,
        user_id: str = "local_user",
        sync_to_gdrive: bool = True,
    ) -> Dict[str, Any]:
        """
        Gathers all interactions, preferences, and memories into one cohesive unit,
        generates the Colab training notebook, and syncs directly to Google Drive.
        """
        return self.unit_builder.gather_and_build_unit(
            config=config,
            user_id=user_id,
            sync_to_gdrive=sync_to_gdrive,
        )

    async def import_colab_model(
        self,
        version_id: Optional[str] = None,
        auto_evaluate: bool = True,
        auto_activate: bool = True,
        mock_eval: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieves the latest or specified trained model checkpoint from Google Drive,
        evaluates against the fixed benchmark, packages with Ollama, and activates.
        """
        return await self.downloader.fetch_and_deploy_model(
            version_id=version_id,
            auto_evaluate=auto_evaluate,
            auto_activate=auto_activate,
            mock_eval=mock_eval,
        )

    def list_colab_models(self) -> Dict[str, Any]:
        """Lists all versioned models available in Google Drive and local registry."""
        return self.downloader.list_all_models()

    # 6. Cloud PostgreSQL 15-Day Sync & 60-Day Training Pipeline
    async def sync_cloud_delta(self, user_id: str = "local_user", force: bool = False) -> Dict[str, Any]:
        """Runs the 15-day delta synchronization to online PostgreSQL."""
        return await self.pg_sync.sync_15_day_batch(user_id=user_id, force=force)

    def get_cloud_sync_status(self, user_id: str = "local_user") -> Dict[str, Any]:
        """Checks the 15-day sync progress and 60-day training readiness."""
        return self.pg_sync.get_status(user_id=user_id)

    def prepare_60_day_training(self, user_id: str = "local_user", force: bool = False) -> Dict[str, Any]:
        """Compiles the 60-day cumulative dataset and creates the GPU training package."""
        return self.pg_sync.prepare_60_day_training_unit(user_id=user_id, force=force)


training_service = TrainingService()


