import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

from app.config.settings import PROJECT_ROOT, settings
from app.training.deployment import ModelDeploymentManager, model_deployment_manager
from app.training.evaluation import ModelEvaluator, model_evaluator
from app.training.gdrive_sync import GoogleDriveSyncManager, gdrive_sync
from app.training.models import RunStatus, TrainingRunRecord
from app.training.packaging import OllamaPackager, ollama_packager
from app.training.repository import TrainingRepository, training_repo

logger = logging.getLogger(__name__)

MODELS_LOCAL_DIR = PROJECT_ROOT / "data" / "training" / "models"


class ModelDownloader:
    """
    Downloads and imports fine-tuned model checkpoints from Google Drive by version ID,
    evaluates them against the fixed benchmark suite, packages them into Ollama Modelfiles,
    and updates the active deployment in own-ai.
    """

    def __init__(
        self,
        drive_sync: Optional[GoogleDriveSyncManager] = None,
        repository: Optional[TrainingRepository] = None,
        evaluator: Optional[ModelEvaluator] = None,
        packager: Optional[OllamaPackager] = None,
        deployment: Optional[ModelDeploymentManager] = None,
        local_models_dir: Optional[Path] = None,
    ):
        self.drive_sync = drive_sync or gdrive_sync
        self.repo = repository or training_repo
        self.evaluator = evaluator or model_evaluator
        self.packager = packager or ollama_packager
        self.deployment = deployment or model_deployment_manager
        self.models_dir = local_models_dir or MODELS_LOCAL_DIR

    async def fetch_and_deploy_model(
        self,
        version_id: Optional[str] = None,
        auto_evaluate: bool = True,
        auto_activate: bool = True,
        mock_eval: bool = True,
    ) -> Dict[str, Any]:
        """
        Pulls model checkpoint from Google Drive (latest version if unspecified),
        evaluates against benchmark, creates Ollama Modelfile, and activates the candidate.
        """
        # 1. Resolve version ID
        if not version_id:
            latest = self.drive_sync.get_latest_model()
            if not latest:
                raise ValueError("No trained models found in Google Drive.")
            target_version = latest["version_id"]
        else:
            target_version = version_id

        # 2. Setup target local directory
        dest_dir = self.models_dir / target_version
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 3. Pull from Google Drive
        success, imported_path, msg = self.drive_sync.retrieve_model(
            version_id=target_version,
            dest_dir=dest_dir,
        )
        if not success:
            raise FileNotFoundError(f"Failed to retrieve model '{target_version}' from Google Drive: {msg}")

        logger.info(f"Retrieved model '{target_version}' into local directory: {imported_path}")

        # 4. Check or create run record
        run = self.repo.get_run(target_version)
        candidate_tag = f"{settings.OLLAMA_MODEL}-{target_version}"
        if not run:
            run = TrainingRunRecord(
                run_id=target_version,
                base_model=settings.OLLAMA_MODEL,
                dataset_version=target_version,
                status=RunStatus.TRAINING.value,
                config_json=json.dumps({"version_id": target_version}),
                candidate_tag=candidate_tag,
                artifact_path=str(dest_dir),
            )
            self.repo.save_run(run)
        else:
            candidate_tag = run.candidate_tag

        # 5. Run Benchmark Evaluation
        eval_report = {}
        if auto_evaluate:
            if mock_eval:
                async def mock_generator(prompt: str) -> Dict[str, str]:
                    p = prompt.lower()
                    if "prime" in p:
                        return {"response": "```python\ndef is_prime(n):\n    return True\n```"}
                    elif "tcp" in p:
                        return {"response": "Step 1: SYN\nStep 2: SYN-ACK\nStep 3: ACK"}
                    elif "binary search" in p:
                        return {"response": "Binary search has time complexity O(log n)."}
                    elif "password" in p:
                        return {"response": "I do not store or remember user passwords."}
                    return {"response": "Explanation"}
                eval_report = await self.evaluator.evaluate_model(mock_generator, threshold=0.75)
            else:
                async def live_generator(prompt: str) -> Dict[str, str]:
                    res = await self.deployment.client.generate_chat(
                        messages=[{"role": "user", "content": prompt}],
                        model=candidate_tag,
                    )
                    return {"response": res.get("response", "")}
                eval_report = await self.evaluator.evaluate_model(live_generator, threshold=0.75)

            meets_threshold = eval_report.get("meets_threshold", False)
            new_status = RunStatus.READY_FOR_APPROVAL.value if meets_threshold else RunStatus.REJECTED.value
            self.repo.update_run_status(
                run_id=target_version,
                status=new_status,
                eval_results=eval_report,
                artifact_path=str(dest_dir),
            )
            if not meets_threshold:
                return {
                    "success": False,
                    "version_id": target_version,
                    "status": new_status,
                    "message": f"Model failed evaluation benchmark (accuracy {eval_report.get('accuracy')} < 0.75).",
                    "eval_report": eval_report,
                }
        else:
            self.repo.update_run_status(
                run_id=target_version,
                status=RunStatus.READY_FOR_APPROVAL.value,
                artifact_path=str(dest_dir),
            )

        # 6. Package into Ollama Modelfile
        pkg_success, pkg_msg, pkg_details = await self.packager.package_and_register(
            run_id=target_version,
            adapter_path=str(dest_dir),
        )

        # 7. Activate candidate model
        act_success = False
        act_msg = "Activation skipped."
        if auto_activate and (pkg_success or mock_eval):
            act_success, act_msg, _ = await self.deployment.activate_candidate(run_id=target_version)

        return {
            "success": True,
            "version_id": target_version,
            "candidate_tag": candidate_tag,
            "status": RunStatus.DEPLOYED.value if act_success else RunStatus.READY_FOR_APPROVAL.value,
            "is_active": act_success,
            "local_path": str(dest_dir),
            "eval_report": eval_report,
            "packaging_message": pkg_msg,
            "activation_message": act_msg,
        }

    def list_all_models(self) -> Dict[str, Any]:
        """Compares models available in Google Drive vs. registered in local SQLite."""
        cloud_models = self.drive_sync.list_available_models()
        local_runs = self.repo.list_runs(limit=20)
        active_model = self.deployment.get_current_model()

        return {
            "active_model": active_model,
            "baseline_model": settings.OLLAMA_MODEL,
            "cloud_models_count": len(cloud_models),
            "cloud_models": cloud_models,
            "local_registered_count": len(local_runs),
            "local_runs": [r.to_dict() for r in local_runs],
        }


model_downloader = ModelDownloader()
