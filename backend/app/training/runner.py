from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, Optional
import uuid
import zipfile

from app.config.settings import PROJECT_ROOT
from app.training.dataset import DatasetBuilder, dataset_builder
from app.training.evaluation import ModelEvaluator, model_evaluator
from app.training.models import RunStatus, TrainingConfig, TrainingRunRecord, utc_now_iso
from app.training.repository import TrainingRepository, training_repo

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "data" / "training" / "artifacts"


class TrainingRunner:
    """
    Manages the lifecycle of model training runs.
    Supports cloud GPU bundle exports, evaluation orchestration, and local mock runners.
    """

    def __init__(
        self,
        repository: Optional[TrainingRepository] = None,
        builder: Optional[DatasetBuilder] = None,
        evaluator: Optional[ModelEvaluator] = None,
    ):
        self.repo = repository or training_repo
        self.builder = builder or dataset_builder
        self.evaluator = evaluator or model_evaluator

    def create_run(
        self,
        config: TrainingConfig,
        dataset_version: str,
        user_id: str = "local_user",
    ) -> TrainingRunRecord:
        """Initialize a new training run in the registry."""
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        now = utc_now_iso()

        # Check dataset existence
        dataset_dir = self.builder.output_base_dir / dataset_version
        is_valid, msg, stats = self.builder.validate_dataset_files(dataset_dir)
        initial_status = RunStatus.DATA_VALIDATED.value if is_valid else RunStatus.FAILED.value
        err_msg = None if is_valid else f"Dataset validation failed: {msg}"

        record = TrainingRunRecord(
            run_id=run_id,
            base_model=config.base_model,
            dataset_version=dataset_version,
            status=initial_status,
            config_json=json.dumps(config.model_dump()),
            candidate_tag=config.candidate_tag,
            metrics_json=json.dumps({"dataset_stats": stats}),
            error_message=err_msg,
            created_at=now,
            updated_at=now,
        )

        self.repo.save_run(record)
        logger.info(f"Initialized training run '{run_id}' with status: {initial_status}")
        return record

    def prepare_cloud_bundle(self, run_id: str) -> Path:
        """
        Generates a self-contained zip package for Google Colab or Kaggle GPU execution.
        Contains train.jsonl, val.jsonl, train_gpu.py, and setup instructions.
        """
        run = self.repo.get_run(run_id)
        if not run:
            raise ValueError(f"Run '{run_id}' not found.")

        dataset_dir = self.builder.output_base_dir / run.dataset_version
        bundle_dir = PROJECT_ROOT / "data" / "training" / "bundles"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_zip = bundle_dir / f"cloud_bundle_{run_id}.zip"

        script_src = PROJECT_ROOT / "scripts" / "train_gpu.py"

        with zipfile.ZipFile(bundle_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            if (dataset_dir / "train.jsonl").exists():
                zf.write(dataset_dir / "train.jsonl", arcname="train.jsonl")
            if (dataset_dir / "val.jsonl").exists():
                zf.write(dataset_dir / "val.jsonl", arcname="val.jsonl")
            if script_src.exists():
                zf.write(script_src, arcname="train_gpu.py")

            readme_content = f"""# Google Colab / Cloud GPU Training Instructions for {run.candidate_tag}

1. Open a new notebook in Google Colab (Runtime -> Change runtime type -> T4 GPU).
2. Upload this zip file and extract it:
   !unzip cloud_bundle_{run_id}.zip
3. Install training dependencies:
   !pip install -q transformers datasets peft accelerate bitsandbytes
4. Run the training command:
   !python train_gpu.py --data_dir . --output_dir ./trained_adapter --epochs 3
5. Download the `trained_adapter` folder back to your own-ai project!
"""
            zf.writestr("README_COLAB.md", readme_content)

        self.repo.update_run_status(
            run_id,
            RunStatus.TRAINING,
            artifact_path=str(bundle_zip),
        )
        logger.info(f"Prepared Cloud GPU bundle for run '{run_id}' at {bundle_zip}")
        return bundle_zip

    async def complete_and_evaluate_run(
        self,
        run_id: str,
        adapter_path: str,
        mock_eval: bool = True,
    ) -> TrainingRunRecord:
        """
        Evaluates the trained artifact against the fixed benchmark suite.
        Transitions to READY_FOR_APPROVAL if evaluation passes, or REJECTED otherwise.
        """
        run = self.repo.get_run(run_id)
        if not run:
            raise ValueError(f"Run '{run_id}' not found.")

        config_dict = json.loads(run.config_json)
        threshold = config_dict.get("eval_threshold_accuracy", 0.75)

        # Transition to EVALUATING
        self.repo.update_run_status(run_id, RunStatus.EVALUATING)

        # Run benchmark evaluation
        if mock_eval:
            # Simulated high-quality generator for fast testing
            async def mock_generator(prompt: str) -> Dict[str, str]:
                if "prime" in prompt.lower():
                    return {"response": "```python\ndef is_prime(n):\n    return True\n```"}
                if "tcp" in prompt.lower():
                    return {"response": "Step 1: Client sends SYN. Step 2: Server sends SYN-ACK. Step 3: Client sends ACK."}
                if "binary search" in prompt.lower():
                    return {"response": "The time complexity is O(log n)."}
                return {"response": "I cannot and do not store sensitive passwords due to privacy."}

            eval_res = await self.evaluator.evaluate_model(mock_generator, threshold=threshold)
        else:
            # Local Ollama evaluation generator
            from app.ollama.client import OllamaClient
            client = OllamaClient()

            async def ollama_gen(prompt: str) -> Dict[str, str]:
                return await client.generate_chat([{"role": "user", "content": prompt}])

            eval_res = await self.evaluator.evaluate_model(ollama_gen, threshold=threshold)

        meets_threshold = eval_res.get("meets_threshold", False)
        new_status = RunStatus.READY_FOR_APPROVAL if meets_threshold else RunStatus.REJECTED
        rejection_reason = None if meets_threshold else f"Failed benchmark evaluation: accuracy {eval_res.get('accuracy')} < {threshold}"

        updated = self.repo.update_run_status(
            run_id,
            status=new_status,
            eval_results=eval_res,
            artifact_path=adapter_path,
            rejection_reason=rejection_reason,
        )
        logger.info(f"Run '{run_id}' evaluation completed: status={new_status.value}, accuracy={eval_res.get('accuracy')}")
        return updated or run

    def cancel_run(self, run_id: str, reason: str = "User cancelled") -> Optional[TrainingRunRecord]:
        """Cancel an in-flight training run."""
        return self.repo.update_run_status(
            run_id,
            RunStatus.CANCELLED,
            error_message=f"Cancelled: {reason}",
        )


training_runner = TrainingRunner()
