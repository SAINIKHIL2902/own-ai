from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    CREATED = "CREATED"
    DATA_VALIDATED = "DATA_VALIDATED"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    PACKAGING = "PACKAGING"
    READY_FOR_APPROVAL = "READY_FOR_APPROVAL"
    DEPLOYED = "DEPLOYED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrainingConsentRecord:
    user_id: str
    opted_in: bool
    scope: str = "behavior_only"
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "opted_in": self.opted_in,
            "scope": self.scope,
            "updated_at": self.updated_at,
        }


class TrainingConfig(BaseModel):
    base_model: str = "llama3.2:1b"
    framework: str = "peft_lora"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    train_val_split: float = 0.85
    random_seed: int = 42
    candidate_tag: str = "llama3.2:1b-behavior-v1"
    eval_threshold_accuracy: float = 0.75
    target_device: str = "cuda"  # for cloud GPU, or "mps"/"cpu"


@dataclass
class TrainingRunRecord:
    run_id: str
    base_model: str
    dataset_version: str
    status: str
    config_json: str
    candidate_tag: str
    metrics_json: Optional[str] = None
    eval_results_json: Optional[str] = None
    artifact_path: Optional[str] = None
    is_active: bool = False
    rejection_reason: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "base_model": self.base_model,
            "dataset_version": self.dataset_version,
            "status": self.status,
            "config": json.loads(self.config_json) if self.config_json else {},
            "metrics": json.loads(self.metrics_json) if self.metrics_json else {},
            "eval_results": json.loads(self.eval_results_json) if self.eval_results_json else {},
            "artifact_path": self.artifact_path,
            "candidate_tag": self.candidate_tag,
            "is_active": self.is_active,
            "rejection_reason": self.rejection_reason,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class TrainingExample:
    messages: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": self.messages,
            "metadata": self.metadata,
        }


class ConsentRequest(BaseModel):
    user_id: str = "local_user"
    opted_in: bool = True


class DatasetBuildRequest(BaseModel):
    user_id: str = "local_user"
    base_model: Optional[str] = None
    train_val_split: Optional[float] = None


class TrainingRunCreateRequest(BaseModel):
    dataset_version: str
    user_id: str = "local_user"
    base_model: Optional[str] = None
    num_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    candidate_tag: Optional[str] = None


class EvaluateRunRequest(BaseModel):
    run_id: str
    adapter_path: Optional[str] = None
    mock_eval: bool = True


class PackageRunRequest(BaseModel):
    run_id: str


class ActivateRunRequest(BaseModel):
    run_id: str


class RollbackRunRequest(BaseModel):
    reason: Optional[str] = None

