import json
import logging
from typing import Any, Dict, List, Optional

from app.storage.database import DatabaseManager, db_manager
from app.training.models import RunStatus, TrainingConsentRecord, TrainingRunRecord, utc_now_iso

logger = logging.getLogger(__name__)


class TrainingRepository:
    """SQLite data access for training consent and versioned model registry."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or db_manager

    # =========================================================================
    # CONSENT
    # =========================================================================
    def get_consent(self, user_id: str = "local_user") -> Optional[TrainingConsentRecord]:
        """Fetch training consent record for user."""
        sql = "SELECT user_id, opted_in, scope, updated_at FROM training_consent WHERE user_id = ?"
        with self.db.get_connection() as conn:
            row = conn.execute(sql, (user_id,)).fetchone()
            if not row:
                return None
            return TrainingConsentRecord(
                user_id=row["user_id"],
                opted_in=bool(row["opted_in"]),
                scope=row["scope"],
                updated_at=row["updated_at"],
            )

    def set_consent(
        self, user_id: str = "local_user", opted_in: bool = False, scope: str = "behavior_only"
    ) -> TrainingConsentRecord:
        """Set explicit training consent for a user."""
        now = utc_now_iso()
        sql = """
        INSERT OR REPLACE INTO training_consent (user_id, opted_in, scope, updated_at)
        VALUES (?, ?, ?, ?)
        """
        with self.db.get_connection() as conn:
            conn.execute(sql, (user_id, 1 if opted_in else 0, scope, now))
            conn.commit()

        logger.info(f"Updated training consent for user '{user_id}': opted_in={opted_in}, scope={scope}")
        return TrainingConsentRecord(
            user_id=user_id,
            opted_in=opted_in,
            scope=scope,
            updated_at=now,
        )

    # =========================================================================
    # MODEL REGISTRY
    # =========================================================================
    def save_run(self, record: TrainingRunRecord) -> None:
        """Insert or replace a training run record."""
        now = utc_now_iso()
        created_at = record.created_at or now
        updated_at = record.updated_at or now

        sql = """
        INSERT OR REPLACE INTO model_registry (
            run_id, base_model, dataset_version, status, config_json,
            metrics_json, eval_results_json, artifact_path, candidate_tag,
            is_active, rejection_reason, error_message, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.get_connection() as conn:
            conn.execute(
                sql,
                (
                    record.run_id,
                    record.base_model,
                    record.dataset_version,
                    record.status,
                    record.config_json,
                    record.metrics_json,
                    record.eval_results_json,
                    record.artifact_path,
                    record.candidate_tag,
                    1 if record.is_active else 0,
                    record.rejection_reason,
                    record.error_message,
                    created_at,
                    updated_at,
                ),
            )
            conn.commit()

    def get_run(self, run_id: str) -> Optional[TrainingRunRecord]:
        """Fetch a specific training run by ID."""
        sql = "SELECT * FROM model_registry WHERE run_id = ?"
        with self.db.get_connection() as conn:
            row = conn.execute(sql, (run_id,)).fetchone()
            if not row:
                return None
            return self._row_to_run(row)

    def list_runs(self, limit: int = 50) -> List[TrainingRunRecord]:
        """List training runs in reverse chronological order."""
        sql = "SELECT * FROM model_registry ORDER BY created_at DESC LIMIT ?"
        with self.db.get_connection() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
            return [self._row_to_run(r) for r in rows]

    def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        metrics: Optional[Dict[str, Any]] = None,
        eval_results: Optional[Dict[str, Any]] = None,
        artifact_path: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[TrainingRunRecord]:
        """Update run stage, metrics, evaluation, or error status."""
        run = self.get_run(run_id)
        if not run:
            return None

        run.status = status.value
        run.updated_at = utc_now_iso()
        if metrics is not None:
            run.metrics_json = json.dumps(metrics)
        if eval_results is not None:
            run.eval_results_json = json.dumps(eval_results)
        if artifact_path is not None:
            run.artifact_path = artifact_path
        if rejection_reason is not None:
            run.rejection_reason = rejection_reason
        if error_message is not None:
            run.error_message = error_message

        self.save_run(run)
        return run

    def get_active_model(self) -> Optional[TrainingRunRecord]:
        """Fetch the currently active deployed model."""
        sql = "SELECT * FROM model_registry WHERE is_active = 1 ORDER BY updated_at DESC LIMIT 1"
        with self.db.get_connection() as conn:
            row = conn.execute(sql).fetchone()
            if not row:
                return None
            return self._row_to_run(row)

    def set_active_model(self, run_id: str) -> bool:
        """Activate a model candidate and deactivate any previous active model."""
        run = self.get_run(run_id)
        if not run:
            return False

        now = utc_now_iso()
        with self.db.get_connection() as conn:
            # Deactivate all
            conn.execute("UPDATE model_registry SET is_active = 0, updated_at = ?", (now,))
            # Activate target
            conn.execute(
                "UPDATE model_registry SET is_active = 1, status = ?, updated_at = ? WHERE run_id = ?",
                (RunStatus.DEPLOYED.value, now, run_id),
            )
            conn.commit()

        logger.info(f"Model run '{run_id}' ({run.candidate_tag}) activated as current production model.")
        return True

    def rollback_model(self) -> Optional[str]:
        """Deactivate all custom candidate models and revert to baseline default model."""
        now = utc_now_iso()
        with self.db.get_connection() as conn:
            conn.execute("UPDATE model_registry SET is_active = 0, updated_at = ?", (now,))
            conn.commit()
        logger.info("Rolled back to baseline default model.")
        return None

    def _row_to_run(self, row: Any) -> TrainingRunRecord:
        return TrainingRunRecord(
            run_id=row["run_id"],
            base_model=row["base_model"],
            dataset_version=row["dataset_version"],
            status=row["status"],
            config_json=row["config_json"],
            candidate_tag=row["candidate_tag"],
            metrics_json=row["metrics_json"],
            eval_results_json=row["eval_results_json"],
            artifact_path=row["artifact_path"],
            is_active=bool(row["is_active"]),
            rejection_reason=row["rejection_reason"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


training_repo = TrainingRepository()
