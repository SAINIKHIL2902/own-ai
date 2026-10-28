import asyncio
import logging
from typing import Optional

from app.training.service import training_service

logger = logging.getLogger(__name__)


class CloudSyncScheduler:
    """
    Automated background worker for local execution.
    Periodically checks if the 15-day sync or 60-day training thresholds are due.
    Because it uses persistent state watermarks, it never misses a cycle even if
    the user's laptop was closed or asleep.
    """

    def __init__(self, check_interval_seconds: int = 3600, user_id: str = "local_user"):
        self.check_interval_seconds = check_interval_seconds
        self.user_id = user_id
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Starts the background checking loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"CloudSyncScheduler started (Heartbeat interval: {self.check_interval_seconds}s, User: {self.user_id})"
        )

    async def stop(self) -> None:
        """Gracefully stops the background loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("CloudSyncScheduler stopped.")

    async def _run_loop(self) -> None:
        """Core loop: checks status immediately on startup, then every check_interval."""
        while self._running:
            try:
                await self.check_and_execute()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in CloudSyncScheduler loop: {e}")

            try:
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                break

    async def check_and_execute(self) -> None:
        """Evaluates whether sync or training milestones are reached and runs them."""
        status = training_service.get_cloud_sync_status(user_id=self.user_id)
        
        # 1. 15-day sync check
        if status.get("sync_due", False):
            logger.info("15-day sync milestone reached. Automatically syncing delta to Supabase...")
            result = await training_service.sync_cloud_delta(user_id=self.user_id, force=False)
            logger.info(
                f"Automatic 15-day sync completed: batch_id={result.get('batch_id')}, "
                f"uploaded_to_cloud={result.get('uploaded_to_cloud')}"
            )

        # 2. 60-day training milestone check
        training_pipeline = status.get("training_pipeline", {})
        if training_pipeline.get("60_day_training_due", False):
            logger.info("60-day training milestone reached (4 batches accumulated). Auto-preparing training unit...")
            train_res = training_service.prepare_60_day_training(user_id=self.user_id, force=False)
            logger.info(
                f"Automatic 60-day training unit prepared: version_id={train_res.get('version_id')}, "
                f"zip={train_res.get('unit_zip_path')}"
            )


cloud_sync_scheduler = CloudSyncScheduler()
