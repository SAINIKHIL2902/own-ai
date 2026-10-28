import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.training.scheduler import CloudSyncScheduler


@pytest.mark.asyncio
async def test_scheduler_lifecycle():
    scheduler = CloudSyncScheduler(check_interval_seconds=1, user_id="test_user")

    with patch.object(scheduler, "check_and_execute", new_callable=AsyncMock) as mock_check:
        await scheduler.start()
        assert scheduler._running is True
        assert scheduler._task is not None

        # Give it a brief moment to run first check
        await asyncio.sleep(0.05)
        assert mock_check.called

        await scheduler.stop()
        assert scheduler._running is False


@pytest.mark.asyncio
async def test_scheduler_triggers_when_due():
    scheduler = CloudSyncScheduler(check_interval_seconds=1, user_id="test_user")

    mock_status = {
        "sync_due": True,
        "training_pipeline": {"60_day_training_due": True},
    }

    with patch("app.training.scheduler.training_service") as mock_service:
        mock_service.get_cloud_sync_status.return_value = mock_status
        mock_service.sync_cloud_delta = AsyncMock(return_value={"batch_id": "b_1", "uploaded_to_cloud": True})
        mock_service.prepare_60_day_training = MagicMock(return_value={"version_id": "v_1", "unit_zip_path": "unit.zip"})

        await scheduler.check_and_execute()

        mock_service.sync_cloud_delta.assert_called_once_with(user_id="test_user", force=False)
        mock_service.prepare_60_day_training.assert_called_once_with(user_id="test_user", force=False)


@pytest.mark.asyncio
async def test_scheduler_skips_when_not_due():
    scheduler = CloudSyncScheduler(check_interval_seconds=1, user_id="test_user")

    mock_status = {
        "sync_due": False,
        "training_pipeline": {"60_day_training_due": False},
    }

    with patch("app.training.scheduler.training_service") as mock_service:
        mock_service.get_cloud_sync_status.return_value = mock_status
        mock_service.sync_cloud_delta = AsyncMock()
        mock_service.prepare_60_day_training = MagicMock()

        await scheduler.check_and_execute()

        mock_service.sync_cloud_delta.assert_not_called()
        mock_service.prepare_60_day_training.assert_not_called()
