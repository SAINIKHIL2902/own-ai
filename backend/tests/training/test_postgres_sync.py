import json
from pathlib import Path
import pytest
from datetime import datetime, timedelta, timezone

from app.training.postgres_sync import PostgresSyncManager


@pytest.fixture
def temp_sync_env(tmp_path):
    state_file = tmp_path / "sync_state.json"
    staging_dir = tmp_path / "cloud_staging"
    manager = PostgresSyncManager(
        cloud_url=None,
        sync_interval_days=15,
        train_interval_days=60,
        state_file=state_file,
        staging_dir=staging_dir,
    )
    return manager, state_file, staging_dir


def test_is_15_day_sync_due_initially(temp_sync_env):
    manager, state_file, _ = temp_sync_env
    is_due, days_elapsed, next_due = manager.is_15_day_sync_due(user_id="local_user")
    assert is_due is True
    assert days_elapsed == 0


def test_is_15_day_sync_due_after_recent_sync(temp_sync_env):
    manager, state_file, _ = temp_sync_env
    # Simulate a sync done 5 days ago
    five_days_ago = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    state = {
        "last_synced_at": five_days_ago,
        "last_training_at": None,
        "synced_batches_count": 1,
        "batches": [],
    }
    manager.save_state(state)

    is_due, days_elapsed, next_due = manager.is_15_day_sync_due(user_id="local_user")
    assert is_due is False
    assert days_elapsed == 5


def test_is_15_day_sync_due_after_16_days(temp_sync_env):
    manager, state_file, _ = temp_sync_env
    # Simulate a sync done 16 days ago
    sixteen_days_ago = (datetime.now(timezone.utc) - timedelta(days=16)).isoformat()
    state = {
        "last_synced_at": sixteen_days_ago,
        "last_training_at": None,
        "synced_batches_count": 1,
        "batches": [],
    }
    manager.save_state(state)

    is_due, days_elapsed, next_due = manager.is_15_day_sync_due(user_id="local_user")
    assert is_due is True
    assert days_elapsed == 16


def test_sanitize_delta_payload(temp_sync_env):
    manager, _, _ = temp_sync_env
    delta = manager.gather_and_sanitize_delta(user_id="local_user")
    assert "user_id" in delta
    assert "profile" in delta
    assert "memories" in delta
    assert "chat_turns" in delta


@pytest.mark.asyncio
async def test_sync_15_day_batch_execution(temp_sync_env):
    manager, state_file, staging_dir = temp_sync_env

    # 1. Run forced sync
    res = await manager.sync_15_day_batch(user_id="local_user", force=True)
    assert res["success"] is True
    assert res["synced"] is True
    assert "batch_v_" in res["batch_id"]

    # 2. Verify state persistence
    state = manager.load_state()
    assert state["synced_batches_count"] == 1
    assert state["last_synced_at"] is not None
    assert len(state["batches"]) == 1

    # 3. Verify local project staging file
    staged_files = list(staging_dir.glob("*.json"))
    assert len(staged_files) == 1


@pytest.mark.asyncio
async def test_skip_sync_when_not_due(temp_sync_env):
    manager, _, _ = temp_sync_env
    # First sync
    await manager.sync_15_day_batch(user_id="local_user", force=True)

    # Second sync without force -> should be skipped
    res = await manager.sync_15_day_batch(user_id="local_user", force=False)
    assert res["success"] is False
    assert res["synced"] is False
    assert "Sync not due yet" in res["message"]


def test_60_day_training_readiness_and_compilation(temp_sync_env):
    manager, state_file, staging_dir = temp_sync_env

    # Simulate 4 accumulated 15-day sync cycles
    batches = []
    for i in range(4):
        bid = f"batch_v_{i}"
        sfile = staging_dir / f"{bid}.json"
        sfile.write_text(json.dumps({
            "batch_id": bid,
            "chat_turns": [{"conversation_id": f"c_{i}", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}],
        }), encoding="utf-8")
        batches.append({
            "batch_id": bid,
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "used_in_training": False,
            "staging_path": str(sfile),
        })

    manager.save_state({
        "last_synced_at": datetime.now(timezone.utc).isoformat(),
        "last_training_at": None,
        "synced_batches_count": 4,
        "batches": batches,
    })

    # Check readiness
    train_due, days_since_train, pending = manager.is_60_day_training_due(user_id="local_user")
    assert train_due is True
    assert pending == 4

    # Prepare 60-day training unit
    compile_res = manager.prepare_60_day_training_unit(user_id="local_user")
    assert compile_res["success"] is True
    assert compile_res["version_id"].startswith("v_")
    assert compile_res["batches_aggregated"] == 4
    assert Path(compile_res["unit_zip_path"]).exists()

    # Verify batches marked as trained
    updated_state = manager.load_state()
    assert all(b["used_in_training"] for b in updated_state["batches"])
    assert updated_state["last_training_at"] is not None
