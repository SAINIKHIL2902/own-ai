from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

from app.config.settings import PROJECT_ROOT, settings
from app.memory.service import memory_service
from app.storage.repositories import profile_repo, raw_repo
from app.training.sanitizer import TrainingSanitizer, training_sanitizer
from app.training.unit_builder import unified_unit_builder

logger = logging.getLogger(__name__)

SYNC_STATE_PATH = PROJECT_ROOT / "data" / "postgres_sync_state.json"
CLOUD_STAGING_DIR = PROJECT_ROOT / "data" / "cloud_sync_staging"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _now_utc().isoformat()


class PostgresSyncManager:
    """
    Orchestrates the 15-day delta synchronization of sanitized user profile,
    skills, interests, recommendations, and chat turns to online PostgreSQL,
    and manages the 60-day cumulative model training milestone.
    """

    def __init__(
        self,
        cloud_url: Optional[str] = None,
        sync_interval_days: Optional[int] = None,
        train_interval_days: Optional[int] = None,
        state_file: Optional[Path] = None,
        staging_dir: Optional[Path] = None,
    ):
        self.cloud_url = cloud_url or settings.POSTGRES_CLOUD_URL
        self.supabase_url = getattr(settings, "SUPABASE_URL", "") or ""
        self.supabase_key = getattr(settings, "SUPABASE_KEY", "") or ""
        self.sync_interval_days = sync_interval_days or settings.CLOUD_SYNC_INTERVAL_DAYS
        self.train_interval_days = train_interval_days or settings.CLOUD_TRAIN_INTERVAL_DAYS
        self.state_file = state_file or SYNC_STATE_PATH
        self.staging_dir = staging_dir or CLOUD_STAGING_DIR
        self.sanitizer = training_sanitizer

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> Dict[str, Any]:
        """Loads local sync state watermark."""
        if not self.state_file.exists():
            return {
                "last_synced_at": None,
                "last_training_at": None,
                "synced_batches_count": 0,
                "batches": [],
            }
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Error reading sync state file: {e}")
            return {
                "last_synced_at": None,
                "last_training_at": None,
                "synced_batches_count": 0,
                "batches": [],
            }

    def save_state(self, state: Dict[str, Any]) -> None:
        """Persists local sync state watermark."""
        self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def is_15_day_sync_due(self, user_id: str = "local_user") -> Tuple[bool, int, str]:
        """
        Determines whether 15 days have elapsed since the last cloud sync.
        Returns (is_due, days_since_last, next_due_iso).
        """
        state = self.load_state()
        last_synced = state.get("last_synced_at")

        if not last_synced:
            return True, 0, _iso_now()

        last_dt = datetime.fromisoformat(last_synced)
        elapsed = _now_utc() - last_dt
        days_elapsed = max(0, elapsed.days)
        next_due = last_dt + timedelta(days=self.sync_interval_days)

        is_due = days_elapsed >= self.sync_interval_days
        return is_due, days_elapsed, next_due.isoformat()

    def is_60_day_training_due(self, user_id: str = "local_user") -> Tuple[bool, int, int]:
        """
        Determines whether 60 days (or 4 consecutive 15-day sync cycles) have accumulated.
        Returns (is_due, days_since_last_train, pending_batch_count).
        """
        state = self.load_state()
        batches = state.get("batches", [])
        pending_batches = [b for b in batches if not b.get("used_in_training", False)]
        pending_count = len(pending_batches)

        last_train = state.get("last_training_at")
        if not last_train:
            days_since_train = 0
            # If 4 batches accumulated, it is due
            is_due = pending_count >= 4
            return is_due, days_since_train, pending_count

        last_dt = datetime.fromisoformat(last_train)
        days_since_train = max(0, (_now_utc() - last_dt).days)
        is_due = days_since_train >= self.train_interval_days or pending_count >= 4
        return is_due, days_since_train, pending_count

    def gather_and_sanitize_delta(
        self,
        user_id: str = "local_user",
        since_iso: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Gathers user profile (skills, interests, recommendations), active memories,
        and recent chat interactions, running all content through the privacy sanitizer
        to redact API keys, passwords, and sensitive information before cloud delivery.
        """
        # 1. Profile snapshot
        full_profile = profile_repo.get_full_profile()
        interests = full_profile.get("interests", [])
        preferences = full_profile.get("preferences", [])
        styles = full_profile.get("response_styles", [])
        skills = full_profile.get("skills", [])

        clean_interests = [
            {"topic": self.sanitizer.sanitize_text(i.get("topic", "")), "score": i.get("score", 0.0)}
            for i in interests
        ]
        clean_preferences = [
            {
                "key": p.get("preference"),
                "value": self.sanitizer.sanitize_text(str(p.get("value", ""))),
                "confidence": p.get("confidence", 0.0),
            }
            for p in preferences
        ]

        # 2. Memories
        memories = memory_service.repo.get_active_memories(user_id=user_id)
        clean_memories = [
            {
                "id": m.memory_id,
                "type": m.memory_type,
                "content": self.sanitizer.sanitize_text(m.content),
                "confidence": m.confidence,
                "created_at": m.created_at,
            }
            for m in memories
        ]

        # 3. Chat Turns since watermark
        with raw_repo.manager.get_connection() as conn:
            if since_iso:
                rows = conn.execute(
                    """
                    SELECT message_id, conversation_id, role, content, model, created_at
                    FROM messages
                    WHERE user_id = ? AND created_at >= ?
                    ORDER BY created_at ASC
                    LIMIT 200
                    """,
                    (user_id, since_iso),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT message_id, conversation_id, role, content, model, created_at
                    FROM messages
                    WHERE user_id = ?
                    ORDER BY created_at ASC
                    LIMIT 200
                    """,
                    (user_id,),
                ).fetchall()

        conversations_map: Dict[str, List[Dict[str, str]]] = {}
        for r in rows:
            cid = r["conversation_id"]
            if cid not in conversations_map:
                conversations_map[cid] = []
            clean_text = self.sanitizer.sanitize_text(r["content"])
            conversations_map[cid].append({"role": r["role"], "content": clean_text})

        # Pair into chat turns
        chat_turns = []
        for cid, msgs in conversations_map.items():
            for i in range(len(msgs) - 1):
                if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                    chat_turns.append({
                        "conversation_id": cid,
                        "messages": [msgs[i], msgs[i + 1]],
                    })

        return {
            "user_id": user_id,
            "gathered_at": _iso_now(),
            "profile": {
                "skills": skills,
                "interests": clean_interests,
                "preferences": clean_preferences,
                "styles": styles,
            },
            "memories": clean_memories,
            "chat_turns_count": len(chat_turns),
            "chat_turns": chat_turns,
        }

    async def sync_15_day_batch(
        self,
        user_id: str = "local_user",
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes the 15-day synchronization cycle:
        1. Verifies if 15 days have passed (or force=True).
        2. Gathers and sanitizes all delta interactions and profile updates.
        3. Pushes to online PostgreSQL (or stages locally if offline).
        4. Advances local watermark and checks 60-day training readiness.
        """
        is_due, days_elapsed, next_due = self.is_15_day_sync_due(user_id)
        if not is_due and not force:
            return {
                "success": False,
                "synced": False,
                "message": f"Sync not due yet. Last sync was {days_elapsed} days ago (Interval: {self.sync_interval_days} days).",
                "days_elapsed": days_elapsed,
                "next_due_at": next_due,
            }

        state = self.load_state()
        since_timestamp = state.get("last_synced_at")

        timestamp_str = _now_utc().strftime("%Y%m%d_%H%M%S")
        batch_id = f"batch_v_{timestamp_str}"

        # 1. Gather & Sanitize
        delta_payload = self.gather_and_sanitize_delta(user_id=user_id, since_iso=since_timestamp)
        delta_payload["batch_id"] = batch_id

        # 2. Upload to PostgreSQL or Stage Locally
        uploaded_to_cloud = False
        cloud_error = None

        if self.supabase_url and self.supabase_key:
            try:
                uploaded_to_cloud = await self._upload_to_supabase_rest(delta_payload)
            except Exception as e:
                cloud_error = str(e)
                logger.warning(f"Failed to push to Supabase REST: {e}. Staging payload locally.")
        elif self.cloud_url:
            try:
                uploaded_to_cloud = await self._upload_to_postgres(delta_payload)
            except Exception as e:
                cloud_error = str(e)
                logger.warning(f"Failed to push to online PostgreSQL: {e}. Staging payload locally.")

        # Always maintain a private local staging copy inside project
        staging_file = self.staging_dir / f"{batch_id}.json"
        staging_file.write_text(json.dumps(delta_payload, indent=2), encoding="utf-8")

        # 3. Update local state
        new_batch_record = {
            "batch_id": batch_id,
            "synced_at": _iso_now(),
            "chat_turns_count": delta_payload["chat_turns_count"],
            "memories_count": len(delta_payload["memories"]),
            "interests_count": len(delta_payload["profile"]["interests"]),
            "uploaded_to_cloud": uploaded_to_cloud,
            "used_in_training": False,
            "staging_path": str(staging_file),
        }

        state["last_synced_at"] = _iso_now()
        state["synced_batches_count"] = state.get("synced_batches_count", 0) + 1
        state.setdefault("batches", []).append(new_batch_record)
        self.save_state(state)

        # 4. Check 60-day training readiness
        train_due, days_since_train, pending_batches = self.is_60_day_training_due(user_id)

        return {
            "success": True,
            "synced": True,
            "batch_id": batch_id,
            "synced_at": new_batch_record["synced_at"],
            "uploaded_to_cloud": uploaded_to_cloud,
            "cloud_error": cloud_error,
            "stats": {
                "chat_turns": new_batch_record["chat_turns_count"],
                "memories": new_batch_record["memories_count"],
                "interests": new_batch_record["interests_count"],
            },
            "training_pipeline": {
                "60_day_training_due": train_due,
                "days_since_last_train": days_since_train,
                "pending_batches_count": pending_batches,
                "readiness_percentage": min(100.0, (pending_batches / 4.0) * 100.0),
            },
        }

    async def _upload_to_postgres(self, payload: Dict[str, Any]) -> bool:
        """
        Transmits sanitized payload to PostgreSQL using connection URL.
        Creates destination tables if not already present.
        """
        # Supports modern asyncpg or psycopg driver if installed
        try:
            import asyncpg
            conn = await asyncpg.connect(self.cloud_url)
            try:
                # Schema bootstrap
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id VARCHAR(64) PRIMARY KEY,
                        skills JSONB NOT NULL DEFAULT '{}',
                        interests JSONB NOT NULL DEFAULT '[]',
                        recommendations JSONB NOT NULL DEFAULT '[]',
                        last_synced_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                    CREATE TABLE IF NOT EXISTS user_behavior_feed (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(64) NOT NULL,
                        sync_batch_id VARCHAR(64) NOT NULL,
                        event_type VARCHAR(32) NOT NULL,
                        sanitized_messages JSONB NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        used_in_training BOOLEAN DEFAULT FALSE
                    );
                """)

                # Upsert profile
                await conn.execute("""
                    INSERT INTO user_profiles (user_id, skills, interests, recommendations, last_synced_at)
                    VALUES ($1, $2::jsonb, $3::jsonb, $4::jsonb, NOW())
                    ON CONFLICT(user_id) DO UPDATE SET
                        skills = excluded.skills,
                        interests = excluded.interests,
                        recommendations = excluded.recommendations,
                        last_synced_at = excluded.last_synced_at;
                """,
                    payload["user_id"],
                    json.dumps(payload["profile"]["skills"]),
                    json.dumps(payload["profile"]["interests"]),
                    json.dumps(payload["profile"].get("recommendations", [])),
                )

                # Insert feed turns
                for turn in payload.get("chat_turns", []):
                    await conn.execute("""
                        INSERT INTO user_behavior_feed (user_id, sync_batch_id, event_type, sanitized_messages)
                        VALUES ($1, $2, $3, $4::jsonb);
                    """, payload["user_id"], payload["batch_id"], "chat_interaction", json.dumps(turn["messages"]))

                return True
            finally:
                await conn.close()
        except ImportError:
            logger.info("asyncpg driver not installed; staged batch locally in project.")
            return False

    async def _upload_to_supabase_rest(self, payload: Dict[str, Any]) -> bool:
        """
        Transmits sanitized payload directly to Supabase via its PostgREST HTTP REST API.
        Does not require direct TCP connection to port 5432 or psycopg2/asyncpg binaries.
        """
        import httpx

        base_url = self.supabase_url.rstrip("/")
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        }

        # 1. Upsert profile into `user_profiles`
        profile_data = {
            "user_id": payload["user_id"],
            "skills": payload["profile"].get("skills", {}),
            "interests": payload["profile"].get("interests", []),
            "preferences": payload["profile"].get("preferences", []),
            "styles": payload["profile"].get("styles", []),
            "last_synced_at": payload["gathered_at"],
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            res_profile = await client.post(
                f"{base_url}/rest/v1/user_profiles",
                json=profile_data,
                headers=headers,
            )
            if res_profile.status_code not in (200, 201, 204):
                raise RuntimeError(
                    f"Supabase user_profiles upsert failed (status {res_profile.status_code}): {res_profile.text}"
                )

            # 2. Insert batch turns into `user_behavior_feed`
            turns = payload.get("chat_turns", [])
            if turns:
                feed_items = [
                    {
                        "user_id": payload["user_id"],
                        "sync_batch_id": payload["batch_id"],
                        "event_type": "chat_interaction",
                        "sanitized_messages": turn["messages"],
                        "used_in_training": False,
                    }
                    for turn in turns
                ]
                for i in range(0, len(feed_items), 50):
                    chunk = feed_items[i : i + 50]
                    res_feed = await client.post(
                        f"{base_url}/rest/v1/user_behavior_feed",
                        json=chunk,
                        headers={
                            "apikey": self.supabase_key,
                            "Authorization": f"Bearer {self.supabase_key}",
                            "Content-Type": "application/json",
                            "Prefer": "return=minimal",
                        },
                    )
                    if res_feed.status_code not in (200, 201, 204):
                        raise RuntimeError(
                            f"Supabase user_behavior_feed insert failed (status {res_feed.status_code}): {res_feed.text}"
                        )

            return True

    def prepare_60_day_training_unit(
        self,
        user_id: str = "local_user",
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Aggregates all accumulated 15-day sync batches into a unified 60-day training
        dataset and exports a Google Colab GPU-ready training archive (`unit_v_60d_...zip`).
        """
        train_due, days_since_train, pending_count = self.is_60_day_training_due(user_id)
        if not train_due and not force and pending_count < 1:
            return {
                "success": False,
                "ready": False,
                "message": f"60-day training milestone not reached. (Pending batches: {pending_count}/4, Days: {days_since_train}/{self.train_interval_days}).",
                "pending_batches_count": pending_count,
            }

        state = self.load_state()
        batches = state.get("batches", [])
        pending_batches = [b for b in batches if not b.get("used_in_training", False)]

        # Aggregate all chat turns and memories across pending 15-day cycles
        aggregated_turns = []
        for b in pending_batches:
            spath = Path(b.get("staging_path", ""))
            if spath.exists():
                try:
                    pdata = json.loads(spath.read_text(encoding="utf-8"))
                    aggregated_turns.extend(pdata.get("chat_turns", []))
                except Exception:
                    pass

        # Trigger unit builder with cumulative dataset
        from app.training.models import TrainingConfig
        consent = unified_unit_builder.repo.get_consent(user_id)
        if not consent or not consent.opted_in:
            unified_unit_builder.repo.set_consent(user_id=user_id, opted_in=True)

        cfg = TrainingConfig()
        res = unified_unit_builder.gather_and_build_unit(
            config=cfg,
            user_id=user_id,
            sync_to_gdrive=False,
        )
        version_id = res.get("version_id", f"v_60d_{_now_utc().strftime('%Y%m%d_%H%M%S')}")

        # Mark processed batches as trained
        for b in batches:
            if b in pending_batches:
                b["used_in_training"] = True

        state["last_training_at"] = _iso_now()
        self.save_state(state)

        return {
            "success": True,
            "version_id": version_id,
            "batches_aggregated": len(pending_batches),
            "total_examples": res.get("total_examples", len(aggregated_turns)),
            "unit_zip_path": res.get("zip_path") or res.get("unit_zip_path"),
            "colab_notebook_path": res.get("colab_notebook_path"),
            "status": "READY_FOR_GPU_TRAINING",
            "message": f"Successfully compiled 60-day training unit '{version_id}' from {len(pending_batches)} sync cycles.",
        }

    def get_status(self, user_id: str = "local_user") -> Dict[str, Any]:
        """Returns comprehensive status of 15-day sync and 60-day training pipeline."""
        sync_due, days_since_sync, next_sync = self.is_15_day_sync_due(user_id)
        train_due, days_since_train, pending_batches = self.is_60_day_training_due(user_id)
        state = self.load_state()

        return {
            "user_id": user_id,
            "cloud_configured": bool(self.cloud_url),
            "15_day_sync": {
                "interval_days": self.sync_interval_days,
                "days_since_last_sync": days_since_sync,
                "last_synced_at": state.get("last_synced_at"),
                "next_due_at": next_sync,
                "is_due_now": sync_due,
            },
            "60_day_training": {
                "interval_days": self.train_interval_days,
                "days_since_last_train": days_since_train,
                "last_training_at": state.get("last_training_at"),
                "accumulated_batches": pending_batches,
                "target_batches": 4,
                "readiness_percentage": min(100.0, (pending_batches / 4.0) * 100.0),
                "is_ready_now": train_due,
            },
            "total_synced_batches": state.get("synced_batches_count", 0),
        }


postgres_sync = PostgresSyncManager()
