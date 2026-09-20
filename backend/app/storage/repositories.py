import json
import logging
from typing import Any, Dict, List, Optional

from app.storage.database import DatabaseManager, db_manager
from app.storage.models import (
    BehaviorStatRecord,
    ContextRecord,
    ConversationRecord,
    FeedbackRecord,
    InterestRecord,
    MessageRecord,
    PreferenceRecord,
    RawEventRecord,
    ResponseStyleRecord,
    utc_now,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# RAW REPOSITORY (Source of Truth)
# ==============================================================================
class RawRepository:
    def __init__(self, manager: Optional[DatabaseManager] = None):
        self.manager = manager or db_manager

    def save_conversation(self, conv_id: str, title: str) -> None:
        now = utc_now()
        with self.manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    updated_at = excluded.updated_at
                """,
                (conv_id, title, now, now),
            )
            conn.commit()

    def save_message(
        self,
        msg_id: str,
        conversation_id: str,
        role: str,
        content: str,
        model: str,
        latency_ms: float = 0.0,
        user_id: str = "local_user",
    ) -> None:
        now = utc_now()
        with self.manager.get_connection() as conn:
            # Ensure parent conversation exists
            conn.execute(
                """
                INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, content[:30], now, now),
            )
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, user_id, role, content, model, latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (msg_id, conversation_id, user_id, role, content, model, latency_ms, now),
            )
            conn.commit()

    def get_message_by_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Fetch message by its unique ID to correlate assistant responses with feedback."""
        with self.manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM messages WHERE id = ?",
                (message_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def save_raw_event(
        self,
        event_id: str,
        event_type: str,
        timestamp: str,
        user_id: str,
        conversation_id: str,
        message_id: str,
        payload_json: str,
    ) -> bool:
        """Saves raw event. Returns True if inserted, False if duplicate already exists."""
        with self.manager.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO raw_events (event_id, event_type, timestamp, user_id, conversation_id, message_id, payload_json, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (event_id, event_type, timestamp, user_id, conversation_id, message_id, payload_json),
            )
            conn.commit()
            return cursor.rowcount > 0

    def is_event_processed(self, event_id: str) -> bool:
        with self.manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT processed FROM raw_events WHERE event_id = ?",
                (event_id,),
            )
            row = cursor.fetchone()
            return bool(row and row["processed"] == 1)

    def mark_event_processed(self, event_id: str) -> None:
        with self.manager.get_connection() as conn:
            conn.execute(
                "UPDATE raw_events SET processed = 1 WHERE event_id = ?",
                (event_id,),
            )
            conn.commit()

    def get_unprocessed_raw_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves raw events that have not yet been consumed into the user profile."""
        with self.manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT payload_json FROM raw_events WHERE processed = 0 ORDER BY timestamp ASC LIMIT ?",
                (limit,),
            )
            events = []
            for row in cursor.fetchall():
                try:
                    events.append(json.loads(row["payload_json"]))
                except Exception as err:
                    logger.error(f"Error parsing raw event payload: {err}")
            return events

    def save_feedback(
        self,
        fb_id: str,
        message_id: str,
        conversation_id: Optional[str],
        feedback_type: str,
        rating: Optional[int] = None,
        comment: Optional[str] = None,
        feedback_value: Optional[str] = None,
        user_id: str = "local_user",
        metadata_json: Optional[str] = None,
    ) -> None:
        """Stores feedback associated directly with an assistant response."""
        now = utc_now()
        with self.manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO feedback (id, message_id, conversation_id, user_id, feedback_type, feedback_value, rating, comment, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    feedback_value = COALESCE(excluded.feedback_value, feedback.feedback_value),
                    metadata_json = COALESCE(excluded.metadata_json, feedback.metadata_json),
                    rating = COALESCE(excluded.rating, feedback.rating),
                    comment = COALESCE(excluded.comment, feedback.comment)
                """,
                (fb_id, message_id, conversation_id, user_id, feedback_type, feedback_value, rating, comment, metadata_json, now),
            )
            conn.commit()

    def get_feedback_by_message_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        with self.manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM feedback WHERE message_id = ? ORDER BY created_at DESC LIMIT 1",
                (message_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        with self.manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def count_raw_events(self) -> int:
        with self.manager.get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM raw_events").fetchone()
            return row["count"] if row else 0


# ==============================================================================
# DERIVED PROFILE REPOSITORY
# ==============================================================================
class ProfileRepository:
    def __init__(self, manager: Optional[DatabaseManager] = None):
        self.manager = manager or db_manager

    def get_interest(self, topic: str) -> Optional[InterestRecord]:
        with self.manager.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM user_interests WHERE topic = ?",
                (topic,),
            ).fetchone()
            if not row:
                return None
            return InterestRecord(
                topic=row["topic"],
                score=row["score"],
                interaction_count=row["interaction_count"],
                recent_interactions=row["recent_interactions"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                updated_at=row["updated_at"],
            )

    def upsert_interest(self, record: InterestRecord) -> None:
        with self.manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_interests (topic, score, interaction_count, recent_interactions, first_seen, last_seen, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(topic) DO UPDATE SET
                    score = excluded.score,
                    interaction_count = excluded.interaction_count,
                    recent_interactions = excluded.recent_interactions,
                    last_seen = excluded.last_seen,
                    updated_at = excluded.updated_at
                """,
                (
                    record.topic,
                    record.score,
                    record.interaction_count,
                    record.recent_interactions,
                    record.first_seen,
                    record.last_seen,
                    record.updated_at,
                ),
            )
            conn.commit()

    def get_all_interests(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM user_interests ORDER BY score DESC, interaction_count DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_preference(self, preference: str) -> Optional[PreferenceRecord]:
        with self.manager.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM user_preferences WHERE preference = ?",
                (preference,),
            ).fetchone()
            if not row:
                return None
            return PreferenceRecord(
                preference=row["preference"],
                value=row["value"],
                confidence=row["confidence"],
                evidence_count=row["evidence_count"],
                updated_at=row["updated_at"],
            )

    def upsert_preference(self, record: PreferenceRecord) -> None:
        with self.manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_preferences (preference, value, confidence, evidence_count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(preference) DO UPDATE SET
                    value = excluded.value,
                    confidence = excluded.confidence,
                    evidence_count = excluded.evidence_count,
                    updated_at = excluded.updated_at
                """,
                (
                    record.preference,
                    record.value,
                    record.confidence,
                    record.evidence_count,
                    record.updated_at,
                ),
            )
            conn.commit()

    def get_all_preferences(self) -> List[Dict[str, Any]]:
        with self.manager.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM user_preferences ORDER BY confidence DESC")
            return [dict(row) for row in cursor.fetchall()]

    def upsert_context(self, record: ContextRecord) -> None:
        with self.manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_contexts (id, topic, activity, context, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    record.id,
                    record.topic,
                    record.activity,
                    record.context,
                    record.status,
                    record.created_at,
                    record.updated_at,
                ),
            )
            conn.commit()

    def get_contexts(self, status: str = "current") -> List[Dict[str, Any]]:
        with self.manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM user_contexts WHERE status = ? ORDER BY updated_at DESC",
                (status,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def upsert_response_style(self, record: ResponseStyleRecord) -> None:
        with self.manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_response_styles (style_key, style_name, description, liked_count, disliked_count, affinity_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(style_key) DO UPDATE SET
                    style_name = excluded.style_name,
                    description = excluded.description,
                    liked_count = excluded.liked_count,
                    disliked_count = excluded.disliked_count,
                    affinity_score = excluded.affinity_score,
                    updated_at = excluded.updated_at
                """,
                (
                    record.style_key,
                    record.style_name,
                    record.description,
                    record.liked_count,
                    record.disliked_count,
                    record.affinity_score,
                    record.updated_at,
                ),
            )
            conn.commit()

    def get_response_style(self, style_key: str) -> Optional[ResponseStyleRecord]:
        with self.manager.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM user_response_styles WHERE style_key = ?",
                (style_key,),
            ).fetchone()
            if not row:
                return None
            return ResponseStyleRecord(
                style_key=row["style_key"],
                style_name=row["style_name"],
                description=row["description"],
                liked_count=row["liked_count"],
                disliked_count=row["disliked_count"],
                affinity_score=row["affinity_score"],
                updated_at=row["updated_at"],
            )

    def get_all_response_styles(self) -> List[Dict[str, Any]]:
        with self.manager.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM user_response_styles ORDER BY affinity_score DESC, liked_count DESC"
            )
            return [dict(row) for row in cursor.fetchall()]

    def set_stat(self, stat_key: str, stat_value: str) -> None:
        now = utc_now()
        with self.manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_behavior_stats (stat_key, stat_value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(stat_key) DO UPDATE SET
                    stat_value = excluded.stat_value,
                    updated_at = excluded.updated_at
                """,
                (stat_key, stat_value, now),
            )
            conn.commit()

    def get_all_stats(self) -> Dict[str, Any]:
        with self.manager.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM user_behavior_stats")
            return {row["stat_key"]: row["stat_value"] for row in cursor.fetchall()}

    def get_full_profile(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "interests": self.get_all_interests(),
            "preferences": self.get_all_preferences(),
            "response_styles": self.get_all_response_styles(),
            "contexts": self.get_contexts(),
            "behavior_stats": self.get_all_stats(),
        }


raw_repo = RawRepository()
profile_repo = ProfileRepository()
