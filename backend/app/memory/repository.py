from datetime import datetime, timezone
import logging
import re
from typing import Any, Dict, List, Optional, Set

from app.memory.models import MemoryRecord, MemoryStatus
from app.storage.database import DatabaseManager, db_manager

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> Set[str]:
    """Normalize and extract alphanumeric tokens for similarity comparison."""
    words = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
    stop_words = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
        "with", "is", "are", "was", "were", "it", "that", "this", "user",
        "prefers", "prefer", "likes", "like", "wants", "want", "remember",
    }
    return {w for w in words if w not in stop_words}


class MemoryRepository:
    """SQLite data access repository for user memories."""

    def __init__(self, db: Optional[DatabaseManager] = None, manager: Optional[DatabaseManager] = None):
        self.db = db or manager or db_manager

    def save_memory(self, record: MemoryRecord) -> None:

        """Insert or replace a memory record."""
        now = _utc_now_iso()
        created_at = record.created_at or now
        updated_at = record.updated_at or now

        sql = """
        INSERT OR REPLACE INTO memories (
            memory_id, user_id, memory_type, content, source_type,
            source_reference, confidence, importance, evidence_count,
            created_at, updated_at, last_confirmed_at, expires_at, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.get_connection() as conn:
            conn.execute(
                sql,
                (
                    record.memory_id,
                    record.user_id,
                    record.memory_type,
                    record.content,
                    record.source_type,
                    record.source_reference,
                    float(record.confidence),
                    float(record.importance),
                    int(record.evidence_count),
                    created_at,
                    updated_at,
                    record.last_confirmed_at,
                    record.expires_at,
                    record.status,
                ),
            )
            conn.commit()

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        """Fetch a single memory by ID."""
        sql = "SELECT * FROM memories WHERE memory_id = ?"
        with self.db.get_connection() as conn:
            row = conn.execute(sql, (memory_id,)).fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def list_memories(
        self,
        user_id: str = "local_user",
        memory_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[MemoryRecord]:
        """List memories with optional type and status filters."""
        query = "SELECT * FROM memories WHERE user_id = ?"
        params: List[Any] = [user_id]

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)

        if status:
            query += " AND status = ?"
            params.append(status)
        else:
            # By default exclude archived memories from listing unless requested
            query += " AND status != 'archived'"

        query += " ORDER BY updated_at DESC"

        with self.db.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_record(r) for r in rows]

    def get_active_memories(self, user_id: str = "local_user") -> List[MemoryRecord]:
        """Retrieve active memories, auto-excluding or marking expired ones as stale."""
        now_iso = _utc_now_iso()
        query = "SELECT * FROM memories WHERE user_id = ? AND status = 'active'"

        active_records: List[MemoryRecord] = []
        expired_ids: List[str] = []

        with self.db.get_connection() as conn:
            rows = conn.execute(query, (user_id,)).fetchall()
            for r in rows:
                rec = self._row_to_record(r)
                if rec.expires_at and rec.expires_at < now_iso:
                    expired_ids.append(rec.memory_id)
                else:
                    active_records.append(rec)

            # Mark expired memories as stale in background
            if expired_ids:
                conn.executemany(
                    "UPDATE memories SET status = 'stale', updated_at = ? WHERE memory_id = ?",
                    [(now_iso, mid) for mid in expired_ids],
                )
                conn.commit()

        return active_records

    def find_similar_memory(
        self,
        content: str,
        memory_type: str,
        user_id: str = "local_user",
        threshold: float = 0.50,
    ) -> Optional[MemoryRecord]:
        """
        Find existing active memory of same type with high token overlap to prevent duplicates.
        """
        target_tokens = _tokenize(content)
        if not target_tokens:
            return None

        candidates = self.list_memories(user_id=user_id, memory_type=memory_type, status="active")
        best_match: Optional[MemoryRecord] = None
        highest_sim = 0.0

        for cand in candidates:
            cand_tokens = _tokenize(cand.content)
            if not cand_tokens:
                continue

            intersection = len(target_tokens.intersection(cand_tokens))
            union = len(target_tokens.union(cand_tokens))
            jaccard = intersection / float(union) if union > 0 else 0.0

            if jaccard >= threshold and jaccard > highest_sim:
                highest_sim = jaccard
                best_match = cand

        return best_match

    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> Optional[MemoryRecord]:
        """Apply safe updates to an existing memory."""
        allowed_fields = {
            "content", "memory_type", "importance", "confidence",
            "evidence_count", "status", "expires_at", "last_confirmed_at",
        }
        filtered = {k: v for k, v in updates.items() if k in allowed_fields}
        if not filtered:
            return self.get_memory(memory_id)

        filtered["updated_at"] = _utc_now_iso()
        set_clauses = [f"{k} = ?" for k in filtered.keys()]
        values = list(filtered.values()) + [memory_id]

        sql = f"UPDATE memories SET {', '.join(set_clauses)} WHERE memory_id = ?"
        with self.db.get_connection() as conn:
            cur = conn.execute(sql, values)
            conn.commit()
            if cur.rowcount == 0:
                return None

        return self.get_memory(memory_id)

    def delete_memory(self, memory_id: str, soft: bool = True) -> bool:
        """Delete or archive a memory."""
        with self.db.get_connection() as conn:
            if soft:
                sql = "UPDATE memories SET status = 'archived', updated_at = ? WHERE memory_id = ?"
                cur = conn.execute(sql, (_utc_now_iso(), memory_id))
            else:
                sql = "DELETE FROM memories WHERE memory_id = ?"
                cur = conn.execute(sql, (memory_id,))
            conn.commit()
            return cur.rowcount > 0

    @staticmethod
    def _row_to_record(row: Any) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            memory_type=row["memory_type"],
            content=row["content"],
            source_type=row["source_type"],
            source_reference=row["source_reference"],
            confidence=float(row["confidence"]),
            importance=float(row["importance"]),
            evidence_count=int(row["evidence_count"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_confirmed_at=row["last_confirmed_at"],
            expires_at=row["expires_at"],
            status=row["status"],
        )


memory_repo = MemoryRepository()
