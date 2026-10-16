from datetime import datetime, timezone
from typing import Optional

from app.memory.models import MemoryRecord, MemoryStatus


class LifecycleManager:
    """
    Evaluates and transitions memory states:
    candidate -> active -> stale -> archived.
    Enforces expiration rules.
    """

    def is_retrievable(self, record: MemoryRecord) -> bool:
        """
        A memory is retrievable only if it is active and unexpired.
        Stale, candidate, and archived memories are excluded from retrieval.
        """
        if record.status != MemoryStatus.ACTIVE.value:
            return False

        if record.expires_at:
            now_iso = datetime.now(timezone.utc).isoformat()
            if record.expires_at < now_iso:
                return False

        return True

    def evaluate_status(self, record: MemoryRecord) -> str:
        """Evaluate whether an active memory should transition to stale due to expiry."""
        if record.status == MemoryStatus.ACTIVE.value and record.expires_at:
            now_iso = datetime.now(timezone.utc).isoformat()
            if record.expires_at < now_iso:
                return MemoryStatus.STALE.value
        return record.status


lifecycle_manager = LifecycleManager()
