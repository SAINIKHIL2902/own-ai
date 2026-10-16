import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import uuid

from app.memory.classifier import MemoryClassifier, memory_classifier
from app.memory.confidence import ConfidenceManager, confidence_manager
from app.memory.context_builder import ContextBuilder, context_builder
from app.memory.detector import MemoryDetector, memory_detector
from app.memory.lifecycle import LifecycleManager, lifecycle_manager
from app.memory.models import (
    MemoryCreateRequest,
    MemoryRecord,
    MemorySource,
    MemoryStatus,
    MemoryType,
    MemoryUpdateRequest,
)
from app.memory.repository import MemoryRepository, memory_repo
from app.memory.retriever import MemoryRetriever, memory_retriever

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryService:
    """
    Central coordinator for Phase 4 Memory & Personalization:
    handles explicit extraction, duplicate reconciliation, contradiction handling,
    confidence reinforcement, lifecycle updates, and API CRUD operations.
    """

    def __init__(
        self,
        repository: Optional[MemoryRepository] = None,
        detector: Optional[MemoryDetector] = None,
        classifier: Optional[MemoryClassifier] = None,
        confidence_mgr: Optional[ConfidenceManager] = None,
        lifecycle_mgr: Optional[LifecycleManager] = None,
        retriever_inst: Optional[MemoryRetriever] = None,
        builder: Optional[ContextBuilder] = None,
    ):
        self.repo = repository or memory_repo
        self.detector = detector or memory_detector
        self.classifier = classifier or memory_classifier
        self.confidence = confidence_mgr or confidence_manager
        self.lifecycle = lifecycle_mgr or lifecycle_manager
        if retriever_inst is not None:
            self.retriever = retriever_inst
        elif repository is not None:
            self.retriever = MemoryRetriever(repository=self.repo)
        else:
            self.retriever = memory_retriever
        self.context_builder = builder or context_builder


    def _emit_event(
        self,
        operation: str,
        memory_id: str,
        memory_type: str,
        content: str,
        confidence: float,
        status: str,
        user_id: str = "local_user",
    ) -> None:
        """Emit asynchronous memory event to Phase 2 outbox / Kafka if loop exists."""
        try:
            from app.events.producer import event_producer
            from app.events.schemas import MemoryEvent

            evt = MemoryEvent(
                operation=operation,
                user_id=user_id,
                memory_id=memory_id,
                memory_type=memory_type,
                content=content,
                confidence=confidence,
                status=status,
            )
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(event_producer.publish_memory_event(evt))
            except RuntimeError:
                pass
        except Exception as exc:
            logger.debug(f"Could not emit memory event: {exc}")

    def retrieve_relevant_memories(
        self,
        prompt: str,
        user_id: str = "local_user",
        top_k: Optional[int] = None,
    ) -> List[MemoryRecord]:
        """Fetch active, relevant memories for prompt personalization."""
        return self.retriever.retrieve_relevant(prompt=prompt, user_id=user_id, top_k=top_k)

    def process_prompt_for_memories(
        self,
        prompt: str,
        user_id: str = "local_user",
        message_id: Optional[str] = None,
    ) -> Optional[MemoryRecord]:
        """
        Scan user prompt for explicit memory statements.
        Handles deduplication, contradictions, and evidence tracking.
        Runs asynchronously after chat response to prevent blocking.
        """
        detection = self.detector.detect(prompt)
        if not detection:
            return None

        content = detection["content"]
        pattern_cat = detection["category"]

        # 1. Classify candidate
        mem_type, importance, expires_at = self.classifier.classify(content, pattern_cat)
        source_type = MemorySource.EXPLICIT_USER.value
        initial_conf = self.confidence.get_initial_confidence(source_type)

        now = _utc_now_iso()

        # 2. Check for contradiction against existing active memories
        active_memories = self.repo.list_memories(user_id=user_id, memory_type=mem_type.value, status="active")
        for existing in active_memories:
            if self.confidence.is_contradiction(existing.content, content):
                logger.info(f"Reconciling contradictory preference: '{existing.content}' -> '{content}'")
                reconciled = self.confidence.reconcile_contradiction(existing, content, source_type)
                reconciled.updated_at = now
                reconciled.source_reference = message_id or existing.source_reference
                self.repo.save_memory(reconciled)
                self._emit_event("reconciled", reconciled.memory_id, reconciled.memory_type, reconciled.content, reconciled.confidence, reconciled.status, user_id)
                return reconciled

        # 3. Check for similar/duplicate memory
        similar = self.repo.find_similar_memory(content, mem_type.value, user_id=user_id)
        if similar:
            # Duplicate / reinforcement: increase confidence and evidence count
            new_conf, new_ev = self.confidence.reinforce(similar.confidence, similar.evidence_count)
            similar.confidence = new_conf
            similar.evidence_count = new_ev
            similar.updated_at = now
            similar.source_reference = message_id or similar.source_reference
            logger.info(f"Reinforcing existing memory '{similar.memory_id}' (evidence={new_ev}, conf={new_conf})")
            self.repo.save_memory(similar)
            self._emit_event("reinforced", similar.memory_id, similar.memory_type, similar.content, similar.confidence, similar.status, user_id)
            return similar

        # 4. Create new memory record
        new_record = MemoryRecord(
            memory_id=f"mem_{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            memory_type=mem_type.value,
            content=content,
            source_type=source_type,
            source_reference=message_id,
            confidence=initial_conf,
            importance=importance,
            evidence_count=1,
            created_at=now,
            updated_at=now,
            last_confirmed_at=now,
            expires_at=expires_at,
            status=MemoryStatus.ACTIVE.value,
        )
        logger.info(f"Created new explicit memory '{new_record.memory_id}': {new_record.content} ({new_record.memory_type})")
        self.repo.save_memory(new_record)
        self._emit_event("created", new_record.memory_id, new_record.memory_type, new_record.content, new_record.confidence, new_record.status, user_id)
        return new_record

    def create_memory_manual(self, req: MemoryCreateRequest, user_id: str = "local_user") -> MemoryRecord:
        """Create memory directly via REST API."""
        now = _utc_now_iso()
        mem_type = req.memory_type.value if hasattr(req.memory_type, "value") else str(req.memory_type)
        conf = req.confidence if req.confidence is not None else 0.95
        imp = req.importance if req.importance is not None else 0.80

        record = MemoryRecord(
            memory_id=f"mem_{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            memory_type=mem_type,
            content=req.content.strip(),
            source_type=MemorySource.EXPLICIT_USER.value,
            source_reference="manual_api",
            confidence=conf,
            importance=imp,
            evidence_count=1,
            created_at=now,
            updated_at=now,
            last_confirmed_at=now,
            expires_at=req.expires_at,
            status=MemoryStatus.ACTIVE.value,
        )
        self.repo.save_memory(record)
        self._emit_event("created", record.memory_id, record.memory_type, record.content, record.confidence, record.status, user_id)
        return record

    def update_memory(self, memory_id: str, req: MemoryUpdateRequest) -> Optional[MemoryRecord]:
        """Update existing memory via REST API."""
        updates: Dict[str, Any] = {}
        if req.content is not None:
            updates["content"] = req.content.strip()
        if req.memory_type is not None:
            updates["memory_type"] = req.memory_type.value if hasattr(req.memory_type, "value") else str(req.memory_type)
        if req.importance is not None:
            updates["importance"] = req.importance
        if req.confidence is not None:
            updates["confidence"] = req.confidence
        if req.status is not None:
            updates["status"] = req.status.value if hasattr(req.status, "value") else str(req.status)
        if req.expires_at is not None:
            updates["expires_at"] = req.expires_at

        updated = self.repo.update_memory(memory_id, updates)
        if updated:
            self._emit_event("updated", updated.memory_id, updated.memory_type, updated.content, updated.confidence, updated.status, updated.user_id)
        return updated

    def delete_memory(self, memory_id: str, soft: bool = True) -> bool:
        """Delete or archive memory."""
        mem = self.repo.get_memory(memory_id)
        success = self.repo.delete_memory(memory_id, soft=soft)
        if success and mem:
            self._emit_event("deleted", mem.memory_id, mem.memory_type, mem.content, mem.confidence, "archived" if soft else "deleted", mem.user_id)
        return success


    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        return self.repo.get_memory(memory_id)

    def list_memories(
        self,
        user_id: str = "local_user",
        memory_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[MemoryRecord]:
        return self.repo.list_memories(user_id=user_id, memory_type=memory_type, status=status)


memory_service = MemoryService()
