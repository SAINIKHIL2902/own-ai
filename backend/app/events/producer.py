import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Optional
from aiokafka import AIOKafkaProducer

from app.config.settings import settings
from app.events.schemas import FeedbackEvent, InteractionEvent, MemoryEvent
from app.storage.repositories import raw_repo

logger = logging.getLogger(__name__)

SPOOL_DIR = Path(settings.resolved_db_path).parent / "events"


class EventProducer:
    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        self.bootstrap_servers = bootstrap_servers or settings.KAFKA_BOOTSTRAP_SERVERS
        self.topic = topic or settings.KAFKA_TOPIC_INTERACTIONS
        self.enabled = enabled if enabled is not None else settings.KAFKA_ENABLED
        self._producer: Optional[AIOKafkaProducer] = None
        self._is_connected = False
        SPOOL_DIR.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Start Kafka producer if enabled. Gracefully handles startup failure."""
        if not self.enabled:
            logger.info("Kafka producer is disabled via configuration.")
            return

        try:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=5000,
            )
            # Timeout quickly on connect attempt so server startup is not blocked
            await asyncio.wait_for(self._producer.start(), timeout=3.0)
            self._is_connected = True
            logger.info(f"Connected to Kafka broker at {self.bootstrap_servers}")
        except Exception as exc:
            self._is_connected = False
            self._producer = None
            logger.warning(
                f"Kafka broker not reachable at {self.bootstrap_servers} ({exc}). "
                "Events will be safely spooled to local storage."
            )

    async def stop(self) -> None:
        if self._producer and self._is_connected:
            try:
                await self._producer.stop()
            except Exception as e:
                logger.warning(f"Error closing Kafka producer: {e}")
            self._is_connected = False
            self._producer = None

    def _spool_locally(self, event_dict: dict) -> None:
        """Fallback when Kafka is unavailable: write event to local spool file."""
        try:
            event_id = event_dict.get("event_id", "evt_unknown")
            SPOOL_DIR.mkdir(parents=True, exist_ok=True)
            file_path = SPOOL_DIR / f"{event_id}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(event_dict, f, indent=2)
            logger.debug(f"Event {event_id} spooled locally to {file_path}")
        except Exception as e:
            logger.error(f"Failed to spool event locally: {e}")

    async def publish_interaction(self, event: InteractionEvent) -> None:
        """Publish interaction event to Kafka and record in raw SQLite."""
        event_dict = event.model_dump()
        payload_json = json.dumps(event_dict)

        # 1. Always record in SQLite raw_events for permanent audit/replay
        try:
            raw_repo.save_raw_event(
                event_id=event.event_id,
                event_type=event.event_type,
                timestamp=event.timestamp,
                user_id=event.user_id,
                conversation_id=event.conversation_id,
                message_id=event.message_id,
                payload_json=payload_json,
            )
        except Exception as e:
            logger.error(f"Failed to persist raw event to SQLite: {e}")

        # 2. Publish to Kafka or fallback to local file spool
        if self._producer and self._is_connected:
            try:
                await self._producer.send_and_wait(
                    self.topic,
                    key=event.user_id.encode("utf-8"),
                    value=event_dict,
                )
                logger.info(f"Published event {event.event_id} to Kafka topic '{self.topic}'")
                return
            except Exception as exc:
                logger.warning(f"Kafka send failed for event {event.event_id}: {exc}. Spooling locally.")
                self._is_connected = False

        self._spool_locally(event_dict)

    async def publish_feedback(self, event: FeedbackEvent) -> None:
        """Publish feedback event and record in SQLite."""
        event_dict = event.model_dump()
        payload_json = json.dumps(event_dict)

        try:
            raw_repo.save_feedback(
                fb_id=event.event_id,
                message_id=event.message_id,
                conversation_id=event.conversation_id,
                feedback_type=event.feedback_type,
                feedback_value=event.feedback_value,
                rating=event.rating,
                comment=event.comment,
                metadata_json=json.dumps(event.metadata) if event.metadata else None,
            )
            raw_repo.save_raw_event(
                event_id=event.event_id,
                event_type=event.event_type,
                timestamp=event.timestamp,
                user_id=event.user_id,
                conversation_id=event.conversation_id or "feedback",
                message_id=event.message_id,
                payload_json=payload_json,
            )
        except Exception as e:
            logger.error(f"Failed to persist feedback event: {e}")

        if self._producer and self._is_connected:
            try:
                await self._producer.send_and_wait(
                    self.topic,
                    key=event.user_id.encode("utf-8"),
                    value=event_dict,
                )
                logger.info(f"Published feedback event {event.event_id} to Kafka")
                return
            except Exception as exc:
                logger.warning(f"Kafka feedback send failed: {exc}. Spooling locally.")
                self._is_connected = False

        self._spool_locally(event_dict)

    async def publish_memory_event(self, event: MemoryEvent) -> None:
        """Publish memory lifecycle event and record in raw SQLite."""
        event_dict = event.model_dump()
        payload_json = json.dumps(event_dict)

        try:
            raw_repo.save_raw_event(
                event_id=event.event_id,
                event_type=event.event_type,
                timestamp=event.timestamp,
                user_id=event.user_id,
                conversation_id="memory",
                message_id=event.memory_id,
                payload_json=payload_json,
            )
        except Exception as e:
            logger.error(f"Failed to persist memory event to SQLite: {e}")

        if self._producer and self._is_connected:
            try:
                await self._producer.send_and_wait(
                    self.topic,
                    key=event.user_id.encode("utf-8"),
                    value=event_dict,
                )
                logger.info(f"Published memory event {event.event_id} to Kafka")
                return
            except Exception as exc:
                logger.warning(f"Kafka memory send failed: {exc}. Spooling locally.")
                self._is_connected = False

        self._spool_locally(event_dict)


event_producer = EventProducer()

