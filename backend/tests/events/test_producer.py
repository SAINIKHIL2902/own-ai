import json
from pathlib import Path
import pytest
from app.events.producer import EventProducer
from app.events.schemas import InteractionEvent
from app.storage.database import DatabaseManager
from app.storage.repositories import RawRepository


@pytest.mark.asyncio
async def test_producer_fallback_spooling(tmp_path):
    # Setup temporary database and spool directory
    db_file = tmp_path / "test_user.db"
    manager = DatabaseManager(db_path=db_file)
    repo = RawRepository(manager=manager)

    producer = EventProducer(
        bootstrap_servers="localhost:9092",
        topic="test.topic",
        enabled=False,  # Simulate Kafka disabled/unavailable
    )

    event = InteractionEvent(
        conversation_id="conv_test",
        message_id="msg_test",
        prompt="Explain Docker containers",
        response="Containers package software...",
        model="llama3.2:1b",
        latency_ms=115.2,
    )

    # Overwrite repo in producer
    from unittest.mock import patch
    with patch("app.events.producer.raw_repo", repo), \
         patch("app.events.producer.SPOOL_DIR", tmp_path / "events"):
        await producer.publish_interaction(event)

        # 1. Verify event stored in raw SQLite table
        assert repo.is_event_processed(event.event_id) is False
        assert repo.count_raw_events() == 1

        # 2. Verify file spooled locally
        spool_file = tmp_path / "events" / f"{event.event_id}.json"
        assert spool_file.exists()
        with open(spool_file, "r", encoding="utf-8") as f:
            spooled_data = json.load(f)
            assert spooled_data["event_id"] == event.event_id
            assert spooled_data["prompt"] == "Explain Docker containers"
