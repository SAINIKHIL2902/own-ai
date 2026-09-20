from app.events.schemas import FeedbackEvent, InteractionEvent


def test_interaction_event_defaults():
    event = InteractionEvent(
        conversation_id="conv_1",
        message_id="msg_1",
        prompt="Explain Kafka partitions",
        response="Kafka partitions are...",
        model="llama3.2:1b",
        latency_ms=120.5,
    )
    assert event.event_id.startswith("evt_")
    assert event.event_type == "chat_interaction"
    assert event.user_id == "local-user"
    assert event.prompt == "Explain Kafka partitions"
    assert event.model == "llama3.2:1b"
    assert event.latency_ms == 120.5

    data = event.model_dump()
    assert "timestamp" in data
    assert data["conversation_id"] == "conv_1"


def test_feedback_event_defaults():
    event = FeedbackEvent(
        message_id="msg_1",
        conversation_id="conv_1",
        feedback_type="positive",
        rating=5,
        comment="Great explanation!",
    )
    assert event.event_id.startswith("fb_")
    assert event.event_type == "user_feedback"
    assert event.feedback_type == "positive"
    assert event.rating == 5
    assert event.comment == "Great explanation!"
