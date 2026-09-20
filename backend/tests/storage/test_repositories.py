from app.storage.database import DatabaseManager
from app.storage.models import (
    ContextRecord,
    InterestRecord,
    PreferenceRecord,
    utc_now,
)
from app.storage.repositories import ProfileRepository, RawRepository


def test_raw_repository_messages_and_events(tmp_path):
    db_file = tmp_path / "test.db"
    manager = DatabaseManager(db_path=db_file)
    raw = RawRepository(manager=manager)

    # 1. Save conversation and message
    raw.save_conversation("conv_1", "Test Conversation")
    raw.save_message("msg_1", "conv_1", "user", "Hello world", "llama3.2:1b", 50.0)
    raw.save_message("msg_2", "conv_1", "assistant", "Hi there!", "llama3.2:1b", 120.0)

    messages = raw.get_messages("conv_1")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"

    # 2. Save raw event
    now = utc_now()
    inserted = raw.save_raw_event(
        event_id="evt_123",
        event_type="chat_interaction",
        timestamp=now,
        user_id="local-user",
        conversation_id="conv_1",
        message_id="msg_2",
        payload_json='{"test": true}',
    )
    assert inserted is True
    assert raw.count_raw_events() == 1
    assert raw.is_event_processed("evt_123") is False

    # Idempotent insert test (duplicate event_id)
    dup_inserted = raw.save_raw_event(
        event_id="evt_123",
        event_type="chat_interaction",
        timestamp=now,
        user_id="local-user",
        conversation_id="conv_1",
        message_id="msg_2",
        payload_json='{"test": true}',
    )
    assert dup_inserted is False
    assert raw.count_raw_events() == 1

    # Mark processed
    raw.mark_event_processed("evt_123")
    assert raw.is_event_processed("evt_123") is True


def test_profile_repository_derived_entities(tmp_path):
    db_file = tmp_path / "test_profile.db"
    manager = DatabaseManager(db_path=db_file)
    profile = ProfileRepository(manager=manager)

    now = utc_now()
    # 1. Upsert interest
    interest = InterestRecord(
        topic="Kafka",
        score=0.82,
        interaction_count=5,
        recent_interactions=3,
        first_seen=now,
        last_seen=now,
        updated_at=now,
    )
    profile.upsert_interest(interest)
    fetched_interest = profile.get_interest("Kafka")
    assert fetched_interest is not None
    assert fetched_interest.score == 0.82
    assert fetched_interest.interaction_count == 5

    # 2. Upsert preference
    pref = PreferenceRecord(
        preference="prefers_code",
        value="true",
        confidence=0.8,
        evidence_count=4,
        updated_at=now,
    )
    profile.upsert_preference(pref)
    fetched_pref = profile.get_preference("prefers_code")
    assert fetched_pref is not None
    assert fetched_pref.confidence == 0.8

    # 3. Upsert context
    ctx = ContextRecord(
        id="ctx_1",
        topic="Kafka",
        activity="learning",
        context="MLOps pipeline",
        status="current",
        created_at=now,
        updated_at=now,
    )
    profile.upsert_context(ctx)
    contexts = profile.get_contexts()
    assert len(contexts) == 1
    assert contexts[0]["topic"] == "Kafka"

    # 4. Stats
    profile.set_stat("total_interactions", "10")
    stats = profile.get_all_stats()
    assert stats["total_interactions"] == "10"

    # 5. Response Styles
    from app.storage.models import ResponseStyleRecord
    style = ResponseStyleRecord(
        style_key="step_by_step_code",
        style_name="Step-by-Step Code Walkthrough",
        description="Progressive steps with code",
        liked_count=3,
        disliked_count=0,
        affinity_score=1.0,
    )
    profile.upsert_response_style(style)
    fetched_style = profile.get_response_style("step_by_step_code")
    assert fetched_style is not None
    assert fetched_style.liked_count == 3
    assert fetched_style.affinity_score == 1.0

    # Full profile
    full = profile.get_full_profile()
    assert full["schema_version"] == 1
    assert len(full["interests"]) == 1
    assert len(full["preferences"]) == 1
    assert len(full["response_styles"]) == 1


def test_assistant_response_feedback_correlation(tmp_path):
    db_file = tmp_path / "test_feedback.db"
    manager = DatabaseManager(db_path=db_file)
    raw = RawRepository(manager=manager)

    # 1. Store conversation, user message, and assistant response
    raw.save_conversation("conv_42", "Pytest Query")
    raw.save_message("msg_user_1", "conv_42", "user", "How do I use pytest?", "llama3.2:1b", 0.0, user_id="local_user")
    raw.save_message("msg_asst_1", "conv_42", "assistant", "Use pytest test_file.py", "llama3.2:1b", 150.0, user_id="local_user")

    asst_msg = raw.get_message_by_id("msg_asst_1")
    assert asst_msg is not None
    assert asst_msg["role"] == "assistant"
    assert asst_msg["user_id"] == "local_user"
    assert asst_msg["content"] == "Use pytest test_file.py"

    # 2. Store feedback associated directly with assistant response message_id
    raw.save_feedback(
        fb_id="fb_101",
        message_id="msg_asst_1",
        conversation_id="conv_42",
        feedback_type="thumbs_up",
        feedback_value="positive",
        rating=5,
        metadata_json='{"style": "concise_answer"}',
    )

    fb = raw.get_feedback_by_message_id("msg_asst_1")
    assert fb is not None
    assert fb["message_id"] == "msg_asst_1"
    assert fb["feedback_type"] == "thumbs_up"
    assert fb["feedback_value"] == "positive"
    assert fb["rating"] == 5


def test_phase3_response_and_feedback_storage_schema(tmp_path):
    """Explicitly verify Phase 3 prompt requirements for messages and feedback tables."""
    db_file = tmp_path / "test_phase3.db"
    manager = DatabaseManager(db_path=db_file)
    raw = RawRepository(manager=manager)

    with manager.get_connection() as conn:
        # 1. Verify messages table columns: message_id, conversation_id, user_id, role, content, model, created_at, latency_ms
        msg_cols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
        expected_msg_cols = {
            "message_id",
            "conversation_id",
            "user_id",
            "role",
            "content",
            "model",
            "created_at",
            "latency_ms",
        }
        assert expected_msg_cols.issubset(msg_cols), f"Missing msg cols: {expected_msg_cols - msg_cols}"

        # 2. Verify feedback table columns: feedback_id, message_id, conversation_id, user_id, feedback_type, feedback_value, created_at, metadata_json
        fb_cols = {r["name"] for r in conn.execute("PRAGMA table_info(feedback)").fetchall()}
        expected_fb_cols = {
            "feedback_id",
            "message_id",
            "conversation_id",
            "user_id",
            "feedback_type",
            "feedback_value",
            "created_at",
            "metadata_json",
        }
        assert expected_fb_cols.issubset(fb_cols), f"Missing fb cols: {expected_fb_cols - fb_cols}"

    # 3. Store conversation -> messages (user message & assistant message)
    raw.save_conversation("conv_p3", "Phase 3 Conversation")
    raw.save_message(
        msg_id="msg_user_p3",
        conversation_id="conv_p3",
        role="user",
        content="Show me how to store feedback",
        model="llama3.2:1b",
        latency_ms=0.0,
        user_id="local_user",
    )
    raw.save_message(
        msg_id="msg_123",
        conversation_id="conv_p3",
        role="assistant",
        content="Here is the response architecture...",
        model="llama3.2:1b",
        latency_ms=120.5,
        user_id="local_user",
    )

    asst_msg = raw.get_message_by_id("msg_123")
    assert asst_msg is not None
    assert asst_msg["message_id"] == "msg_123"
    assert asst_msg["role"] == "assistant"
    assert asst_msg["latency_ms"] == 120.5

    # 4. User clicks 👍 -> store feedback with message_id="msg_123", feedback_type="thumbs_up"
    raw.save_feedback(
        fb_id="fb_p3_01",
        message_id="msg_123",
        conversation_id="conv_p3",
        feedback_type="thumbs_up",
        feedback_value="positive",
    )

    fb = raw.get_feedback_by_message_id("msg_123")
    assert fb is not None
    assert fb["message_id"] == "msg_123"
    assert fb["feedback_type"] == "thumbs_up"
    assert fb["feedback_id"] == "fb_p3_01"


