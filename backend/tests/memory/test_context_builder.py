from app.memory.context_builder import ContextBuilder
from app.memory.models import MemoryRecord, MemoryType


def test_context_builder_empty_memories():
    builder = ContextBuilder()
    messages = [{"role": "user", "content": "Hello!"}]
    res = builder.personalize_messages(messages, [])
    assert res == messages


def test_context_builder_injects_directive_and_prioritizes_current_request():
    builder = ContextBuilder()
    memories = [
        MemoryRecord(
            memory_id="mem_1",
            user_id="local_user",
            memory_type=MemoryType.PREFERENCE.value,
            content="User prefers concise answers",
            source_type="explicit_user",
            confidence=0.95,
            importance=0.8,
        ),
        MemoryRecord(
            memory_id="mem_2",
            user_id="local_user",
            memory_type=MemoryType.GOAL.value,
            content="Mastering Kubernetes architecture",
            source_type="explicit_user",
            confidence=0.95,
            importance=0.9,
        ),
    ]

    messages = [{"role": "user", "content": "Give me a detailed explanation with examples."}]
    personalized = builder.personalize_messages(messages, memories)

    # Must insert a system message with personal context
    assert len(personalized) == 2
    assert personalized[0]["role"] == "system"
    system_text = personalized[0]["content"]

    # Must contain the memories
    assert "User prefers concise answers" in system_text
    assert "Mastering Kubernetes architecture" in system_text

    # MANDATORY PRINCIPLE: Current request priority directive MUST be explicitly included!
    assert "The user's explicit instructions in the current conversation always supersede these background memories" in system_text

    # Original user message must remain intact
    assert personalized[1]["role"] == "user"
    assert personalized[1]["content"] == "Give me a detailed explanation with examples."


def test_context_builder_merges_with_existing_system_message():
    builder = ContextBuilder()
    memories = [
        MemoryRecord(
            memory_id="mem_1",
            user_id="local_user",
            memory_type=MemoryType.INSTRUCTION.value,
            content="Always use dark mode color palette in diagrams",
            source_type="explicit_user",
            confidence=0.95,
            importance=0.8,
        )
    ]

    messages = [
        {"role": "system", "content": "You are a helpful software engineering mentor."},
        {"role": "user", "content": "Draw an architecture diagram."},
    ]

    personalized = builder.personalize_messages(messages, memories)
    assert len(personalized) == 2
    assert personalized[0]["role"] == "system"
    assert "You are a helpful software engineering mentor." in personalized[0]["content"]
    assert "Always use dark mode color palette in diagrams" in personalized[0]["content"]
    assert "supersede these background memories" in personalized[0]["content"]
