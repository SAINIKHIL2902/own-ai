import pytest
from app.memory.detector import MemoryDetector


@pytest.fixture
def detector():
    return MemoryDetector()


def test_explicit_preference_detection(detector):
    p1 = "I prefer concise answers with bullet points."
    res1 = detector.detect(p1)
    assert res1 is not None
    assert res1["category"] == "preference"
    assert "concise answers" in res1["content"].lower()

    p2 = "Remember that I usually prefer dark mode and clean syntax."
    res2 = detector.detect(p2)
    assert res2 is not None
    assert "clean syntax" in res2["content"].lower()

    p3 = "I don't like verbose boilerplates."
    res3 = detector.detect(p3)
    assert res3 is not None
    assert "verbose boilerplates" in res3["content"].lower()


def test_goal_detection(detector):
    p1 = "My goal is to learn Rust and distributed systems this year."
    res1 = detector.detect(p1)
    assert res1 is not None
    assert res1["category"] == "goal"
    assert "learn rust" in res1["content"].lower()

    p2 = "I want to learn how to build low-latency compilers."
    res2 = detector.detect(p2)
    assert res2 is not None
    assert res2["category"] == "goal"
    assert "compilers" in res2["content"].lower()


def test_interest_detection(detector):
    p1 = "I am interested in event-driven architectures with Apache Kafka."
    res1 = detector.detect(p1)
    assert res1 is not None
    assert res1["category"] == "interest"
    assert "kafka" in res1["content"].lower()

    p2 = "I like exploring deep learning and PyTorch."
    res2 = detector.detect(p2)
    assert res2 is not None
    assert res2["category"] == "interest"


def test_context_and_instruction_detection(detector):
    p1 = "I am currently working on an MLOps deployment pipeline."
    res1 = detector.detect(p1)
    assert res1 is not None
    assert res1["category"] == "context"

    p2 = "Always use Python type hints and Pydantic models."
    res2 = detector.detect(p2)
    assert res2 is not None
    assert res2["category"] == "instruction"

    p3 = "Never use deprecated functions."
    res3 = detector.detect(p3)
    assert res3 is not None
    assert res3["category"] == "instruction"


def test_normal_questions_do_not_create_memory(detector):
    questions = [
        "What is Kafka?",
        "How does Python memory management work?",
        "Explain APIs in simple terms.",
        "Can you help me debug this error?",
        "Why is SQLite single-writer?",
        "What are the best practices for Docker containers?",
        "Tell me a joke about programming.",
    ]
    for q in questions:
        assert detector.detect(q) is None, f"Question '{q}' should NOT trigger memory detection!"
