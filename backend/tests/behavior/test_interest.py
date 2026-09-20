from app.behavior.interest import extract_topics


def test_extract_explicit_programming_topics():
    topics = extract_topics("How do I write an async function in Python?")
    assert "Python" in topics


def test_extract_framework_and_infrastructure_topics():
    topics = extract_topics("Explain how Kafka topic partitions scale across brokers.")
    assert "Kafka" in topics


def test_extract_multiple_topics():
    topics = extract_topics("Can I use FastAPI with Docker and SQLite for MLOps?")
    assert "FastAPI" in topics
    assert "Docker" in topics
    assert "SQLite" in topics
    assert "MLOps" in topics


def test_extract_fallback_general_qa():
    topics = extract_topics("What is the capital of France?")
    assert "General Q&A" in topics
