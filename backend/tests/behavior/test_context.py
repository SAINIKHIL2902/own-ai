from app.behavior.context import extract_contexts


def test_extract_learning_for_project_context():
    prompt = "I am currently learning Kafka for my MLOps project."
    contexts = extract_contexts(prompt)
    assert len(contexts) >= 1
    ctx = contexts[0]
    assert ctx.topic == "Kafka"
    assert ctx.activity == "learning"
    assert "MLOps project" in ctx.context
    assert ctx.status == "current"
    assert ctx.created_at is not None


def test_extract_building_context():
    prompt = "I'm building an automated trading bot using Python."
    contexts = extract_contexts(prompt)
    assert len(contexts) >= 1
    ctx = contexts[0]
    assert ctx.activity == "building"
    assert "automated trading bot" in ctx.context
