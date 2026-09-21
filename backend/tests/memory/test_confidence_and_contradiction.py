import pytest
from app.memory.confidence import ConfidenceManager
from app.memory.models import MemoryRecord, MemoryStatus, MemoryType


@pytest.fixture
def conf_mgr():
    return ConfidenceManager()


def test_initial_confidence(conf_mgr):
    assert conf_mgr.get_initial_confidence("explicit_user") == 0.95
    assert conf_mgr.get_initial_confidence("conversation") == 0.70
    assert conf_mgr.get_initial_confidence("behavior") == 0.60
    assert conf_mgr.get_initial_confidence("unknown") == 0.50


def test_confidence_asymptotic_reinforcement(conf_mgr):
    initial_conf = 0.70
    evidence = 1

    # Reinforce 3 times
    conf1, ev1 = conf_mgr.reinforce(initial_conf, evidence)
    assert conf1 > initial_conf
    assert ev1 == 2

    conf2, ev2 = conf_mgr.reinforce(conf1, ev1)
    assert conf2 > conf1
    assert ev2 == 3

    conf3, ev3 = conf_mgr.reinforce(conf2, ev2)
    assert conf3 > conf2
    assert ev3 == 4

    # Confidence must never exceed 1.0
    high_conf, _ = conf_mgr.reinforce(0.999, 10)
    assert high_conf <= 1.0


def test_contradiction_detection(conf_mgr):
    # Opposing pairs: concise vs detailed
    assert conf_mgr.is_contradiction("I prefer concise answers", "I prefer detailed explanations")
    assert conf_mgr.is_contradiction("Prefers brief explanations", "detailed step-by-step responses")

    # Dark mode vs light mode
    assert conf_mgr.is_contradiction("Always use dark mode", "I like light mode")

    # Fast vs thorough
    assert conf_mgr.is_contradiction("Prefers fast responses", "Wants thorough, deep answers")

    # Non-contradictory statements
    assert not conf_mgr.is_contradiction("I prefer Python", "I like TypeScript")
    assert not conf_mgr.is_contradiction("I prefer concise answers", "I prefer bullet points")


def test_contradiction_reconciliation(conf_mgr):
    existing = MemoryRecord(
        memory_id="mem_123",
        user_id="local_user",
        memory_type=MemoryType.PREFERENCE.value,
        content="I prefer concise answers without preamble",
        source_type="explicit_user",
        confidence=0.95,
        importance=0.85,
        evidence_count=3,
        status=MemoryStatus.ACTIVE.value,
    )

    new_content = "I prefer detailed explanations with extensive code examples"
    reconciled = conf_mgr.reconcile_contradiction(existing, new_content, "explicit_user")

    # Existing memory must be updated with the latest preference
    assert reconciled.memory_id == "mem_123"
    assert reconciled.content == new_content
    # Evidence count incremented to preserve historical provenance
    assert reconciled.evidence_count == 4
    # Confidence remains active and calibrated
    assert 0.80 <= reconciled.confidence <= 0.95
    assert reconciled.status == MemoryStatus.ACTIVE.value
