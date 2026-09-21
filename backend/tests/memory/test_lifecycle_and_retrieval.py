from datetime import datetime, timedelta, timezone
import pytest
from app.memory.models import MemoryRecord, MemoryStatus, MemoryType
from app.memory.repository import MemoryRepository
from app.memory.retriever import MemoryRetriever
from app.memory.scorer import MemoryScorer
from app.storage.database import DatabaseManager


@pytest.fixture
def temp_repo(tmp_path):
    db_file = tmp_path / "test_memory.db"
    mgr = DatabaseManager(db_path=db_file)
    return MemoryRepository(manager=mgr)


def test_duplicate_memory_detection(temp_repo):
    mem1 = MemoryRecord(
        memory_id="mem_1",
        user_id="local_user",
        memory_type=MemoryType.PREFERENCE.value,
        content="I prefer concise code without comments",
        source_type="explicit_user",
        confidence=0.95,
        importance=0.8,
    )
    temp_repo.save_memory(mem1)

    # Similar phrasing with high token overlap
    similar = temp_repo.find_similar_memory(
        "I prefer concise code without any comments",
        memory_type=MemoryType.PREFERENCE.value,
        threshold=0.6,
    )
    assert similar is not None
    assert similar.memory_id == "mem_1"

    # Completely different phrasing
    different = temp_repo.find_similar_memory(
        "I am studying machine learning algorithms",
        memory_type=MemoryType.PREFERENCE.value,
    )
    assert different is None


def test_expiration_and_stale_memory_exclusion(temp_repo):
    now = datetime.now(timezone.utc)
    expired_time = (now - timedelta(hours=2)).isoformat()
    future_time = (now + timedelta(days=7)).isoformat()

    expired_mem = MemoryRecord(
        memory_id="mem_expired",
        user_id="local_user",
        memory_type=MemoryType.CONTEXT.value,
        content="Working on deadline today",
        source_type="explicit_user",
        expires_at=expired_time,
        status="active",
    )
    active_mem = MemoryRecord(
        memory_id="mem_active",
        user_id="local_user",
        memory_type=MemoryType.PREFERENCE.value,
        content="Prefers Python 3.12 syntax",
        source_type="explicit_user",
        expires_at=future_time,
        status="active",
    )

    temp_repo.save_memory(expired_mem)
    temp_repo.save_memory(active_mem)

    # get_active_memories automatically detects expired entries and marks them stale
    active_records = temp_repo.get_active_memories()
    active_ids = [m.memory_id for m in active_records]

    assert "mem_active" in active_ids
    assert "mem_expired" not in active_ids

    # Verify DB was updated
    reloaded_expired = temp_repo.get_memory("mem_expired")
    assert reloaded_expired.status == MemoryStatus.STALE.value


def test_relevant_memory_retrieval_and_irrelevant_exclusion(temp_repo):
    retriever = MemoryRetriever(repository=temp_repo)

    mem_python = MemoryRecord(
        memory_id="mem_py",
        user_id="local_user",
        memory_type=MemoryType.PREFERENCE.value,
        content="I prefer Python typing and modern async patterns",
        source_type="explicit_user",
        confidence=0.95,
        importance=0.85,
    )
    mem_cooking = MemoryRecord(
        memory_id="mem_cook",
        user_id="local_user",
        memory_type=MemoryType.INTEREST.value,
        content="I love authentic Italian pasta recipes",
        source_type="explicit_user",
        confidence=0.95,
        importance=0.85,
    )

    temp_repo.save_memory(mem_python)
    temp_repo.save_memory(mem_cooking)

    # Query about Python
    retrieved = retriever.retrieve_relevant("How do I write an async context manager in Python?")
    retrieved_ids = [m.memory_id for m in retrieved]

    assert "mem_py" in retrieved_ids
    # Irrelevant cooking memory must be excluded by threshold
    assert "mem_cook" not in retrieved_ids


def test_top_k_retrieval_limit(temp_repo):
    retriever = MemoryRetriever(repository=temp_repo)

    # Insert 8 relevant memories
    for i in range(8):
        mem = MemoryRecord(
            memory_id=f"mem_{i}",
            user_id="local_user",
            memory_type=MemoryType.PREFERENCE.value,
            content=f"Docker container deployment rule number {i} for microservices",
            source_type="explicit_user",
            confidence=0.95,
            importance=0.5 + (i * 0.05),
        )
        temp_repo.save_memory(mem)

    # Retrieve with top_k=3
    results = retriever.retrieve_relevant("How should I configure Docker for microservices?", top_k=3)
    assert len(results) == 3

    # Ensure ranking is descending by score (higher importance memories first)
    assert results[0].memory_id in ["mem_7", "mem_6"]
