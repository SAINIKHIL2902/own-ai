import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.memory.models import MemoryStatus, MemoryType


@pytest.fixture
def client():
    return TestClient(app)


def test_memory_crud_api_lifecycle(client):
    # 1. Create a memory via POST /memory
    create_payload = {
        "content": "I prefer using FastAPI and SQLite for local services",
        "memory_type": "preference",
        "confidence": 0.95,
        "importance": 0.85,
    }
    post_res = client.post("/memory", json=create_payload)
    assert post_res.status_code == 201
    created_data = post_res.json()
    memory_id = created_data["memory_id"]
    assert created_data["content"] == create_payload["content"]
    assert created_data["memory_type"] == "preference"
    assert created_data["status"] == "active"

    # 2. Get single memory via GET /memory/{memory_id}
    get_res = client.get(f"/memory/{memory_id}")
    assert get_res.status_code == 200
    assert get_res.json()["memory_id"] == memory_id

    # 3. List memories via GET /memory
    list_res = client.get("/memory?type=preference")
    assert list_res.status_code == 200
    memories = list_res.json()
    assert any(m["memory_id"] == memory_id for m in memories)

    # 4. Update memory via PATCH /memory/{memory_id}
    patch_payload = {
        "content": "I prefer using FastAPI and PostgreSQL for enterprise services",
        "importance": 0.90,
    }
    patch_res = client.patch(f"/memory/{memory_id}", json=patch_payload)
    assert patch_res.status_code == 200
    updated_data = patch_res.json()
    assert updated_data["content"] == patch_payload["content"]
    assert updated_data["importance"] == 0.90

    # 5. Delete / soft archive memory via DELETE /memory/{memory_id}
    del_res = client.delete(f"/memory/{memory_id}?soft=true")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

    # Verify status changed to archived
    archived_res = client.get(f"/memory/{memory_id}")
    assert archived_res.status_code == 200
    assert archived_res.json()["status"] == MemoryStatus.ARCHIVED.value


def test_get_nonexistent_memory_returns_404(client):
    res = client.get("/memory/mem_nonexistent_12345")
    assert res.status_code == 404
