import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.database import db_manager
from app.storage.repositories import profile_repo


@pytest.fixture(autouse=True)
def clean_skills():
    db_manager.init_schema()
    with db_manager.get_connection() as conn:
        conn.execute("DELETE FROM user_skills")
        conn.commit()
    yield
    with db_manager.get_connection() as conn:
        conn.execute("DELETE FROM user_skills")
        conn.commit()


def test_add_and_get_skill():
    client = TestClient(app)

    # 1. Get empty skills
    res = client.get("/api/skills")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 0
    assert data["skills"] == []

    # 2. Add single skill
    payload = {
        "skill_name": "Python",
        "level": "Expert",
        "category": "Programming",
        "years_experience": 5.0,
    }
    post_res = client.post("/api/skills", json=payload)
    assert post_res.status_code == 200
    post_data = post_res.json()
    assert post_data["success"] is True

    # 3. Verify in repository and API
    get_res = client.get("/api/skills")
    assert get_res.status_code == 200
    skills = get_res.json()["skills"]
    assert len(skills) == 1
    assert skills[0]["skill_name"] == "Python"
    assert skills[0]["level"] == "Expert"
    assert skills[0]["category"] == "Programming"


def test_bulk_upload_skills():
    client = TestClient(app)

    # 1. Bulk upload from text
    payload = {
        "raw_text": "FastAPI, Docker, PostgreSQL, React",
        "default_level": "Advanced",
        "default_category": "Frameworks",
    }
    upload_res = client.post("/api/skills/upload", json=payload)
    assert upload_res.status_code == 200
    data = upload_res.json()
    assert data["success"] is True
    assert data["imported_count"] == 4

    # 2. Verify all are in DB
    get_res = client.get("/api/skills")
    assert get_res.status_code == 200
    skills = get_res.json()["skills"]
    assert len(skills) == 4
    names = {s["skill_name"] for s in skills}
    assert {"FastAPI", "Docker", "PostgreSQL", "React"}.issubset(names)


def test_delete_skill():
    client = TestClient(app)

    # Add skill
    client.post("/api/skills", json={"skill_name": "Kubernetes", "level": "Intermediate", "category": "DevOps"})
    assert len(profile_repo.get_all_skills()) == 1

    # Delete skill
    del_res = client.delete("/api/skills/Kubernetes")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # Verify deleted
    assert len(profile_repo.get_all_skills()) == 0
