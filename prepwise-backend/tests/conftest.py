import pytest
import requests
import os
import time

FASTAPI_BASE = "http://localhost:8000"


@pytest.fixture(scope="session")
def session_id():
    """Upload Akshit's resume once, reuse session across all tests."""
    resume_path = os.path.join(os.path.dirname(__file__), "..", "test_resume.pdf")

    if not os.path.exists(resume_path):
        pytest.skip("test_resume.pdf not found in project root")

    # upload
    with open(resume_path, "rb") as f:
        res = requests.post(f"{FASTAPI_BASE}/upload", files={"file": ("test_resume.pdf", f, "application/pdf")})
    assert res.status_code == 200
    sid = res.json()["session_id"]

    # embed
    res = requests.post(f"{FASTAPI_BASE}/embed", params={"session_id": sid})
    assert res.status_code == 200

    time.sleep(1)
    return sid


@pytest.fixture(scope="session")
def retrieve(session_id):
    """Helper — retrieve chunks for a topic."""
    def _retrieve(topic: str, top_k: int = 3):
        res = requests.get(
            f"{FASTAPI_BASE}/retrieve",
            params={"session_id": session_id, "q": topic}
        )
        assert res.status_code == 200
        return res.json()
    return _retrieve


@pytest.fixture(scope="session")
def generate(session_id):
    """Helper — generate questions for a topic."""
    def _generate(topic: str):
        res = requests.get(
            f"{FASTAPI_BASE}/generate-questions",
            params={"session_id": session_id, "topic": topic}
        )
        if res.status_code != 200:
            print(f"\nERROR {res.status_code}: {res.text}")
        assert res.status_code == 200
        return res.json()
    return _generate